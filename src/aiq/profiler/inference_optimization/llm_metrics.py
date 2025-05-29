# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd

from aiq.data_models.intermediate_step import IntermediateStep
from aiq.profiler.utils import create_standardized_dataframe


class LLMMetrics:
    """
    A utility class for computing derived metrics on standardized LLM call logs.
    """

    @staticmethod
    def compute_profiling_metrics(all_steps: list[list[IntermediateStep]]) -> pd.DataFrame:
        """
        Compute and append the following columns to the provided DataFrame:

        1. NOVA-Event-ID (str):

           - The name of the calling function (`function_name`).

        2. NOVA-Requests-Remaining-In-Event (int):

           - For each row, how many future LLM_START events will occur (strictly after
             this row's event_timestamp) in the same (example_number, function_name).

        3. NOVA-Time-To-Next-Event (float):

           - For each row, the number of milliseconds until the next LLM_START event in
             the same (example_number, function_name). If no future event, set to -1.

        4. NOVA-Time-To-Event-End (float):

           - For each row, the number of milliseconds until the last future LLM_START
             event in the same (example_number, function_name). If no future event, set to -1.

        5. NOVA-Predicted-OSL (float or int):

           - For rows where event_type == 'LLM_START', this column will hold the
             `completion_tokens` of the corresponding LLM_END (matched by UUID). If no match,
             set to NaN (or another sentinel).

        6. NOVA-Time-To-Session-End (float):

           - For each row, the total milliseconds remaining in the workflow invocation,
             i.e. until the max event_timestamp within that example_number.

        Assumptions:

        - event_timestamp is an epoch timestamp in *seconds*.
        - Columns required in the input df (at minimum)::

            ['example_number', 'event_timestamp', 'event_type', 'function_name', 'UUID', 'completion_tokens']

        - 'LLM_START' / 'LLM_END' events share the same UUID.
        - The DataFrame may have additional columns such as 'llm_text_input', 'llm_text_output',
           'function_id', 'parent_function_name', 'parent_function_id', etc.

        :param all_steps: All intermediate steps for each example.
        :return:   The same DataFrame with the six NOVA- columns appended.
        """

        df = create_standardized_dataframe(all_steps)

        if df.empty:
            return df

        # ---------------------------------------------------------------------
        # 1. NOVA-Event-ID
        #    This is simply the function_name.
        # ---------------------------------------------------------------------
        df['NOVA-Event-ID'] = df['function_name']

        # ---------------------------------------------------------------------
        # 2. NOVA-Requests-Remaining-In-Event,
        # 3. NOVA-Time-To-Next-Event,
        # 4. NOVA-Time-To-Event-End
        #
        # We'll compute these by grouping on (example_number, function_name),
        # sorting by event_timestamp, and for each row calculating:
        #
        #  - how many LLM_START events lie strictly in the future,
        #  - the time to the next LLM_START event in the future,
        #  - the time to the last LLM_START event in the future.
        #
        # For times, we convert to milliseconds by multiplying by 1000,
        # assuming event_timestamp is in seconds.
        # ---------------------------------------------------------------------

        # Initialize columns with default values.
        df['NOVA-Requests-Remaining-In-Event'] = -1
        df['NOVA-Time-To-Next-Event'] = -1.0
        df['NOVA-Time-To-Event-End'] = -1.0

        def _compute_group_metrics(subdf: pd.DataFrame) -> pd.DataFrame:
            """
            For a sub-DataFrame with a unique (example_number, function_name),
            compute the requested columns for each row.
            """
            # Sort by time to ensure chronological order.
            subdf = subdf.sort_values('event_timestamp').copy()

            # Collect all LLM_START timestamps in this group as a sorted array.
            llm_start_mask = (subdf['event_type'] == 'LLM_START')
            llm_start_ts = subdf.loc[llm_start_mask, 'event_timestamp'].values

            # If no LLM_START events present, we can return immediately.
            if len(llm_start_ts) == 0:
                # No future LLM_START events to compute; everything stays default -1.
                return subdf

            def _rowwise_calc(row):
                """
                For each row, compute:
                  - how many LLM_START events lie strictly in the future,
                  - time to the next LLM_START event,
                  - time to the last LLM_START event (in the future).
                """
                row_ts = row['event_timestamp']

                # Use searchsorted to find how many LLM_START events lie after this row's timestamp.
                # side='right' means we treat any LLM_START at exactly row_ts as not 'in the future'.
                insertion_idx = np.searchsorted(llm_start_ts, row_ts, side='right')

                # (A) Requests remaining = how many LLM_START events are strictly after row_ts
                requests_remaining = len(llm_start_ts) - insertion_idx

                # (B) Time to next LLM_START (if any)
                if insertion_idx < len(llm_start_ts):
                    next_event_time = llm_start_ts[insertion_idx]
                    time_to_next_event = (next_event_time - row_ts) * 1000.0
                else:
                    time_to_next_event = -1.0

                # (C) Time to the last LLM_START in the future (if any).
                # The last LLM_START in the future is simply the last entry of llm_start_ts
                # if there's at least one future LLM_START. We'll check that it is strictly > row_ts.
                if requests_remaining > 0:
                    last_future_llm_start = llm_start_ts[-1]
                    # double-check that it's truly in the future
                    if last_future_llm_start > row_ts:
                        time_to_event_end = (last_future_llm_start - row_ts) * 1000.0
                    else:
                        time_to_event_end = -1.0
                else:
                    time_to_event_end = -1.0

                return pd.Series({
                    'NOVA-Requests-Remaining-In-Event': requests_remaining,
                    'NOVA-Time-To-Next-Event': time_to_next_event,
                    'NOVA-Time-To-Event-End': time_to_event_end
                })

            # Apply row-wise calculations
            metrics_df = subdf.apply(_rowwise_calc, axis=1)

            # Merge back into subdf
            subdf[['NOVA-Requests-Remaining-In-Event', 'NOVA-Time-To-Next-Event',
                   'NOVA-Time-To-Event-End']] = metrics_df

            return subdf

        # Apply the group metrics
        df_group = df.groupby(['example_number', 'function_name'], group_keys=False)
        df = df_group[df.columns].apply(_compute_group_metrics).sort_index()

        # ---------------------------------------------------------------------
        # 5. NOVA-Predicted-OSL
        #
        # For each LLM_START event, we want the completion_tokens from its
        # corresponding LLM_END event. Both share the same UUID.
        # We'll do a map from UUID -> completion_tokens for LLM_END rows.
        # ---------------------------------------------------------------------
        df['NOVA-Predicted-OSL'] = np.nan

        # Build a map of UUID -> completion_tokens from LLM_END
        llm_end_map = (df.loc[df['event_type'] == 'LLM_END', ['UUID', 'completion_tokens']].dropna(
            subset=['UUID']).set_index('UUID')['completion_tokens'].to_dict())

        # Only assign to rows which are LLM_START
        llm_start_mask = (df['event_type'] == 'LLM_START')
        df.loc[llm_start_mask, 'NOVA-Predicted-OSL'] = (df.loc[llm_start_mask, 'UUID'].map(llm_end_map))

        # ---------------------------------------------------------------------
        # 6. NOVA-Time-To-Session-End
        #
        # For each example_number, we want the difference (in ms) between
        # the row's event_timestamp and the final (max) event_timestamp
        # in that example_number.
        # ---------------------------------------------------------------------
        max_ts_per_example = (df.groupby('example_number')['event_timestamp'].transform('max'))

        # We'll subtract row's timestamp from the max, and convert to ms
        df['NOVA-Time-To-Session-End'] = (max_ts_per_example - df['event_timestamp']) * 1000.0

        # Return the updated DataFrame
        return df
