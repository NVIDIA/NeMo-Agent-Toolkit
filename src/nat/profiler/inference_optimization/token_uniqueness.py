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

import re

import numpy as np

from nat.data_models.intermediate_step import IntermediateStep
from nat.profiler.inference_optimization.data_models import LLMUniquenessMetrics
from nat.profiler.inference_optimization.data_models import LLMUniquenessMetricsByLLM
from nat.profiler.utils import create_standardized_dataframe


# ----------------------------------------------------------------
# 1. Main Function
# ----------------------------------------------------------------
def compute_inter_query_token_uniqueness_by_llm(all_steps: list[list[IntermediateStep]]) -> LLMUniquenessMetricsByLLM:
    """
    Computes p90, p95, and p99 of 'new words added' between consecutive llm_start events,
    grouped by (llm_name, example_number).

    Steps:

    1. Filter df to only llm_start events.
    2. Group first by (llm_name, example_number), then sort by event_timestamp in each group.
    3. Compare each llm_text_input to the previous one in the same group to find how many new words appear.
    4. Aggregate all 'new words count' across each llm_name, compute p90/p95/p99 for each LLM.
    5. Return a Pydantic RootModel containing a dictionary::

         { llm_name -> LLMUniquenessMetrics(p90, p95, p99) }.
    """
    df = create_standardized_dataframe(all_steps)
    # Validate that the necessary columns exist
    required_cols = {'event_type', 'llm_name', 'example_number', 'event_timestamp', 'llm_text_input'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # 1) Filter to llm_start events
    cdf = df[df['event_type'] == 'LLM_START'].copy()
    if cdf.empty:
        # Return an empty dictionary if no llm_start events
        return LLMUniquenessMetricsByLLM(root={})

    # Helper to tokenize text into a set of words
    def tokenize_to_set(text: str) -> set:
        if not isinstance(text, str):
            return set()
        return set(re.findall(r"\w+", text.lower()))

    # We'll store new_words counts for each llm_name
    llm_to_counts: dict[str, list[int]] = {}

    # 2) Group by (llm_name, example_number), then sort each group
    grouped = cdf.groupby(['llm_name', 'example_number'], as_index=False, group_keys=True)

    for (llm, ex_num), group_df in grouped:  # pylint: disable=unused-variable
        # Sort by event_timestamp
        group_df = group_df.sort_values('event_timestamp', ascending=True)

        # Shift the llm_text_input to compare consecutive calls
        group_df['prev_llm_text_input'] = group_df['llm_text_input'].shift(1)

        # Compute new words for each row (excluding the first in the group)
        def compute_new_words(row):
            current_tokens = tokenize_to_set(row['llm_text_input'])
            prev_tokens = tokenize_to_set(row['prev_llm_text_input'])
            return len(current_tokens - prev_tokens)

        group_df['new_words_count'] = group_df.apply(compute_new_words, axis=1)

        # Drop rows where there's no 'previous' call
        valid_rows = group_df.dropna(subset=['prev_llm_text_input'])

        # Gather the new_words_count
        counts = valid_rows['new_words_count'].tolist()
        if counts:
            # Accumulate them in llm_to_counts
            if llm not in llm_to_counts:
                llm_to_counts[llm] = []
            llm_to_counts[llm].extend(counts)

    # 4) For each llm_name, compute p90, p95, p99
    output_dict = {}
    for llm_name, counts_list in llm_to_counts.items():
        arr = np.array(counts_list)
        p90_val = float(np.percentile(arr, 90))
        p95_val = float(np.percentile(arr, 95))
        p99_val = float(np.percentile(arr, 99))

        output_dict[llm_name] = LLMUniquenessMetrics(p90=p90_val, p95=p95_val, p99=p99_val)

    ret_val = LLMUniquenessMetricsByLLM(root=output_dict)
    # Validate & return as a RootModel
    return ret_val
