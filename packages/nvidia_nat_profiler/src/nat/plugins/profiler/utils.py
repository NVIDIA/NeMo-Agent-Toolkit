# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

import pandas as pd

# moved to core; re-exported so existing imports keep working
from nat.builder.framework_instrumentation import _FRAMEWORK_REGEX_MAP  # noqa: F401
from nat.builder.framework_instrumentation import detect_llm_frameworks_in_build_fn  # noqa: F401
from nat.data_models.intermediate_step import IntermediateStep
from nat.plugins.profiler.data_frame_row import DataFrameRow

logger = logging.getLogger(__name__)


def create_standardized_dataframe(requests_data: list[list[IntermediateStep]]) -> pd.DataFrame:
    """
    Merge usage stats for *all* requests into one DataFrame, each row representing a usage_stats entry.
    - Include a column 'example_number' to mark which request it originated from.
    """
    all_rows = []
    try:
        for i, steps in enumerate(requests_data):
            for step in steps:
                # Create a DataFrameRow
                all_rows.append(
                    DataFrameRow(event_timestamp=step.event_timestamp,
                                 example_number=i,
                                 prompt_tokens=step.token_usage.prompt_tokens,
                                 completion_tokens=step.token_usage.completion_tokens,
                                 total_tokens=step.token_usage.total_tokens,
                                 llm_text_input=step.llm_text_input,
                                 llm_text_output=step.llm_text_output,
                                 llm_new_token=step.llm_text_chunk,
                                 llm_name=step.llm_name,
                                 tool_name=step.tool_name,
                                 function_name=step.function_name,
                                 function_id=step.function_id,
                                 parent_function_name=step.parent_function_name,
                                 parent_function_id=step.parent_function_id,
                                 UUID=step.payload.UUID,
                                 framework=step.framework,
                                 event_type=step.event_type).model_dump(), )

    except Exception as e:
        logger.exception("Error creating standardized DataFrame: %s", e)
        return pd.DataFrame()

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame.from_records(all_rows)
