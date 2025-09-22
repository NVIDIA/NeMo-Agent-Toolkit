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

import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def first_valid_query(series):
    # Apply extract_user_query to each value
    extracted_values = series.apply(extract_user_query)

    # Return the first non-None result
    # If there are multiple non-None results, return the first one
    # Otherwise, we return the trace_id instead.
    non_none_results = extracted_values.dropna()
    if not non_none_results.empty:
        return non_none_results.iloc[0]

    return None


def extract_user_query(input_value):
    if pd.isna(input_value):
        return None
    try:
        # Try to parse as JSON
        data = json.loads(input_value.replace('""', '"'))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("type") == "human":
                    content = item.get("content", "")
                    # Extract the actual query which is often at the end
                    return content.strip()
            return input_value[:20] + "..."
        elif isinstance(data, dict) and "input_message" in data:
            return data.get("input_message")
        else:
            return input_value[:20] + "..."
    except Exception as e:
        logger.warning("Error extracting user query: %s", e)
        return input_value[:20] + "..."
