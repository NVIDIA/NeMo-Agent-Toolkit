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

import fnmatch

import pandas as pd

from nat.data_models.dataset_handler import EvalFilterConfig


class DatasetFilter:
    """
    Apply allowlist and denylist filters to the DataFrame based on specified column filters.
        - If a allowlist is provided, only keep rows matching the filter values.
        - If a denylist is provided, remove rows matching the filter values.
        - If the filter column does not exist in the DataFrame, the filtering is skipped for that column.
        - Supports Unix shell-style wildcards (``*``, ``?``, ``[seq]``, ``[!seq]``) for string matching.

    This is a utility class that is dataset agnostic and can be used to filter any DataFrame based on the provided
    filter configuration.
    """

    def __init__(self, filter_config: EvalFilterConfig):

        self.filter_config = filter_config

    @staticmethod
    def _match_wildcard_patterns(series: pd.Series, patterns: list[str | int | float]) -> pd.Series:
        """
        Match series values against wildcard patterns and exact values.

        Args:
            series (pd.Series): pandas Series to match against
            patterns (list[str | int | float]): List of patterns/values

        Returns:
            pd.Series: Boolean Series indicating matches
        """
        # Convert series to string for pattern matching
        str_series = series.astype(str)

        # Initialize boolean mask
        matches = pd.Series([False] * len(series), index=series.index)

        # Check each pattern using fnmatch with list comprehension to avoid lambda capture
        for pattern in patterns:
            pattern_str = str(pattern)
            pattern_matches = pd.Series([fnmatch.fnmatch(val, pattern_str) for val in str_series],
                                        index=str_series.index)
            matches |= pattern_matches

        return matches

    def apply_filters(self, df) -> pd.DataFrame:

        filtered_df = df.copy()

        # Apply allowlist (only keep specified rows)
        if self.filter_config.allowlist:
            for column, values in self.filter_config.allowlist.field.items():
                if column in filtered_df.columns:
                    matches = self._match_wildcard_patterns(filtered_df[column], values)
                    filtered_df = filtered_df[matches]

        # Apply denylist (remove specified rows)
        if self.filter_config.denylist:
            for column, values in self.filter_config.denylist.field.items():
                if column in filtered_df.columns:
                    matches = self._match_wildcard_patterns(filtered_df[column], values)
                    filtered_df = filtered_df[~matches]

        return filtered_df
