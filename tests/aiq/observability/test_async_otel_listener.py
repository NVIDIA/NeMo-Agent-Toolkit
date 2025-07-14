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

from aiq.observability.async_otel_listener import merge_dicts


def test_merge_dicts_basic():
    """Test basic dictionary merging functionality."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2, "c": 4}


def test_merge_dicts_with_none_values():
    """Test merging dictionaries with None values."""
    dict1 = {"a": None, "b": 2, "c": None}
    dict2 = {"a": 1, "b": 3, "c": 4}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2, "c": 4}


def test_merge_dicts_empty_dicts():
    """Test merging empty dictionaries."""
    dict1 = {}
    dict2 = {}
    result = merge_dicts(dict1, dict2)
    assert result == {}


def test_merge_dicts_one_empty():
    """Test merging when one dictionary is empty."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2}

    dict1 = {}
    dict2 = {"a": 1, "b": 2}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2}


def test_merge_dicts_nested_values():
    """Test merging dictionaries with nested values."""
    dict1 = {"a": {"x": 1}, "b": None}
    dict2 = {"a": {"y": 2}, "b": {"z": 3}}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": {"x": 1}, "b": {"z": 3}}


def test_merge_dicts_complex_types():
    """Test merging dictionaries with complex types."""
    dict1 = {"a": [1, 2, 3], "b": None}
    dict2 = {"a": [4, 5, 6], "b": "test"}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": [1, 2, 3], "b": "test"}
