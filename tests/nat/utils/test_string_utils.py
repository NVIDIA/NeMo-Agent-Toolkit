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

import dataclasses

import pytest
from pydantic import BaseModel

from nat.utils.string_utils import convert_to_str
from nat.utils.string_utils import truncate_string


class _M(BaseModel):
    a: int
    b: str | None = None


def test_convert_to_str_primitives():
    assert convert_to_str("x") == "x"
    assert convert_to_str([1, 2, 3]) == "1, 2, 3"
    s = convert_to_str({"k": 1, "z": 2})
    assert (s.startswith("k: 1") or s.startswith("z: 2"))


def test_convert_to_str_object_with_str():

    @dataclasses.dataclass
    class C:
        x: int

        def __str__(self):
            return f"C({self.x})"

    assert convert_to_str(C(3)) == "C(3)"


def test_convert_to_str_pydantic_model():
    """Test convert_to_str with Pydantic BaseModel."""
    model = _M(a=42, b="test")
    result = convert_to_str(model)
    assert '"a":42' in result
    assert '"b":"test"' in result


def test_convert_to_str_pydantic_model_excludes_none():
    """Test that Pydantic model serialization excludes None values."""
    model = _M(a=42)
    result = convert_to_str(model)
    assert '"a":42' in result
    assert '"b"' not in result


def test_convert_to_str_empty_list():
    """Test convert_to_str with empty list."""
    assert convert_to_str([]) == ""


def test_convert_to_str_empty_dict():
    """Test convert_to_str with empty dictionary."""
    assert convert_to_str({}) == ""


def test_convert_to_str_nested_list():
    """Test convert_to_str with nested structures in list."""
    result = convert_to_str([[1, 2], [3, 4]])
    assert "[1, 2]" in result
    assert "[3, 4]" in result


def test_convert_to_str_numeric_types():
    """Test convert_to_str with various numeric types."""
    assert convert_to_str(42) == "42"
    assert convert_to_str(3.14) == "3.14"
    assert convert_to_str(True) == "True"


class TestTruncateString:
    """Tests for truncate_string function."""

    def test_truncate_none_input(self):
        """Test that None input returns None."""
        assert truncate_string(None) is None

    def test_truncate_empty_string(self):
        """Test that empty string returns empty string."""
        assert truncate_string("") == ""

    def test_truncate_short_string(self):
        """Test that strings shorter than max_length are not truncated."""
        text = "Hello, World!"
        assert truncate_string(text, max_length=100) == text

    def test_truncate_exact_length(self):
        """Test string with exact max_length is not truncated."""
        text = "x" * 100
        assert truncate_string(text, max_length=100) == text

    def test_truncate_long_string(self):
        """Test that long strings are properly truncated with ellipsis."""
        text = "x" * 150
        result = truncate_string(text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")
        assert result == "x" * 97 + "..."

    def test_truncate_custom_max_length(self):
        """Test truncation with custom max_length."""
        text = "This is a test string"
        result = truncate_string(text, max_length=10)
        assert len(result) == 10
        assert result == "This is..."

    def test_truncate_very_short_max_length(self):
        """Test truncation with very short max_length."""
        text = "Hello"
        result = truncate_string(text, max_length=4)
        assert result == "H..."

    def test_truncate_preserves_type(self):
        """Test that truncate_string preserves string type."""
        result = truncate_string("test", max_length=100)
        assert isinstance(result, str)
