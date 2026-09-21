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


def test_truncate_string_none_returns_none():
    assert truncate_string(None) is None


def test_truncate_string_shorter_than_limit_unchanged():
    assert truncate_string("hello", max_length=100) == "hello"


def test_truncate_string_equal_to_limit_unchanged():
    assert truncate_string("abcde", max_length=5) == "abcde"


def test_truncate_string_truncates_with_ellipsis():
    assert truncate_string("abcdefghij", max_length=5) == "ab..."


@pytest.mark.parametrize("max_length", [0, 1, 2, 3, 5, 8, 99])
def test_truncate_string_never_exceeds_max_length(max_length):
    # Regression: for max_length < 3 the old implementation used the negative
    # slice text[:max_length - 3], which cut characters off the end of the input
    # and returned a string *longer* than max_length (e.g. len 12 for max 2).
    result = truncate_string("abcdefghijklmnopqrstuvwxyz", max_length=max_length)
    assert result is not None
    assert len(result) <= max_length


def test_truncate_string_small_limits_return_partial_ellipsis():
    assert truncate_string("abcdefghij", max_length=2) == ".."
    assert truncate_string("abcdefghij", max_length=1) == "."
    assert truncate_string("abcdefghij", max_length=0) == ""
