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

import dataclasses

from pydantic import BaseModel

from nat.utils.string_utils import convert_to_str


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
