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

from pydantic import BaseModel

from nat.profiler.parameter_optimization.update_helpers import apply_suggestions
from nat.profiler.parameter_optimization.update_helpers import nest_updates


def test_nest_updates_merges_nested_keys():
    flat = {"a.b.c": 1, "a.b.d": 2, "x": 9, "d.x": 3, "d.y.z": 4}
    nested = nest_updates(flat)

    assert nested == {
        "a": {
            "b": {
                "c": 1, "d": 2
            }
        },
        "x": 9,
        "d": {
            "x": 3, "y": {
                "z": 4
            }
        },
    }


def test_nest_updates_promotes_leaf_to_mapping_when_needed():
    # When both 'a' and 'a.b' are present, nested path should take precedence
    flat = {"a": 1, "a.b": 2}
    nested = nest_updates(flat)

    assert nested == {"a": {"b": 2}}


class Child(BaseModel):
    foo: int = 0
    bar: str = "x"


class RootModel(BaseModel):
    child: Child = Child()
    settings: dict[str, Child] = {}
    flag: bool = False


def test_apply_suggestions_updates_nested_and_dicts_without_mutating_original():
    original = RootModel()

    # Apply nested updates to child and to dict-of-models under settings
    updated = apply_suggestions(
        original,
        {
            "child.foo": 42,
            "settings.user1.bar": "hello",
            "settings.user2.foo": 99,
            "flag": True,
        },
    )

    # Original should remain unchanged
    assert original is not updated
    assert original.child.foo == 0
    assert original.flag is False
    assert original.settings == {}

    # Updated reflects changes
    assert isinstance(updated, RootModel)
    assert updated.child.foo == 42
    # Unchanged sibling remains the same
    assert updated.child.bar == "x"

    # Dict-of-models created and populated
    assert "user1" in updated.settings and "user2" in updated.settings
    assert updated.settings["user1"].bar == "hello"
    assert updated.settings["user2"].foo == 99
    assert updated.flag is True
