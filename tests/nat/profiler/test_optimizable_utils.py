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

import logging

import pytest
from pydantic import BaseModel

from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import OptimizableMixin
from nat.data_models.optimizable import SearchSpace
from nat.profiler.parameter_optimization.optimizable_utils import walk_optimizables


class InnerModel(BaseModel):
    space_x: SearchSpace = SearchSpace(low=0, high=10)
    x: int = OptimizableField(1, space=space_x)
    y: str = "not_optimizable"


class RootModel(OptimizableMixin):
    space_z: SearchSpace = SearchSpace(low=0.0, high=1.0)
    inner: InnerModel = InnerModel()
    z: float = OptimizableField(0.5, space=space_z)
    mapping: dict[str, InnerModel] = {}


def test_walk_optimizables_honors_allowlist_and_nested():
    root = RootModel(optimizable_params=["z", "inner", "mapping"])  # allow traversal
    root.mapping = {"a": InnerModel(), "b": InnerModel()}

    spaces = walk_optimizables(root)

    # Top-level field
    assert "z" in spaces
    assert isinstance(spaces["z"], SearchSpace)
    assert spaces["z"].low == 0.0 and spaces["z"].high == 1.0

    # Nested field inside BaseModel
    assert "inner.x" in spaces
    assert spaces["inner.x"].low == 0 and spaces["inner.x"].high == 10

    # Dict[str, BaseModel] container traversal
    assert "mapping.a.x" in spaces
    assert "mapping.b.x" in spaces


def test_walk_optimizables_respects_allowlist_exclusions():
    # Exclude mapping from allowlist so it is not traversed
    root = RootModel(optimizable_params=["z", "inner"])  # mapping excluded
    root.mapping = {"a": InnerModel()}

    spaces = walk_optimizables(root)

    assert "z" in spaces
    assert "inner.x" in spaces
    assert not any(k.startswith("mapping.") for k in spaces.keys())


def test_walk_optimizables_warns_when_no_allowlist(caplog: pytest.LogCaptureFixture):

    class SimpleModel(BaseModel):
        a: int = OptimizableField(0, space=SearchSpace(low=0, high=5))

    model = SimpleModel()
    with caplog.at_level(logging.WARNING, logger="nat.profiler.parameter_optimization.optimizable_utils"):
        spaces = walk_optimizables(model)

    # Warning was emitted
    assert any("optimizable fields" in r.message for r in caplog.records)
    # Current behavior: fields are still returned (despite warning wording)
    assert "a" in spaces
    assert isinstance(spaces["a"], SearchSpace)


def test_walk_optimizables_uses_search_space_overrides():

    class MyModel(OptimizableMixin):
        a: float = 0.1

    cfg = MyModel(optimizable_params=["a"], search_space={"a": SearchSpace(low=0, high=1)})

    spaces = walk_optimizables(cfg)

    assert "a" in spaces
    assert spaces["a"].low == 0 and spaces["a"].high == 1


def test_walk_optimizables_requires_search_space():

    class MyModel(OptimizableMixin):
        a: int = OptimizableField(0)

    cfg = MyModel(optimizable_params=["a"])

    with pytest.raises(ValueError, match="no search space"):
        walk_optimizables(cfg)


def test_walk_optimizables_can_mark_without_space_in_code():

    class MyModel(OptimizableMixin):
        a: int = OptimizableField(0)

    cfg = MyModel(optimizable_params=["a"], search_space={"a": SearchSpace(low=0, high=1)})

    spaces = walk_optimizables(cfg)

    assert "a" in spaces and spaces["a"].low == 0


def test_static_type_fallback_for_dict_of_models():

    class Item(BaseModel):
        v: int = 1

    class Container(BaseModel):
        children: dict[str, Item]

    # Call with the class (type) to trigger the static-annotation path
    spaces = walk_optimizables(Container)

    # Sentinel entry for any key in the mapping
    assert "children.*" in spaces
    sentinel = spaces["children.*"]
    assert isinstance(sentinel, SearchSpace)
    assert sentinel.low is None and sentinel.high is None
