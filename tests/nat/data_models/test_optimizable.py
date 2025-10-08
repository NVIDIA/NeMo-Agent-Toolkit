# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest import mock

import pytest
from pydantic import BaseModel

from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import OptimizableMixin
from nat.data_models.optimizable import SearchSpace


class TestSearchSpaceSuggest:

    def test_prompt_not_supported(self):
        space = SearchSpace(low=0, high=1, is_prompt=True)
        trial = mock.MagicMock()

        with pytest.raises(ValueError, match="Prompt optimization not currently supported"):
            space.suggest(trial, name="x")

    def test_categorical_choice(self):
        space = SearchSpace(values=["a", "b", "c"])
        trial = mock.MagicMock()
        trial.suggest_categorical.return_value = "b"

        result = space.suggest(trial, name="category")

        assert result == "b"
        trial.suggest_categorical.assert_called_once_with("category", ["a", "b", "c"])

    def test_integer_range(self):
        space = SearchSpace(low=1, high=9, log=True, step=2)
        trial = mock.MagicMock()
        trial.suggest_int.return_value = 5

        result = space.suggest(trial, name="int_param")

        assert result == 5
        trial.suggest_int.assert_called_once_with("int_param", 1, 9, log=True, step=2)

    def test_float_range(self):
        space = SearchSpace(low=0.1, high=1.0, log=False, step=0.1)
        trial = mock.MagicMock()
        trial.suggest_float.return_value = 0.4

        result = space.suggest(trial, name="float_param")

        assert result == 0.4
        trial.suggest_float.assert_called_once_with("float_param", 0.1, 1.0, log=False, step=0.1)


class TestOptimizableField:

    def test_basic_metadata_added(self):
        space = SearchSpace(low=0, high=10)

        class M(BaseModel):
            x: int = OptimizableField(5, space=space)

        extras = dict(M.model_fields)["x"].json_schema_extra
        assert extras["optimizable"] is True
        assert extras["search_space"] is space

    def test_space_optional(self):

        class M(BaseModel):
            x: int = OptimizableField(5)

        extras = dict(M.model_fields)["x"].json_schema_extra
        assert extras["optimizable"] is True
        assert "search_space" not in extras

    def test_preserves_user_extras_and_merges(self):
        space = SearchSpace(values=["red", "blue"])

        class M(BaseModel):
            x: str = OptimizableField(
                "red",
                space=space,
                json_schema_extra={
                    "note": "keep this", "another": 123
                },
            )

        extras = dict(M.model_fields)["x"].json_schema_extra
        assert extras["optimizable"] is True
        assert extras["search_space"] is space
        assert extras["note"] == "keep this"
        assert extras["another"] == 123

    def test_merge_conflict_overwrite(self):
        space = SearchSpace(low=0, high=1)
        user_space = "user"

        class M(BaseModel):
            x: int = OptimizableField(
                0,
                space=space,
                merge_conflict="overwrite",
                json_schema_extra={
                    "optimizable": False, "search_space": user_space
                },
            )

        extras = dict(M.model_fields)["x"].json_schema_extra
        assert extras["optimizable"] is True
        assert extras["search_space"] is space

    def test_merge_conflict_keep(self):
        space = SearchSpace(low=0, high=1)
        user_space = "user"

        class M(BaseModel):
            x: int = OptimizableField(
                0,
                space=space,
                merge_conflict="keep",
                json_schema_extra={
                    "optimizable": False, "search_space": user_space
                },
            )

        extras = dict(M.model_fields)["x"].json_schema_extra
        assert extras["optimizable"] is False
        assert extras["search_space"] == user_space

    def test_merge_conflict_error(self):
        space = SearchSpace(low=0, high=1)

        with pytest.raises(ValueError) as err:
            _ = type(
                "M",
                (BaseModel, ),
                {
                    "x":
                        OptimizableField(
                            0,
                            space=space,
                            merge_conflict="error",
                            json_schema_extra={
                                "optimizable": False, "search_space": "user"
                            },
                        )
                },
            )

        assert "optimizable" in str(err.value)
        assert "search_space" in str(err.value)

    def test_json_schema_extra_type_validation(self):
        space = SearchSpace(low=0, high=1)

        with pytest.raises(TypeError, match="json_schema_extra.*mapping"):
            _ = type(
                "M",
                (BaseModel, ),
                {
                    "x":
                        OptimizableField(
                            0,
                            space=space,
                            json_schema_extra=["not", "a", "dict"],  # type: ignore[arg-type]
                        )
                },
            )


class TestOptimizableMixin:

    def test_default_and_assignment(self):

        class MyModel(OptimizableMixin):
            a: int = 1

        m = MyModel()
        assert m.optimizable_params == []
        assert m.search_space == {}

        m2 = MyModel(optimizable_params=["a"], search_space={"a": SearchSpace(low=0, high=1)})
        assert m2.optimizable_params == ["a"]
        assert "a" in m2.search_space and m2.search_space["a"].low == 0

    def test_schema_contains_description(self):

        class MyModel(OptimizableMixin):
            a: int = 1

        schema = MyModel.model_json_schema()
        field = schema["properties"]["optimizable_params"]
        assert field["type"] == "array"
        assert field["description"] == "List of parameters that can be optimized."
