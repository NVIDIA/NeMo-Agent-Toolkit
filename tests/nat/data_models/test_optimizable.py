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
        space = SearchSpace(is_prompt=True, prompt="test prompt")
        trial = mock.MagicMock()

        with pytest.raises(ValueError, match="Prompt optimization not currently supported using Optuna"):
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


class TestSearchSpaceToGridValues:
    """Test SearchSpace.to_grid_values() for grid search."""

    def test_prompt_not_supported(self):
        space = SearchSpace(is_prompt=True, prompt="test prompt")
        with pytest.raises(ValueError, match="Prompt optimization not currently supported using Optuna"):
            space.to_grid_values()

    def test_explicit_values(self):
        space = SearchSpace(values=[0.1, 0.5, 0.9])
        result = space.to_grid_values()
        assert result == [0.1, 0.5, 0.9]

    def test_integer_range_with_step(self):
        space = SearchSpace(low=0, high=10, step=2)
        result = space.to_grid_values()
        assert result == [0, 2, 4, 6, 8, 10]

    def test_float_range_with_step(self):
        space = SearchSpace(low=0.0, high=1.0, step=0.25)
        result = space.to_grid_values()
        assert len(result) == 5
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)
        # Check intermediate values
        assert result[1] == pytest.approx(0.25)
        assert result[2] == pytest.approx(0.5)
        assert result[3] == pytest.approx(0.75)

    def test_range_without_step_raises_error(self):
        space = SearchSpace(low=0.1, high=0.9)
        with pytest.raises(ValueError, match="requires 'step' to be specified"):
            space.to_grid_values()

    def test_log_scale_not_supported_for_integer_ranges(self):
        space = SearchSpace(low=1, high=100, step=10, log=True)
        with pytest.raises(ValueError, match="Log scale is not supported for integer ranges"):
            space.to_grid_values()

    def test_log_scale_not_supported_for_float_ranges(self):
        space = SearchSpace(low=0.01, high=1.0, step=0.1, log=True)
        with pytest.raises(ValueError, match="Log scale is not yet supported for grid search"):
            space.to_grid_values()

    def test_missing_low_high_raises_error(self):
        space = SearchSpace(low=None, high=None)
        with pytest.raises(ValueError, match="requires either 'values' or both 'low' and 'high'"):
            space.to_grid_values()

    def test_categorical_values_returned_as_list(self):
        space = SearchSpace(values=["small", "medium", "large"])
        result = space.to_grid_values()
        assert result == ["small", "medium", "large"]

    def test_small_float_step(self):
        """Test with a small step size to ensure proper discretization."""
        space = SearchSpace(low=0.0, high=0.1, step=0.02)
        result = space.to_grid_values()
        assert len(result) == 6  # 0.0, 0.02, 0.04, 0.06, 0.08, 0.1
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(0.1)

    def test_integer_range_with_non_integral_step_returns_floats(self):
        """Test that non-integral step for integer range returns float values."""
        space = SearchSpace(low=0, high=10, step=1.5)
        result = space.to_grid_values()
        # Should get float values: 0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.0
        assert len(result) == 8
        assert all(isinstance(v, float) for v in result)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.5)
        assert result[-1] == pytest.approx(10.0)

    def test_integer_range_with_negative_step_raises_error(self):
        """Test that negative step raises clear error."""
        space = SearchSpace(low=0, high=10, step=-2)
        with pytest.raises(ValueError, match="Grid search step must be positive; got step=-2"):
            space.to_grid_values()

    def test_integer_range_with_zero_step_raises_error(self):
        """Test that zero step raises clear error."""
        space = SearchSpace(low=0, high=10, step=0)
        with pytest.raises(ValueError, match="Grid search step must be positive; got step=0"):
            space.to_grid_values()

    def test_integer_range_with_float_integral_step_works(self):
        """Test that integral step as float (e.g., 2.0) works correctly."""
        space = SearchSpace(low=0, high=10, step=2.0)
        result = space.to_grid_values()
        assert result == [0, 2, 4, 6, 8, 10]


class TestSearchSpaceValidation:
    """Test SearchSpace model validation at construction time."""

    def test_prompt_with_low_high_raises_error(self):
        """Test that is_prompt=True with low/high raises validation error."""
        with pytest.raises(ValueError, match="'is_prompt=True' cannot have 'low' or 'high' parameters"):
            SearchSpace(is_prompt=True, low=0, high=10)

    def test_prompt_with_only_low_raises_error(self):
        """Test that is_prompt=True with only low raises validation error."""
        with pytest.raises(ValueError, match="'is_prompt=True' cannot have 'low' or 'high' parameters"):
            SearchSpace(is_prompt=True, low=0)

    def test_prompt_with_only_high_raises_error(self):
        """Test that is_prompt=True with only high raises validation error."""
        with pytest.raises(ValueError, match="'is_prompt=True' cannot have 'low' or 'high' parameters"):
            SearchSpace(is_prompt=True, high=10)

    def test_prompt_with_log_raises_error(self):
        """Test that is_prompt=True with log=True raises validation error."""
        with pytest.raises(ValueError, match="'is_prompt=True' cannot have 'log=True'"):
            SearchSpace(is_prompt=True, log=True, prompt="test")

    def test_prompt_with_step_raises_error(self):
        """Test that is_prompt=True with step raises validation error."""
        with pytest.raises(ValueError, match="'is_prompt=True' cannot have 'step' parameter"):
            SearchSpace(is_prompt=True, step=0.1, prompt="test")

    def test_empty_values_raises_error(self):
        """Test that empty values list raises validation error."""
        with pytest.raises(ValueError, match="'values' must not be empty"):
            SearchSpace(values=[])

    def test_low_equals_high_raises_error(self):
        """Test that low == high raises validation error."""
        with pytest.raises(ValueError, match="'low' must be less than 'high'"):
            SearchSpace(low=5, high=5)

    def test_low_greater_than_high_raises_error(self):
        """Test that low > high raises validation error."""
        with pytest.raises(ValueError, match="'low' must be less than 'high'"):
            SearchSpace(low=10, high=5)

    def test_valid_prompt_space(self):
        """Test that valid prompt SearchSpace can be created."""
        space = SearchSpace(is_prompt=True, prompt="test prompt", prompt_purpose="testing")
        assert space.is_prompt is True
        assert space.prompt == "test prompt"
        assert space.prompt_purpose == "testing"

    def test_valid_values_space(self):
        """Test that valid values-based SearchSpace can be created."""
        space = SearchSpace(values=[1, 2, 3])
        assert space.values == [1, 2, 3]

    def test_valid_range_space(self):
        """Test that valid range-based SearchSpace can be created."""
        space = SearchSpace(low=0, high=10, step=1)
        assert space.low == 0
        assert space.high == 10
        assert space.step == 1


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
