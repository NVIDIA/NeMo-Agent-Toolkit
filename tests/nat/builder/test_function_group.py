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

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from nat.builder.function import FunctionGroup
from nat.builder.function import LambdaFunction
from nat.data_models.function import EmptyFunctionConfig
from nat.data_models.function import FunctionGroupBaseConfig


class FunctionGroupTestConfig(FunctionGroupBaseConfig, name="test_function_group"):
    pass


class FunctionGroupTestIncludeConfig(FunctionGroupBaseConfig, name="test_function_group_include"):
    include: list[str] = ["func1", "func2"]


class FunctionGroupTestExcludeConfig(FunctionGroupBaseConfig, name="test_function_group_exclude"):
    exclude: list[str] = ["func3"]


def test_function_group_basic_initialization():
    """Test basic FunctionGroup initialization and function addition."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config, instance_name="test_group")

    assert group.get_config() == config
    assert group._instance_name == "test_group"
    assert len(group._functions) == 0

    # Test adding functions
    async def test_fn1(x: str) -> str:
        return x + "_fn1"

    async def test_fn2(x: str) -> str:
        return x + "_fn2"

    group.add_function("func1", test_fn1, description="Test function 1")
    group.add_function("func2", test_fn2, description="Test function 2")

    assert len(group._functions) == 2
    assert "func1" in group._functions
    assert "func2" in group._functions


def test_function_group_add_function_validation():
    """Test validation when adding functions to FunctionGroup."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config)

    async def test_fn(x: str) -> str:
        return x

    # Test empty/blank name validation
    with pytest.raises(ValueError, match="Function name cannot be empty or blank"):
        group.add_function("", test_fn)

    with pytest.raises(ValueError, match="Function name cannot be empty or blank"):
        group.add_function("   ", test_fn)

    # Test invalid character validation
    with pytest.raises(ValueError,
                       match="Function name can only contain letters, numbers, underscores, periods, and hyphens"):
        group.add_function("func@name", test_fn)

    with pytest.raises(ValueError,
                       match="Function name can only contain letters, numbers, underscores, periods, and hyphens"):
        group.add_function("func name", test_fn)

    # Test duplicate function names
    group.add_function("test_func", test_fn)
    with pytest.raises(ValueError, match="Function test_func already exists in function group"):
        group.add_function("test_func", test_fn)


@pytest.mark.asyncio
async def test_function_group_filter_fn():
    """Test FunctionGroup-level filter functions."""
    config = FunctionGroupTestConfig()

    # Filter function that only includes functions starting with "func1"
    async def group_filter(names: Sequence[str]) -> Sequence[str]:
        return [name for name in names if name.startswith("func1")]

    group = FunctionGroup(config=config, filter_fn=group_filter)

    # Add test functions
    async def test_fn1(x: str) -> str:
        return x + "_fn1"

    async def test_fn2(x: str) -> str:
        return x + "_fn2"

    async def test_fn3(x: str) -> str:
        return x + "_fn3"

    group.add_function("func1", test_fn1)
    group.add_function("func1_alt", test_fn2)
    group.add_function("func2", test_fn3)

    # Test get_accessible_functions with group filter
    accessible = await group.get_accessible_functions()
    expected_keys = {"test_function_group.func1", "test_function_group.func1_alt"}
    assert set(accessible.keys()) == expected_keys

    # Test get_all_functions with group filter
    all_funcs = await group.get_all_functions()
    assert set(all_funcs.keys()) == expected_keys

    # Test overriding filter function at call time
    async def override_filter(names: Sequence[str]) -> Sequence[str]:
        return [name for name in names if name == "func2"]

    accessible_override = await group.get_accessible_functions(filter_fn=override_filter)
    assert set(accessible_override.keys()) == {"test_function_group.func2"}


@pytest.mark.asyncio
async def test_function_group_per_function_filter():
    """Test per-function filter functions."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config)

    async def test_fn1(x: str) -> str:
        return x + "_fn1"

    async def test_fn2(x: str) -> str:
        return x + "_fn2"

    async def test_fn3(x: str) -> str:
        return x + "_fn3"

    # Add functions with per-function filters
    async def exclude_func1(name: str) -> bool:
        return False  # Always exclude func1

    async def include_func2(name: str) -> bool:
        return True  # Always include func2

    group.add_function("func1", test_fn1, filter_fn=exclude_func1)
    group.add_function("func2", test_fn2, filter_fn=include_func2)
    group.add_function("func3", test_fn3)  # No per-function filter

    # Test that func1 is excluded by its per-function filter
    accessible = await group.get_accessible_functions()
    expected_keys = {"test_function_group.func2", "test_function_group.func3"}
    assert set(accessible.keys()) == expected_keys

    # Test get_all_functions also respects per-function filters
    all_funcs = await group.get_all_functions()
    assert set(all_funcs.keys()) == expected_keys


@pytest.mark.asyncio
async def test_function_group_filter_interaction_with_include_config():
    """Test interaction between filters and include configuration."""
    config = FunctionGroupTestIncludeConfig()  # includes func1, func2

    # Group filter that only allows func2, func3
    async def group_filter(names: Sequence[str]) -> Sequence[str]:
        return [name for name in names if name in ["func2", "func3"]]

    group = FunctionGroup(config=config, filter_fn=group_filter)

    async def test_fn(x: str) -> str:
        return x

    # Add functions
    group.add_function("func1", test_fn)
    group.add_function("func2", test_fn)
    group.add_function("func3", test_fn)

    # Only func2 should be accessible (intersection of include config and group filter)
    accessible = await group.get_accessible_functions()
    assert set(accessible.keys()) == {"test_function_group_include.func2"}

    # get_included_functions should also respect the group filter
    included = await group.get_included_functions()
    assert set(included.keys()) == {"test_function_group_include.func2"}


@pytest.mark.asyncio
async def test_function_group_filter_interaction_with_exclude_config():
    """Test interaction between filters and exclude configuration."""
    config = FunctionGroupTestExcludeConfig()  # excludes func3

    # Group filter that only allows func1, func3
    async def group_filter(names: Sequence[str]) -> Sequence[str]:
        return [name for name in names if name in ["func1", "func3"]]

    group = FunctionGroup(config=config, filter_fn=group_filter)

    async def test_fn(x: str) -> str:
        return x

    # Add functions
    group.add_function("func1", test_fn)
    group.add_function("func2", test_fn)
    group.add_function("func3", test_fn)

    # Only func1 should be accessible (group filter allows func1,func3 but config excludes func3)
    accessible = await group.get_accessible_functions()
    assert set(accessible.keys()) == {"test_function_group_exclude.func1"}


@pytest.mark.asyncio
async def test_function_group_complex_filter_interaction():
    """Test complex interaction between group filters, per-function filters, and config."""
    config = FunctionGroupTestConfig()

    # Group filter that excludes func4
    async def group_filter(names: Sequence[str]) -> Sequence[str]:
        return [name for name in names if name != "func4"]

    group = FunctionGroup(config=config, filter_fn=group_filter)

    async def test_fn(x: str) -> str:
        return x

    # Per-function filter that excludes func2
    async def exclude_func2(name: str) -> bool:
        return False

    # Add functions
    group.add_function("func1", test_fn)  # Should be included
    group.add_function("func2", test_fn, filter_fn=exclude_func2)  # Excluded by per-function filter
    group.add_function("func3", test_fn)  # Should be included
    group.add_function("func4", test_fn)  # Excluded by group filter

    # Only func1 and func3 should be accessible
    accessible = await group.get_accessible_functions()
    expected_keys = {"test_function_group.func1", "test_function_group.func3"}
    assert set(accessible.keys()) == expected_keys

    # Test excluded functions
    excluded = await group.get_excluded_functions()
    assert set(excluded.keys()) == {"test_function_group.func2", "test_function_group.func4"}


@pytest.mark.asyncio
async def test_function_group_set_filter_fn():
    """Test set_filter_fn method."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config)

    async def test_fn(x: str) -> str:
        return x

    # Add functions
    group.add_function("func1", test_fn)
    group.add_function("func2", test_fn)
    group.add_function("func3", test_fn)

    # Initially no filter, all functions accessible
    accessible = await group.get_accessible_functions()
    assert len(accessible) == 3

    # Set a filter function that only includes func1
    async def new_filter(names: Sequence[str]) -> Sequence[str]:
        return [name for name in names if name == "func1"]

    group.set_filter_fn(new_filter)

    # Now only func1 should be accessible
    accessible = await group.get_accessible_functions()
    assert set(accessible.keys()) == {"test_function_group.func1"}


@pytest.mark.asyncio
async def test_function_group_set_per_function_filter_fn():
    """Test set_per_function_filter_fn method."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config)

    async def test_fn(x: str) -> str:
        return x

    # Add functions
    group.add_function("func1", test_fn)
    group.add_function("func2", test_fn)

    # Initially all functions accessible
    accessible = await group.get_accessible_functions()
    assert len(accessible) == 2

    # Set per-function filter to exclude func1
    async def exclude_func1(name: str) -> bool:
        return False

    group.set_per_function_filter_fn("func1", exclude_func1)

    # Now only func2 should be accessible
    accessible = await group.get_accessible_functions()
    assert set(accessible.keys()) == {"test_function_group.func2"}

    # Test error when setting filter for non-existent function
    with pytest.raises(ValueError, match="Function nonexistent not found in function group"):
        group.set_per_function_filter_fn("nonexistent", exclude_func1)


@pytest.mark.asyncio
async def test_function_group_config_validation_errors():
    """Test error cases for include/exclude configuration validation."""
    config = FunctionGroupTestIncludeConfig()
    group = FunctionGroup(config=config)

    async def test_fn(x: str) -> str:
        return x

    # Add only func3, but config expects func1 and func2
    group.add_function("func3", test_fn)

    # Should raise error for unknown included functions
    with pytest.raises(ValueError, match="Unknown included functions: \\['func1', 'func2'\\]"):
        await group.get_included_functions()

    with pytest.raises(ValueError, match="Unknown included functions: \\['func1', 'func2'\\]"):
        await group.get_accessible_functions()  # Uses get_included_functions internally


@pytest.mark.asyncio
async def test_function_group_exclude_config_validation_errors():
    """Test error cases for exclude configuration validation."""
    config = FunctionGroupTestExcludeConfig()
    group = FunctionGroup(config=config)

    async def test_fn(x: str) -> str:
        return x

    # Add functions that don't include the excluded func3
    group.add_function("func1", test_fn)
    group.add_function("func2", test_fn)

    # Should raise error for unknown excluded functions
    with pytest.raises(ValueError, match="Unknown excluded functions: \\['func3'\\]"):
        await group.get_excluded_functions()

    with pytest.raises(ValueError, match="Unknown excluded functions: \\['func3'\\]"):
        await group.get_accessible_functions()  # Uses _get_all_but_excluded_functions internally


@pytest.mark.asyncio
async def test_function_group_empty_filter_behavior():
    """Test behavior with empty filter functions."""
    config = FunctionGroupTestConfig()

    # Filter that returns empty list
    async def empty_filter(names: Sequence[str]) -> Sequence[str]:
        return []

    group = FunctionGroup(config=config, filter_fn=empty_filter)

    async def test_fn(x: str) -> str:
        return x

    group.add_function("func1", test_fn)
    group.add_function("func2", test_fn)

    # No functions should be accessible due to empty filter
    accessible = await group.get_accessible_functions()
    assert len(accessible) == 0

    all_funcs = await group.get_all_functions()
    assert len(all_funcs) == 0


@pytest.mark.asyncio
async def test_function_group_filter_override_precedence():
    """Test that parameter filter_fn takes precedence over instance filter_fn."""
    config = FunctionGroupTestConfig()

    # Instance filter includes only func1
    async def instance_filter(names: Sequence[str]) -> Sequence[str]:
        return [name for name in names if name == "func1"]

    group = FunctionGroup(config=config, filter_fn=instance_filter)

    async def test_fn(x: str) -> str:
        return x

    group.add_function("func1", test_fn)
    group.add_function("func2", test_fn)

    # With instance filter, only func1 is accessible
    accessible = await group.get_accessible_functions()
    assert set(accessible.keys()) == {"test_function_group.func1"}

    # Override filter includes only func2
    async def override_filter(names: Sequence[str]) -> Sequence[str]:
        return [name for name in names if name == "func2"]

    accessible_override = await group.get_accessible_functions(filter_fn=override_filter)
    assert set(accessible_override.keys()) == {"test_function_group.func2"}

    # Instance filter should still work when no override provided
    accessible_instance = await group.get_accessible_functions()
    assert set(accessible_instance.keys()) == {"test_function_group.func1"}


def test_function_group_instance_name_defaults():
    """Test instance_name defaults to config.type when not provided."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config)  # No instance_name provided
    assert group._instance_name == config.type
    assert group._instance_name == "test_function_group"


def test_function_group_get_config():
    """Test get_config() returns the correct configuration."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config, instance_name="test_group")
    assert group.get_config() is config
    assert group.get_config() == config


def test_function_group_with_pydantic_input_schema():
    """Test adding functions with Pydantic BaseModel input schemas."""

    class TestInput(BaseModel):
        value: str
        count: int = 1

    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config)

    async def test_fn(input_data: TestInput) -> str:
        return f"{input_data.value}_{input_data.count}"

    group.add_function("func_with_schema", test_fn, input_schema=TestInput, description="Function with Pydantic input")

    # Verify the function was added with correct schema
    assert "func_with_schema" in group._functions
    func = group._functions["func_with_schema"]
    assert func.input_schema == TestInput
    assert func.description == "Function with Pydantic input"


def test_function_group_with_converters():
    """Test adding functions with custom converters."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config)

    def custom_converter(value: str) -> str:
        return str(value).upper()

    async def test_fn(x: str) -> str:
        return f"converted_{x}"

    group.add_function("func_with_converter",
                       test_fn,
                       converters=[custom_converter],
                       description="Function with converter")

    assert "func_with_converter" in group._functions
    func = group._functions["func_with_converter"]
    # Verify converters were passed through to the LambdaFunction
    assert func._converter is not None


@pytest.mark.asyncio
async def test_function_group_function_name_generation():
    """Test that function names are correctly generated and stored."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config, instance_name="my_group")

    async def test_fn(x: str) -> str:
        return x

    group.add_function("test_func", test_fn)

    # Test internal storage uses short name
    assert "test_func" in group._functions
    assert "my_group.test_func" not in group._functions

    # Test that generated full names are correct in returned functions
    accessible = await group.get_accessible_functions()
    assert "my_group.test_func" in accessible
    assert accessible["my_group.test_func"] is group._functions["test_func"]

    # Test _get_fn_name method directly
    assert group._get_fn_name("test_func") == "my_group.test_func"


def test_function_group_both_include_and_exclude():
    """Test behavior when config has both include and exclude (edge case)."""
    # The framework validates that include and exclude cannot be used together
    # This test documents that behavior

    with pytest.raises(ValueError):

        class MixedConfig(FunctionGroupBaseConfig, name="mixed_config"):
            include: list[str] = ["func1", "func2"]
            exclude: list[str] = ["func2", "func3"]

        _ = MixedConfig()


@pytest.mark.asyncio
async def test_function_group_empty_include_exclude():
    """Test behavior with empty include and exclude lists."""

    class EmptyListsConfig(FunctionGroupBaseConfig, name="empty_lists"):
        include: list[str] = []
        exclude: list[str] = []

    config = EmptyListsConfig()
    group = FunctionGroup(config=config)

    async def test_fn(x: str) -> str:
        return x

    group.add_function("func1", test_fn)
    group.add_function("func2", test_fn)

    # Empty lists should behave like no configuration - all functions accessible
    accessible = await group.get_accessible_functions()
    assert set(accessible.keys()) == {"empty_lists.func1", "empty_lists.func2"}


@pytest.mark.asyncio
async def test_function_group_preserves_function_metadata():
    """Test that function descriptions and schemas are preserved correctly."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config)

    async def test_fn(x: str) -> str:
        """This is a test function."""
        return x

    group.add_function("test_func", test_fn, description="Custom description")

    func = group._functions["test_func"]
    assert func.description == "Custom description"
    assert func.instance_name == "test_function_group.test_func"


def test_function_group_lambda_function_creation():
    """Test that LambdaFunction objects are created correctly."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config)

    async def test_fn(x: str) -> str:
        return x

    group.add_function("test_func", test_fn)

    func = group._functions["test_func"]
    # Verify it's a LambdaFunction instance
    assert isinstance(func, LambdaFunction)
    # Verify it was created with EmptyFunctionConfig
    assert isinstance(func.config, EmptyFunctionConfig)


@pytest.mark.asyncio
async def test_function_group_get_excluded_functions_no_exclusions():
    """Test get_excluded_functions when no functions are actually excluded."""
    config = FunctionGroupTestConfig()  # No include/exclude
    group = FunctionGroup(config=config)

    async def test_fn(x: str) -> str:
        return x

    group.add_function("func1", test_fn)
    group.add_function("func2", test_fn)

    excluded = await group.get_excluded_functions()
    assert len(excluded) == 0


@pytest.mark.asyncio
async def test_function_group_get_included_functions_no_includes():
    """Test get_included_functions when config has no includes specified."""
    config = FunctionGroupTestConfig()  # No include specified
    group = FunctionGroup(config=config)

    async def test_fn(x: str) -> str:
        return x

    group.add_function("func1", test_fn)

    # When there's no include configuration, get_included_functions should work
    # but return empty since the config.include is empty
    included = await group.get_included_functions()
    assert len(included) == 0  # No functions included since include list is empty


@pytest.mark.asyncio
async def test_function_group_per_function_filter_logic():
    """Test _fn_should_be_included method behavior."""
    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config)

    async def test_fn(x: str) -> str:
        return x

    # Add function without per-function filter
    group.add_function("func1", test_fn)
    assert await group._fn_should_be_included("func1") is True

    # Add function with per-function filter that returns False
    async def exclude_filter(name: str) -> bool:
        return False

    group.add_function("func2", test_fn, filter_fn=exclude_filter)
    assert await group._fn_should_be_included("func2") is False

    # Add function with per-function filter that returns True
    async def include_filter(name: str) -> bool:
        return True

    group.add_function("func3", test_fn, filter_fn=include_filter)
    assert await group._fn_should_be_included("func3") is True


@pytest.mark.asyncio
async def test_function_group_comprehensive_metadata():
    """Test comprehensive function metadata preservation and handling."""

    class CustomInput(BaseModel):
        data: str
        value: int = 42

    class CustomOutput(BaseModel):
        result: str

    config = FunctionGroupTestConfig()
    group = FunctionGroup(config=config, instance_name="comprehensive_test")

    def custom_converter(x: str) -> str:
        return str(x).lower()

    async def test_fn(input_data: CustomInput) -> CustomOutput:
        return CustomOutput(result=f"{input_data.data}:{input_data.value}")

    group.add_function("complex_func",
                       test_fn,
                       input_schema=CustomInput,
                       description="A complex function with all features",
                       converters=[custom_converter])

    func = group._functions["complex_func"]

    # Test all metadata is preserved
    assert func.description == "A complex function with all features"
    assert func.input_schema == CustomInput
    assert func.instance_name == "comprehensive_test.complex_func"
    assert func._converter is not None
    assert isinstance(func, LambdaFunction)
    assert isinstance(func.config, EmptyFunctionConfig)

    # Test function appears correctly in accessible functions
    accessible = await group.get_accessible_functions()
    assert "comprehensive_test.complex_func" in accessible
    assert accessible["comprehensive_test.complex_func"] is func
