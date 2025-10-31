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
"""Tests for function intercept component architecture."""

import pytest
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_function_intercept
from nat.cli.type_registry import GlobalTypeRegistry
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function_intercept import FunctionInterceptBaseConfig
from nat.intercepts.function_intercept import FunctionIntercept


class TestInterceptConfig(FunctionInterceptBaseConfig, name="test_intercept"):
    """Test intercept configuration."""

    test_param: str = Field(default="default_value")
    call_order: list[str] = Field(default_factory=list)


class TestIntercept(FunctionIntercept):
    """Test intercept that records calls."""

    def __init__(self, *, test_param: str, call_order: list[str]):
        super().__init__()
        self.test_param = test_param
        self.call_order = call_order

    async def intercept_invoke(self, value, call_next, context):
        self.call_order.append(f"{self.test_param}_pre")
        result = await call_next(value)
        self.call_order.append(f"{self.test_param}_post")
        return result


@pytest.fixture(scope="module", autouse=True)
def register_test_intercept():
    """Register test intercept."""

    @register_function_intercept(config_type=TestInterceptConfig)
    async def test_intercept(config: TestInterceptConfig, builder: Builder):
        yield TestIntercept(test_param=config.test_param, call_order=config.call_order)


class TestInterceptRegistration:
    """Test function intercept registration."""

    def test_intercept_registered_in_global_registry(self):
        """Test that intercept is registered in global registry."""
        registry = GlobalTypeRegistry.get()
        registered = registry.get_registered_function_intercepts()

        # Find our test intercept
        test_intercepts = [r for r in registered if r.config_type == TestInterceptConfig]
        assert len(test_intercepts) == 1
        assert test_intercepts[0].full_type == TestInterceptConfig.full_type

    def test_can_retrieve_intercept_registration(self):
        """Test that we can retrieve intercept registration info."""
        registry = GlobalTypeRegistry.get()
        registration = registry.get_function_intercept(TestInterceptConfig)

        assert registration.config_type == TestInterceptConfig
        assert registration.full_type == TestInterceptConfig.full_type
        assert registration.build_fn is not None


class TestBuilderMethods:
    """Test builder methods for function intercepts."""

    async def test_add_function_intercept(self):
        """Test adding a function intercept to the builder."""
        config = TestInterceptConfig(test_param="builder_test", call_order=[])

        async with WorkflowBuilder() as builder:
            intercept = await builder.add_function_intercept("test_intercept_1", config)

            assert isinstance(intercept, TestIntercept)
            assert intercept.test_param == "builder_test"

    async def test_get_function_intercept(self):
        """Test retrieving a function intercept from the builder."""
        config = TestInterceptConfig(test_param="get_test", call_order=[])

        async with WorkflowBuilder() as builder:
            await builder.add_function_intercept("test_intercept_2", config)
            retrieved = await builder.get_function_intercept("test_intercept_2")

            assert isinstance(retrieved, TestIntercept)
            assert retrieved.test_param == "get_test"

    async def test_get_function_intercept_config(self):
        """Test retrieving intercept config from the builder."""
        config = TestInterceptConfig(test_param="config_test", call_order=[])

        async with WorkflowBuilder() as builder:
            await builder.add_function_intercept("test_intercept_3", config)
            retrieved_config = builder.get_function_intercept_config("test_intercept_3")

            assert isinstance(retrieved_config, TestInterceptConfig)
            assert retrieved_config.test_param == "config_test"

    async def test_get_function_intercepts_batch(self):
        """Test retrieving multiple intercepts at once."""
        config1 = TestInterceptConfig(test_param="batch1", call_order=[])
        config2 = TestInterceptConfig(test_param="batch2", call_order=[])

        async with WorkflowBuilder() as builder:
            await builder.add_function_intercept("batch_1", config1)
            await builder.add_function_intercept("batch_2", config2)

            intercepts = await builder.get_function_intercepts(["batch_1", "batch_2"])

            assert len(intercepts) == 2
            assert all(isinstance(i, TestIntercept) for i in intercepts)
            params = {i.test_param for i in intercepts}
            assert params == {"batch1", "batch2"}

    async def test_duplicate_intercept_raises_error(self):
        """Test that adding duplicate intercept raises error."""
        config = TestInterceptConfig(test_param="duplicate", call_order=[])

        async with WorkflowBuilder() as builder:
            await builder.add_function_intercept("duplicate_test", config)

            with pytest.raises(ValueError, match="already exists"):
                await builder.add_function_intercept("duplicate_test", config)

    async def test_get_nonexistent_intercept_raises_error(self):
        """Test that getting nonexistent intercept raises error."""
        async with WorkflowBuilder() as builder:
            with pytest.raises(ValueError, match="not found"):
                await builder.get_function_intercept("nonexistent")


class TestYAMLIntegration:
    """Test YAML configuration integration."""

    async def test_intercept_from_yaml_config(self):
        """Test building intercepts from YAML config."""
        config_dict = {
            "function_intercepts": {
                "yaml_intercept": {
                    "_type": "test_intercept",
                    "test_param": "from_yaml",
                }
            },
            "functions": {},
        }
        config = Config.model_validate(config_dict)

        async with WorkflowBuilder() as builder:
            # Build intercepts from config
            from nat.builder.component_utils import build_dependency_sequence

            sequence = build_dependency_sequence(config)

            for component in sequence:
                if component.component_group.value == "function_intercepts":
                    await builder.add_function_intercept(component.name, component.config)

            # Verify intercept was built
            intercept = await builder.get_function_intercept("yaml_intercept")
            assert isinstance(intercept, TestIntercept)
            assert intercept.test_param == "from_yaml"


class TestInterceptWithFunctions:
    """Test intercepts integrated with functions."""

    @pytest.fixture(scope="class")
    def register_test_function(self):
        """Register a test function that uses intercepts."""

        class TestFunctionConfig(FunctionBaseConfig, name="test_func_with_intercepts"):
            pass

        @register_function(config_type=TestFunctionConfig, intercept_names=["func_intercept_1", "func_intercept_2"])
        async def test_function(config: TestFunctionConfig, builder: Builder):
            from nat.builder.function import LambdaFunction
            from nat.builder.function_info import FunctionInfo

            async def process(value: int) -> int:
                return value * 2

            info = FunctionInfo.from_fn(process)
            yield LambdaFunction.from_info(config=config, info=info, instance_name="test_func")

    async def test_function_with_intercepts_via_builder(self, register_test_function):
        """Test that functions can use intercepts configured in builder."""
        call_order = []

        config_dict = {
            "function_intercepts": {
                "func_intercept_1": {
                    "_type": "test_intercept",
                    "test_param": "first",
                },
                "func_intercept_2": {
                    "_type": "test_intercept",
                    "test_param": "second",
                },
            },
            "functions": {
                "test_func": {
                    "_type": "test_func_with_intercepts",
                }
            },
        }
        config = Config.model_validate(config_dict)

        async with WorkflowBuilder() as builder:
            # Manually build intercepts first
            for name, intercept_config in config.function_intercepts.items():
                # Pass shared call_order to track execution
                intercept_config.call_order = call_order
                await builder.add_function_intercept(name, intercept_config)

            # Now build function
            func = await builder.add_function("test_func", config.functions["test_func"])

            # Invoke function and check intercepts were called in order
            result = await func.ainvoke(5, to_type=int)
            assert result == 10

            # Verify intercepts were called in correct order
            assert call_order == ["first_pre", "second_pre", "second_post", "first_post"]


class TestInterceptBuildOrder:
    """Test that intercepts are built before functions."""

    async def test_intercepts_built_before_functions(self):
        """Test that component build order has intercepts before functions."""
        from nat.builder.component_utils import _component_group_order
        from nat.data_models.component import ComponentGroup

        intercepts_idx = _component_group_order.index(ComponentGroup.FUNCTION_INTERCEPTS)
        functions_idx = _component_group_order.index(ComponentGroup.FUNCTIONS)
        function_groups_idx = _component_group_order.index(ComponentGroup.FUNCTION_GROUPS)

        # Intercepts must be before functions and function groups
        assert intercepts_idx < functions_idx
        assert intercepts_idx < function_groups_idx


class TestCacheInterceptComponent:
    """Test that the built-in cache intercept works as a component."""

    async def test_cache_intercept_registration(self):
        """Test that cache intercept is registered."""
        from nat.intercepts.register import CacheInterceptConfig

        registry = GlobalTypeRegistry.get()
        registration = registry.get_function_intercept(CacheInterceptConfig)

        assert registration.config_type == CacheInterceptConfig
        assert registration.full_type == CacheInterceptConfig.full_type

    async def test_cache_intercept_from_yaml(self):
        """Test building cache intercept from YAML."""
        from nat.intercepts.cache_intercept import CacheIntercept

        config_dict = {
            "function_intercepts": {
                "my_cache": {
                    "_type": "cache",
                    "enabled_mode": "always",
                    "similarity_threshold": 1.0,
                }
            }
        }
        config = Config.model_validate(config_dict)

        async with WorkflowBuilder() as builder:
            intercept = await builder.add_function_intercept("my_cache", config.function_intercepts["my_cache"])

            assert isinstance(intercept, CacheIntercept)
            assert intercept.is_final is True

    async def test_cache_intercept_with_different_configs(self):
        """Test cache intercept with various configurations."""
        from nat.intercepts.cache_intercept import CacheIntercept

        configs = [
            {
                "enabled_mode": "always", "similarity_threshold": 1.0
            },
            {
                "enabled_mode": "eval", "similarity_threshold": 0.95
            },
        ]

        async with WorkflowBuilder() as builder:
            for i, config_params in enumerate(configs):
                config_dict = {"function_intercepts": {f"cache_{i}": {"_type": "cache", **config_params}}}
                config = Config.model_validate(config_dict)

                intercept = await builder.add_function_intercept(f"cache_{i}", config.function_intercepts[f"cache_{i}"])

                assert isinstance(intercept, CacheIntercept)


class TestInterceptErrorHandling:
    """Test error handling for intercepts."""

    async def test_missing_intercept_in_function_raises_error(self):
        """Test that referencing nonexistent intercept raises error."""

        class MissingInterceptFunctionConfig(FunctionBaseConfig, name="missing_intercept_func"):
            pass

        @register_function(config_type=MissingInterceptFunctionConfig, intercept_names=["nonexistent_intercept"])
        async def function_with_missing_intercept(config, builder):
            from nat.builder.function import LambdaFunction
            from nat.builder.function_info import FunctionInfo

            async def process(value: int) -> int:
                return value

            info = FunctionInfo.from_fn(process)
            yield LambdaFunction.from_info(config=config, info=info, instance_name="test")

        config_dict = {"functions": {"test_func": {"_type": "missing_intercept_func", }}}
        config = Config.model_validate(config_dict)

        async with WorkflowBuilder() as builder:
            with pytest.raises(ValueError, match="Function intercept `nonexistent_intercept` not found"):
                await builder.add_function("test_func", config.functions["test_func"])
