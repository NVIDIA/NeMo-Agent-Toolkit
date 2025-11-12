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
"""Tests for function middleware component architecture."""

import pytest
from pydantic import Field

# Register built-in middlewares
import nat.middleware.cache_middleware  # noqa: F401
from nat.builder.builder import Builder
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_middleware
from nat.cli.type_registry import GlobalTypeRegistry
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.middleware import MiddlewareBaseConfig
from nat.middleware.function_middleware import FunctionMiddleware


class TestMiddlewareConfig(MiddlewareBaseConfig, name="test_component_middleware"):
    """Test middleware configuration."""

    test_param: str = Field(default="default_value")
    call_order: list[str] = Field(default_factory=list)


class TestMiddleware(FunctionMiddleware):
    """Test middleware that records calls."""

    def __init__(self, *, test_param: str, call_order: list[str]):
        super().__init__()
        self.test_param = test_param
        self.call_order = call_order

    async def function_middleware_invoke(self, value, call_next, context):
        self.call_order.append(f"{self.test_param}_pre")
        result = await call_next(value)
        self.call_order.append(f"{self.test_param}_post")
        return result


@pytest.fixture(scope="module", autouse=True)
def register_test_middleware():
    """Register test middleware."""

    @register_middleware(config_type=TestMiddlewareConfig)
    async def test_middleware(config: TestMiddlewareConfig, builder: Builder):
        yield TestMiddleware(test_param=config.test_param, call_order=config.call_order)


class TestMiddlewareRegistration:
    """Test function middleware registration."""

    def test_middleware_registered_in_global_registry(self):
        """Test that middleware is registered in global registry."""
        registry = GlobalTypeRegistry.get()
        registered = registry.get_registered_middleware()

        # Find our test middleware
        test_middlewares = [r for r in registered if r.config_type == TestMiddlewareConfig]
        assert len(test_middlewares) == 1
        assert test_middlewares[0].full_type == TestMiddlewareConfig.full_type

    def test_can_retrieve_middleware_registration(self):
        """Test that we can retrieve middleware registration info."""
        registry = GlobalTypeRegistry.get()
        registration = registry.get_middleware(TestMiddlewareConfig)

        assert registration.config_type == TestMiddlewareConfig
        assert registration.full_type == TestMiddlewareConfig.full_type
        assert registration.build_fn is not None


class TestBuilderMethods:
    """Test builder methods for function middlewares."""

    async def test_add_middleware(self):
        """Test adding a function middleware to the builder."""
        config = TestMiddlewareConfig(test_param="builder_test", call_order=[])

        async with WorkflowBuilder() as builder:
            middleware = await builder.add_middleware("test_middleware_1", config)

            assert isinstance(middleware, TestMiddleware)
            assert middleware.test_param == "builder_test"

    async def test_get_middleware(self):
        """Test retrieving a function middleware from the builder."""
        config = TestMiddlewareConfig(test_param="get_test", call_order=[])

        async with WorkflowBuilder() as builder:
            await builder.add_middleware("test_middleware_2", config)
            retrieved = await builder.get_middleware("test_middleware_2")

            assert isinstance(retrieved, TestMiddleware)
            assert retrieved.test_param == "get_test"

    async def test_get_middleware_config(self):
        """Test retrieving middleware config from the builder."""
        config = TestMiddlewareConfig(test_param="config_test", call_order=[])

        async with WorkflowBuilder() as builder:
            await builder.add_middleware("test_middleware_3", config)
            retrieved_config = builder.get_middleware_config("test_middleware_3")

            assert isinstance(retrieved_config, TestMiddlewareConfig)
            assert retrieved_config.test_param == "config_test"

    async def test_get_middlewares_batch(self):
        """Test retrieving multiple middlewares at once."""
        config1 = TestMiddlewareConfig(test_param="batch1", call_order=[])
        config2 = TestMiddlewareConfig(test_param="batch2", call_order=[])

        async with WorkflowBuilder() as builder:
            await builder.add_middleware("batch_1", config1)
            await builder.add_middleware("batch_2", config2)

            middlewares = await builder.get_middleware_list(["batch_1", "batch_2"])

            assert len(middlewares) == 2
            assert all(isinstance(i, TestMiddleware) for i in middlewares)
            params = {i.test_param for i in middlewares}
            assert params == {"batch1", "batch2"}

    async def test_duplicate_middleware_raises_error(self):
        """Test that adding duplicate middleware raises error."""
        config = TestMiddlewareConfig(test_param="duplicate", call_order=[])

        async with WorkflowBuilder() as builder:
            await builder.add_middleware("duplicate_test", config)

            with pytest.raises(ValueError, match="already exists"):
                await builder.add_middleware("duplicate_test", config)

    async def test_get_nonexistent_middleware_raises_error(self):
        """Test that getting nonexistent middleware raises error."""
        async with WorkflowBuilder() as builder:
            with pytest.raises(ValueError, match="not found"):
                await builder.get_middleware("nonexistent")


class TestYAMLIntegration:
    """Test YAML configuration integration."""

    async def test_middleware_from_yaml_config(self):
        """Test building middlewares from YAML config."""
        config_dict = {
            "middleware": {
                "yaml_middleware": {
                    "_type": "test_component_middleware",
                    "test_param": "from_yaml",
                }
            },
            "functions": {},
        }
        config = Config.model_validate(config_dict)

        async with WorkflowBuilder() as builder:
            # Build middlewares from config
            from nat.builder.component_utils import build_dependency_sequence

            sequence = build_dependency_sequence(config)

            for component in sequence:
                if component.component_group.value == "middleware":
                    await builder.add_middleware(component.name, component.config)

            # Verify middleware was built
            middleware = await builder.get_middleware("yaml_middleware")
            assert isinstance(middleware, TestMiddleware)
            assert middleware.test_param == "from_yaml"


class TestMiddlewareWithFunctions:
    """Test middlewares integrated with functions."""

    @pytest.fixture(scope="class")
    def register_test_function(self):
        """Register a test function that uses middlewares."""

        class TestFunctionConfig(FunctionBaseConfig, name="test_func_with_middlewares"):
            pass

        @register_function(config_type=TestFunctionConfig)
        async def test_function(config: TestFunctionConfig, builder: Builder):
            from nat.builder.function import LambdaFunction
            from nat.builder.function_info import FunctionInfo

            async def process(value: int) -> int:
                return value * 2

            info = FunctionInfo.from_fn(process)
            yield LambdaFunction.from_info(config=config, info=info, instance_name="test_func")

    async def test_function_with_middlewares_via_builder(self, register_test_function):
        """Test that functions can use middlewares configured in builder."""
        call_order = []

        config_dict = {
            "middleware": {
                "func_middleware_1": {
                    "_type": "test_component_middleware",
                    "test_param": "first",
                },
                "func_middleware_2": {
                    "_type": "test_component_middleware",
                    "test_param": "second",
                },
            },
            "functions": {
                "test_func": {
                    "_type": "test_func_with_middlewares",
                    "middleware": ["func_middleware_1", "func_middleware_2"],
                }
            },
        }
        config = Config.model_validate(config_dict)

        async with WorkflowBuilder() as builder:
            # Manually build middlewares first
            for name, middleware_config in config.middleware.items():
                # Pass shared call_order to track execution
                middleware_config.call_order = call_order
                await builder.add_middleware(name, middleware_config)

            # Now build function
            func = await builder.add_function("test_func", config.functions["test_func"])

            # Invoke function and check middlewares were called in order
            result = await func.ainvoke(5, to_type=int)
            assert result == 10

            # Verify middlewares were called in correct order
            assert call_order == ["first_pre", "second_pre", "second_post", "first_post"]


class TestMiddlewareBuildOrder:
    """Test that middlewares are built before functions."""

    async def test_middlewares_built_before_functions(self):
        """Test that component build order has middlewares before functions."""
        from nat.builder.component_utils import _component_group_order
        from nat.data_models.component import ComponentGroup

        middlewares_idx = _component_group_order.index(ComponentGroup.MIDDLEWARE)
        functions_idx = _component_group_order.index(ComponentGroup.FUNCTIONS)
        function_groups_idx = _component_group_order.index(ComponentGroup.FUNCTION_GROUPS)

        # Middlewares must be before functions and function groups
        assert middlewares_idx < functions_idx
        assert middlewares_idx < function_groups_idx


class TestCacheMiddlewareComponent:
    """Test that the built-in cache middleware works as a component."""

    async def test_cache_middleware_registration(self):
        """Test that cache middleware is registered."""
        from nat.middleware.register import CacheMiddlewareConfig

        registry = GlobalTypeRegistry.get()
        registration = registry.get_middleware(CacheMiddlewareConfig)

        assert registration.config_type == CacheMiddlewareConfig
        assert registration.full_type == CacheMiddlewareConfig.full_type

    async def test_cache_middleware_from_yaml(self):
        """Test building cache middleware from YAML."""
        from nat.middleware.cache_middleware import CacheMiddleware

        config_dict = {
            "middleware": {
                "my_cache": {
                    "_type": "cache",
                    "enabled_mode": "always",
                    "similarity_threshold": 1.0,
                }
            }
        }
        config = Config.model_validate(config_dict)

        async with WorkflowBuilder() as builder:
            middleware = await builder.add_middleware("my_cache", config.middleware["my_cache"])

            assert isinstance(middleware, CacheMiddleware)
            assert middleware.is_final is True

    async def test_cache_middleware_with_different_configs(self):
        """Test cache middleware with various configurations."""
        from nat.middleware.cache_middleware import CacheMiddleware

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
                config_dict = {"middleware": {f"cache_{i}": {"_type": "cache", **config_params}}}
                config = Config.model_validate(config_dict)

                middleware = await builder.add_middleware(f"cache_{i}", config.middleware[f"cache_{i}"])

                assert isinstance(middleware, CacheMiddleware)


class TestMiddlewareErrorHandling:
    """Test error handling for middlewares."""

    async def test_missing_middleware_in_function_raises_error(self):
        """Test that referencing nonexistent middleware raises error."""

        class MissingMiddlewareFunctionConfig(FunctionBaseConfig, name="missing_middleware_func"):
            pass

        @register_function(config_type=MissingMiddlewareFunctionConfig)
        async def function_with_missing_middleware(config, builder):
            from nat.builder.function import LambdaFunction
            from nat.builder.function_info import FunctionInfo

            async def process(value: int) -> int:
                return value

            info = FunctionInfo.from_fn(process)
            yield LambdaFunction.from_info(config=config, info=info, instance_name="test")

        config_dict = {
            "functions": {
                "test_func": {
                    "_type": "missing_middleware_func", "middleware": ["nonexistent_middleware"]
                }
            }
        }
        config = Config.model_validate(config_dict)

        async with WorkflowBuilder() as builder:
            with pytest.raises(ValueError, match="Middleware `nonexistent_middleware` not found"):
                await builder.add_function("test_func", config.functions["test_func"])


class TestFunctionGroupMiddlewares:
    """Test middlewares with function groups."""

    @pytest.fixture(scope="class")
    def register_test_function_group(self):
        """Register a test function group."""
        from nat.cli.register_workflow import register_function_group
        from nat.data_models.function import FunctionGroupBaseConfig

        class TestFunctionGroupConfig(FunctionGroupBaseConfig, name="test_func_group_with_middlewares"):
            pass

        @register_function_group(config_type=TestFunctionGroupConfig)
        async def test_function_group(config: TestFunctionGroupConfig, builder: Builder):
            from nat.builder.function import FunctionGroup

            group = FunctionGroup(config=config)

            async def func1(value: int) -> int:
                return value * 2

            async def func2(value: int) -> int:
                return value + 10

            group.add_function("func1", func1, description="Multiply by 2")
            group.add_function("func2", func2, description="Add 10")

            yield group

    async def test_function_group_with_middlewares_via_builder(self, register_test_function_group):
        """Test that function groups can use middlewares configured in builder."""
        call_order = []

        config_dict = {
            "middleware": {
                "group_middleware_1": {
                    "_type": "test_component_middleware",
                    "test_param": "group_first",
                },
                "group_middleware_2": {
                    "_type": "test_component_middleware",
                    "test_param": "group_second",
                },
            },
            "function_groups": {
                "test_group": {
                    "_type": "test_func_group_with_middlewares",
                    "middleware": ["group_middleware_1", "group_middleware_2"],
                }
            },
        }
        config = Config.model_validate(config_dict)

        async with WorkflowBuilder() as builder:
            # Manually build middlewares first
            for name, middleware_config in config.middleware.items():
                # Pass shared call_order to track execution
                middleware_config.call_order = call_order
                await builder.add_middleware(name, middleware_config)

            # Now build function group
            group = await builder.add_function_group("test_group", config.function_groups["test_group"])

            # Get accessible functions from the group
            functions = await group.get_accessible_functions()

            # Test that middlewares are applied to func1
            func1 = functions["test_group.func1"]
            result = await func1.ainvoke(5)
            assert result == 10  # 5 * 2

            # Verify middlewares were called in correct order for func1
            assert call_order == ["group_first_pre", "group_second_pre", "group_second_post", "group_first_post"]

            # Clear call order for next test
            call_order.clear()

            # Test that middlewares are applied to func2
            func2 = functions["test_group.func2"]
            result = await func2.ainvoke(5)
            assert result == 15  # 5 + 10

            # Verify middlewares were called for func2 as well
            assert call_order == ["group_first_pre", "group_second_pre", "group_second_post", "group_first_post"]

    async def test_function_group_middlewares_propagated_to_new_functions(self):
        """Test that middlewares are propagated to functions added after group creation."""
        from nat.builder.function import FunctionGroup
        from nat.data_models.function import FunctionGroupBaseConfig

        call_order = []

        # Create test middleware
        middleware = TestMiddleware(test_param="dynamic", call_order=call_order)

        # Create function group with middlewares
        config = FunctionGroupBaseConfig()
        group = FunctionGroup(config=config, middleware=[middleware])

        # Add function after group creation
        async def new_func(value: int) -> int:
            return value * 3

        group.add_function("dynamic_func", new_func)

        # Get the function and test it has middlewares
        func = group._functions["dynamic_func"]
        result = await func.ainvoke(4)
        assert result == 12  # 4 * 3

        # Verify middlewares were called
        assert call_order == ["dynamic_pre", "dynamic_post"]

    async def test_function_group_configure_middlewares_updates_existing(self):
        """Test that configure_middlewares updates existing functions."""
        from nat.builder.function import FunctionGroup
        from nat.data_models.function import FunctionGroupBaseConfig

        call_order1 = []
        call_order2 = []

        # Create function group without middlewares initially
        config = FunctionGroupBaseConfig()
        group = FunctionGroup(config=config)

        # Add functions
        async def func1(value: int) -> int:
            return value * 2

        async def func2(value: int) -> int:
            return value + 5

        group.add_function("func1", func1)
        group.add_function("func2", func2)

        # Test functions without middlewares
        result1 = await group._functions["func1"].ainvoke(3)
        assert result1 == 6
        assert len(call_order1) == 0  # No middlewares called

        # Now configure middlewares
        middleware1 = TestMiddleware(test_param="after1", call_order=call_order1)
        middleware2 = TestMiddleware(test_param="after2", call_order=call_order2)
        group.configure_middleware([middleware1, middleware2])

        # Test functions with middlewares
        result2 = await group._functions["func1"].ainvoke(3)
        assert result2 == 6
        assert call_order1 == ["after1_pre", "after1_post"]
        assert call_order2 == ["after2_pre", "after2_post"]

        # Clear and test func2
        call_order1.clear()
        call_order2.clear()
        result3 = await group._functions["func2"].ainvoke(3)
        assert result3 == 8
        assert call_order1 == ["after1_pre", "after1_post"]
        assert call_order2 == ["after2_pre", "after2_post"]

    async def test_function_group_missing_middleware_raises_error(self):
        """Test that referencing nonexistent middleware in function group raises error."""
        from nat.cli.register_workflow import register_function_group
        from nat.data_models.function import FunctionGroupBaseConfig

        class MissingMiddlewareGroupConfig(FunctionGroupBaseConfig, name="missing_middleware_group"):
            pass

        @register_function_group(config_type=MissingMiddlewareGroupConfig)
        async def function_group_with_missing_middleware(config, builder):
            from nat.builder.function import FunctionGroup

            group = FunctionGroup(config=config)

            async def test_func(value: int) -> int:
                return value

            group.add_function("test", test_func)
            yield group

        config_dict = {
            "function_groups": {
                "test_group": {
                    "_type": "missing_middleware_group", "middleware": ["nonexistent_group_middleware"]
                }
            }
        }
        config = Config.model_validate(config_dict)

        async with WorkflowBuilder() as builder:
            with pytest.raises(ValueError,
                               match="Middleware `nonexistent_group_middleware` not found for function group"):
                await builder.add_function_group("test_group", config.function_groups["test_group"])

    async def test_function_group_middlewares_with_cache(self):
        """Test function group with cache middleware."""
        from nat.cli.register_workflow import register_function_group
        from nat.data_models.function import FunctionGroupBaseConfig

        class CachedGroupConfig(FunctionGroupBaseConfig, name="cached_group"):
            pass

        @register_function_group(config_type=CachedGroupConfig)
        async def cached_function_group(config, builder):
            from nat.builder.function import FunctionGroup

            group = FunctionGroup(config=config)

            # Counter to track function calls
            call_count = {"func1": 0, "func2": 0}

            async def func1(value: str) -> str:
                call_count["func1"] += 1
                return f"func1_result_{value}_{call_count['func1']}"

            async def func2(value: str) -> str:
                call_count["func2"] += 1
                return f"func2_result_{value}_{call_count['func2']}"

            group.add_function("func1", func1)
            group.add_function("func2", func2)

            # Store call_count for testing
            group._test_call_count = call_count
            yield group

        config_dict = {
            "middleware": {
                "group_cache": {
                    "_type": "cache",
                    "enabled_mode": "always",
                    "similarity_threshold": 1.0,
                }
            },
            "function_groups": {
                "cached_group": {
                    "_type": "cached_group",
                    "middleware": ["group_cache"],
                }
            }
        }
        config = Config.model_validate(config_dict)

        async with WorkflowBuilder() as builder:
            # Build middlewares
            for name, middleware_config in config.middleware.items():
                await builder.add_middleware(name, middleware_config)

            # Build function group
            group = await builder.add_function_group("cached_group", config.function_groups["cached_group"])

            # Get functions
            functions = await group.get_accessible_functions()
            func1 = functions["cached_group.func1"]
            func2 = functions["cached_group.func2"]

            # Test func1 caching
            result1 = await func1.ainvoke("test1")
            assert result1 == "func1_result_test1_1"
            assert group._test_call_count["func1"] == 1

            # Second call should use cache
            result2 = await func1.ainvoke("test1")
            assert result2 == "func1_result_test1_1"
            assert group._test_call_count["func1"] == 1  # No additional call

            # Different input should call function
            result3 = await func1.ainvoke("different")
            assert result3 == "func1_result_different_2"
            assert group._test_call_count["func1"] == 2

            # Test func2 also has cache (use different input to avoid cross-function cache hit)
            result4 = await func2.ainvoke("test2")
            assert result4 == "func2_result_test2_1"
            assert group._test_call_count["func2"] == 1

            # Second call should use cache
            result5 = await func2.ainvoke("test2")
            assert result5 == "func2_result_test2_1"
            assert group._test_call_count["func2"] == 1  # No additional call

    async def test_function_group_middlewares_order_matters(self):
        """Test that middleware order is preserved and matters for function groups."""
        from nat.builder.function import FunctionGroup
        from nat.data_models.function import FunctionGroupBaseConfig

        results = []

        class OrderTestMiddleware(FunctionMiddleware):

            def __init__(self, name: str):
                super().__init__()
                self.name = name

            async def function_middleware_invoke(self, value, call_next, context):
                results.append(f"{self.name}_pre")
                # Modify value based on middleware name
                if self.name == "first":
                    value = value * 2
                elif self.name == "second":
                    value = value + 10
                result = await call_next(value)
                results.append(f"{self.name}_post")
                return result

        # Create function group with ordered middlewares
        config = FunctionGroupBaseConfig()
        middlewares = [OrderTestMiddleware("first"), OrderTestMiddleware("second")]
        group = FunctionGroup(config=config, middleware=middlewares)

        async def test_func(value: int) -> int:
            return value

        group.add_function("order_test", test_func)

        # Test the function
        func = group._functions["order_test"]
        result = await func.ainvoke(5)

        # Value is first multiplied by 2 (10), then added 10 (20)
        assert result == 20
        assert results == ["first_pre", "second_pre", "second_post", "first_post"]
