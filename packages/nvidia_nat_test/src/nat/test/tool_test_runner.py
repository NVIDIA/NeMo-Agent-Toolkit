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

import asyncio
import inspect
import logging
import typing
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.builder.function_info import FunctionInfo
from nat.cli.type_registry import GlobalTypeRegistry
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function_dependencies import FunctionDependencies
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.data_models.retriever import RetrieverBaseConfig
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.object_store.interfaces import ObjectStore
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins

logger = logging.getLogger(__name__)


class MockBuilder(Builder):
    """
    A lightweight mock builder for tool testing that provides minimal dependencies.
    """

    def __init__(self):
        self._functions = {}
        self._mocks = {}

    def mock_function(self, name: str, mock_response: typing.Any):
        """Add a mock function that returns a fixed response."""
        self._mocks[name] = mock_response

    def mock_llm(self, name: str, mock_response: typing.Any):
        """Add a mock LLM that returns a fixed response."""
        self._mocks[f"llm_{name}"] = mock_response

    def mock_embedder(self, name: str, mock_response: typing.Any):
        """Add a mock embedder that returns a fixed response."""
        self._mocks[f"embedder_{name}"] = mock_response

    def mock_memory_client(self, name: str, mock_response: typing.Any):
        """Add a mock memory client that returns a fixed response."""
        self._mocks[f"memory_{name}"] = mock_response

    def mock_retriever(self, name: str, mock_response: typing.Any):
        """Add a mock retriever that returns a fixed response."""
        self._mocks[f"retriever_{name}"] = mock_response

    def mock_object_store(self, name: str, mock_response: typing.Any):
        """Add a mock object store that returns a fixed response."""
        self._mocks[f"object_store_{name}"] = mock_response

    def mock_ttc_strategy(self, name: str, mock_response: typing.Any):
        """Add a mock TTC strategy that returns a fixed response."""
        self._mocks[f"ttc_strategy_{name}"] = mock_response

    def mock_auth_provider(self, name: str, mock_response: typing.Any):
        """Add a mock auth provider that returns a fixed response."""
        self._mocks[f"auth_provider_{name}"] = mock_response

    async def add_ttc_strategy(self, name: str, config: TTCStrategyBaseConfig) -> None:
        """Mock implementation (no‑op)."""
        pass

    async def get_ttc_strategy(self, strategy_name: str, pipeline_type: PipelineTypeEnum,
                               stage_type: StageTypeEnum) -> typing.Any:
        """Return a mock TTC strategy if one is configured."""
        key = f"ttc_strategy_{strategy_name}"
        if key in self._mocks:
            mock_strategy = MagicMock()
            # Provide common callable patterns used in tests
            mock_strategy.invoke = MagicMock(return_value=self._mocks[key])
            mock_strategy.ainvoke = AsyncMock(return_value=self._mocks[key])
            return mock_strategy
        raise ValueError(f"TTC strategy '{strategy_name}' not mocked. Use mock_ttc_strategy() to add it.")

    async def get_ttc_strategy_config(self,
                                      strategy_name: str,
                                      pipeline_type: PipelineTypeEnum,
                                      stage_type: StageTypeEnum) -> TTCStrategyBaseConfig:
        """Mock implementation."""
        return TTCStrategyBaseConfig()

    async def add_auth_provider(self, name: str, config: AuthProviderBaseConfig) -> None:
        """Mock implementation (no‑op)."""
        pass

    async def get_auth_provider(self, auth_provider_name: str) -> AuthProviderBase:
        """Return a mock auth provider if one is configured."""
        key = f"auth_provider_{auth_provider_name}"
        if key in self._mocks:
            mock_auth = MagicMock()
            mock_auth.authenticate = AsyncMock(return_value=self._mocks[key])
            return mock_auth
        raise ValueError(f"Auth provider '{auth_provider_name}' not mocked. Use mock_auth_provider() to add it.")

    async def add_function(self, name: str, config: FunctionBaseConfig) -> Function:
        """Mock implementation - not used in tool testing."""
        raise NotImplementedError("Mock implementation does not support add_function")

    def get_function(self, name: str) -> Function:
        """Return a mock function if one is configured."""
        if name in self._mocks:
            mock_fn = AsyncMock()
            mock_fn.ainvoke = AsyncMock(return_value=self._mocks[name])
            return mock_fn
        raise ValueError(f"Function '{name}' not mocked. Use mock_function() to add it.")

    def get_function_config(self, name: str) -> FunctionBaseConfig:
        """Mock implementation."""
        return FunctionBaseConfig()

    async def set_workflow(self, config: FunctionBaseConfig) -> Function:
        """Mock implementation."""
        mock_fn = AsyncMock()
        mock_fn.ainvoke = AsyncMock(return_value="mock_workflow_result")
        return mock_fn

    def get_workflow(self) -> Function:
        """Mock implementation."""
        mock_fn = AsyncMock()
        mock_fn.ainvoke = AsyncMock(return_value="mock_workflow_result")
        return mock_fn

    def get_workflow_config(self) -> FunctionBaseConfig:
        """Mock implementation."""
        return FunctionBaseConfig()

    def get_tool(self, fn_name: str, wrapper_type):
        """Mock implementation."""
        pass

    async def add_llm(self, name: str, config) -> None:
        """Mock implementation."""
        pass

    async def get_llm(self, llm_name: str, wrapper_type):
        """Return a mock LLM if one is configured."""
        key = f"llm_{llm_name}"
        if key in self._mocks:
            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=self._mocks[key])
            mock_llm.ainvoke = AsyncMock(return_value=self._mocks[key])
            return mock_llm
        raise ValueError(f"LLM '{llm_name}' not mocked. Use mock_llm() to add it.")

    def get_llm_config(self, llm_name: str) -> LLMBaseConfig:
        """Mock implementation."""
        return LLMBaseConfig()

    async def add_embedder(self, name: str, config) -> None:
        """Mock implementation."""
        pass

    async def get_embedder(self, embedder_name: str, wrapper_type):
        """Return a mock embedder if one is configured."""
        key = f"embedder_{embedder_name}"
        if key in self._mocks:
            mock_embedder = MagicMock()
            mock_embedder.embed_query = MagicMock(return_value=self._mocks[key])
            mock_embedder.embed_documents = MagicMock(return_value=self._mocks[key])
            return mock_embedder
        raise ValueError(f"Embedder '{embedder_name}' not mocked. Use mock_embedder() to add it.")

    def get_embedder_config(self, embedder_name: str) -> EmbedderBaseConfig:
        """Mock implementation."""
        return EmbedderBaseConfig()

    async def add_memory_client(self, name: str, config) -> None:
        """Mock implementation."""
        pass

    def get_memory_client(self, memory_name: str):
        """Return a mock memory client if one is configured."""
        key = f"memory_{memory_name}"
        if key in self._mocks:
            mock_memory = MagicMock()
            mock_memory.add = AsyncMock(return_value=self._mocks[key])
            mock_memory.search = AsyncMock(return_value=self._mocks[key])
            return mock_memory
        raise ValueError(f"Memory client '{memory_name}' not mocked. Use mock_memory_client() to add it.")

    def get_memory_client_config(self, memory_name: str) -> MemoryBaseConfig:
        """Mock implementation."""
        return MemoryBaseConfig()

    async def add_retriever(self, name: str, config) -> None:
        """Mock implementation."""
        pass

    async def get_retriever(self, retriever_name: str, wrapper_type=None):
        """Return a mock retriever if one is configured."""
        key = f"retriever_{retriever_name}"
        if key in self._mocks:
            mock_retriever = MagicMock()
            mock_retriever.retrieve = AsyncMock(return_value=self._mocks[key])
            return mock_retriever
        raise ValueError(f"Retriever '{retriever_name}' not mocked. Use mock_retriever() to add it.")

    async def get_retriever_config(self, retriever_name: str) -> RetrieverBaseConfig:
        """Mock implementation."""
        return RetrieverBaseConfig()

    async def add_object_store(self, name: str, config: ObjectStoreBaseConfig) -> None:
        """Mock implementation for object store."""
        pass

    async def get_object_store_client(self, object_store_name: str) -> ObjectStore:
        """Return a mock object store client if one is configured."""
        key = f"object_store_{object_store_name}"
        if key in self._mocks:
            mock_object_store = MagicMock()
            mock_object_store.put_object = AsyncMock(return_value=self._mocks[key])
            mock_object_store.get_object = AsyncMock(return_value=self._mocks[key])
            mock_object_store.delete_object = AsyncMock(return_value=self._mocks[key])
            mock_object_store.list_objects = AsyncMock(return_value=self._mocks[key])
            return mock_object_store
        raise ValueError(f"Object store '{object_store_name}' not mocked. Use mock_object_store() to add it.")

    def get_object_store_config(self, object_store_name: str) -> ObjectStoreBaseConfig:
        """Mock implementation for object store config."""
        return ObjectStoreBaseConfig()

    def get_user_manager(self):
        """Mock implementation."""
        mock_user = MagicMock()
        mock_user.get_id = MagicMock(return_value="test_user")
        return mock_user

    def get_function_dependencies(self, fn_name: str) -> FunctionDependencies:
        """Mock implementation."""
        return FunctionDependencies()


class ToolTestRunner:
    """
    A test runner that enables isolated testing of NAT tools without requiring
    full workflow setup, LLMs, or complex dependencies.

    Usage:
        runner = ToolTestRunner()

        # Test a tool with minimal setup
        result = await runner.test_tool(
            config_type=MyToolConfig,
            config_params={"param1": "value1"},
            input_data="test input"
        )

        # Test a tool with mocked dependencies
        async with runner.with_mocks() as mock_builder:
            mock_builder.mock_llm("my_llm", "mocked response")
            result = await runner.test_tool(
                config_type=MyToolConfig,
                config_params={"llm_name": "my_llm"},
                input_data="test input"
            )
    """

    def __init__(self):
        self._ensure_plugins_loaded()

    def _ensure_plugins_loaded(self):
        """Ensure all plugins are loaded for tool registration."""
        discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    async def test_tool(self,
                        config_type: type[FunctionBaseConfig],
                        config_params: dict[str, typing.Any] | None = None,
                        input_data: typing.Any = None,
                        expected_output: typing.Any = None,
                        **kwargs) -> typing.Any:
        """
        Test a tool in isolation with minimal setup.

        Args:
            config_type: The tool configuration class
            config_params: Parameters to pass to the config constructor
            input_data: Input data to pass to the tool
            expected_output: Expected output for assertion (optional)
            **kwargs: Additional parameters

        Returns:
            The tool's output

        Raises:
            AssertionError: If expected_output is provided and doesn't match
            ValueError: If tool registration or execution fails
        """
        config_params = config_params or {}

        # Create tool configuration
        config = config_type(**config_params)

        # Get the registered tool function
        registry = GlobalTypeRegistry.get()
        try:
            tool_registration = registry.get_function(config_type)
        except KeyError:
            raise ValueError(
                f"Tool {config_type} is not registered. Make sure it's imported and registered with @register_function."
            )

        # Create a mock builder for dependencies
        mock_builder = MockBuilder()

        # Build the tool function
        async with tool_registration.build_fn(config, mock_builder) as tool_result:

            # Handle different tool result types
            if isinstance(tool_result, Function):
                tool_function = tool_result
            elif isinstance(tool_result, FunctionInfo):
                # Extract the actual function from FunctionInfo
                if tool_result.single_fn:
                    tool_function = tool_result.single_fn
                elif tool_result.stream_fn:
                    tool_function = tool_result.stream_fn
                else:
                    raise ValueError("Tool function not found in FunctionInfo")
            elif callable(tool_result):
                tool_function = tool_result
            else:
                raise ValueError(f"Unexpected tool result type: {type(tool_result)}")

            # Execute the tool
            if input_data is not None:
                if isinstance(tool_function, Function):
                    result = await tool_function.ainvoke(input_data)
                elif asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function(input_data)
                else:
                    result = tool_function(input_data)
            elif isinstance(tool_function, Function):
                # Function objects require input, so pass None if no input_data
                result = await tool_function.ainvoke(None)
            elif asyncio.iscoroutinefunction(tool_function):
                result = await tool_function()
            else:
                result = tool_function()

            # Assert expected output if provided
            if expected_output is not None:
                assert result == expected_output, f"Expected {expected_output}, got {result}"

            return result

    @asynccontextmanager
    async def with_mocks(self):
        """
        Context manager that provides a mock builder for setting up dependencies.

        Usage:
            async with runner.with_mocks() as mock_builder:
                mock_builder.mock_llm("my_llm", "mocked response")
                result = await runner.test_tool_with_builder(
                    config_type=MyToolConfig,
                    builder=mock_builder,
                    input_data="test input"
                )
        """
        mock_builder = MockBuilder()
        try:
            yield mock_builder
        finally:
            pass

    async def test_tool_with_builder(
        self,
        config_type: type[FunctionBaseConfig],
        builder: MockBuilder,
        config_params: dict[str, typing.Any] | None = None,
        input_data: typing.Any = None,
        expected_output: typing.Any = None,
    ) -> typing.Any:
        """
        Test a tool with a pre-configured mock builder.

        Args:
            config_type: The tool configuration class
            builder: Pre-configured MockBuilder with mocked dependencies
            config_params: Parameters to pass to the config constructor
            input_data: Input data to pass to the tool
            expected_output: Expected output for assertion (optional)

        Returns:
            The tool's output
        """
        config_params = config_params or {}

        # Create tool configuration
        config = config_type(**config_params)

        # Get the registered tool function
        registry = GlobalTypeRegistry.get()
        try:
            tool_registration = registry.get_function(config_type)
        except KeyError:
            raise ValueError(
                f"Tool {config_type} is not registered. Make sure it's imported and registered with @register_function."
            )

        # Build the tool function with the provided builder
        async with tool_registration.build_fn(config, builder) as tool_result:

            # Handle different tool result types (same as above)
            if isinstance(tool_result, Function):
                tool_function = tool_result
            elif isinstance(tool_result, FunctionInfo):
                if tool_result.single_fn:
                    tool_function = tool_result.single_fn
                elif tool_result.stream_fn:
                    tool_function = tool_result.stream_fn
                else:
                    raise ValueError("Tool function not found in FunctionInfo")
            elif callable(tool_result):
                tool_function = tool_result
            else:
                raise ValueError(f"Unexpected tool result type: {type(tool_result)}")

            # Execute the tool
            if input_data is not None:
                if isinstance(tool_function, Function):
                    result = await tool_function.ainvoke(input_data)
                else:
                    maybe_result = tool_function(input_data)
                    result = await maybe_result if inspect.isawaitable(maybe_result) else maybe_result
            elif isinstance(tool_function, Function):
                # Function objects require input, so pass None if no input_data
                result = await tool_function.ainvoke(None)
            else:
                maybe_result = tool_function()
                result = await maybe_result if inspect.isawaitable(maybe_result) else maybe_result

            # Assert expected output if provided
            if expected_output is not None:
                assert result == expected_output, f"Expected {expected_output}, got {result}"

            return result


@asynccontextmanager
async def with_mocked_dependencies():
    """
    Convenience context manager for testing tools with mocked dependencies.

    Usage:
        async with with_mocked_dependencies() as (runner, mock_builder):
            mock_builder.mock_llm("my_llm", "mocked response")
            result = await runner.test_tool_with_builder(
                config_type=MyToolConfig,
                builder=mock_builder,
                input_data="test input"
            )
    """
    runner = ToolTestRunner()
    async with runner.with_mocks() as mock_builder:
        yield runner, mock_builder
