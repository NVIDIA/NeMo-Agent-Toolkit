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
import logging
import typing
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from aiq.builder.builder import Builder
from aiq.builder.function import Function
from aiq.builder.function_info import FunctionInfo
from aiq.cli.type_registry import GlobalTypeRegistry
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.object_store import ObjectStoreBaseConfig
from aiq.object_store.interfaces import ObjectStore
from aiq.runtime.loader import PluginTypes
from aiq.runtime.loader import discover_and_register_plugins

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

    def mock_its_strategy(self, name: str, mock_response: typing.Any):
        """Add a mock ITS strategy that returns a fixed response."""
        self._mocks[f"its_strategy_{name}"] = mock_response

    async def add_its_strategy(self, name: str, config):
        """Mock implementation (no‑op)."""
        pass

    async def get_its_strategy(self,
                               strategy_name: str,
                               pipeline_type: typing.Any = None,
                               stage_type: typing.Any = None):
        """Return a mock ITS strategy if one is configured."""
        key = f"its_strategy_{strategy_name}"
        if key in self._mocks:
            mock_strategy = MagicMock()
            # Provide common callable patterns used in tests
            mock_strategy.invoke = MagicMock(return_value=self._mocks[key])
            mock_strategy.ainvoke = AsyncMock(return_value=self._mocks[key])
            return mock_strategy
        raise ValueError(f"ITS strategy '{strategy_name}' not mocked. Use mock_its_strategy() to add it.")

    async def get_its_strategy_config(self,
                                      strategy_name: str,
                                      pipeline_type: typing.Any = None,
                                      stage_type: typing.Any = None):
        """Mock implementation."""
        pass

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
        pass

    async def set_workflow(self, config: FunctionBaseConfig) -> Function:
        """Mock implementation."""
        pass

    def get_workflow(self) -> Function:
        """Mock implementation."""
        pass

    def get_workflow_config(self) -> FunctionBaseConfig:
        """Mock implementation."""
        pass

    def get_tool(self, fn_name: str, wrapper_type):
        """Mock implementation."""
        pass

    async def add_llm(self, name: str, config):
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

    def get_llm_config(self, llm_name: str):
        """Mock implementation."""
        pass

    async def add_embedder(self, name: str, config):
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

    def get_embedder_config(self, embedder_name: str):
        """Mock implementation."""
        pass

    async def add_memory_client(self, name: str, config):
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

    def get_memory_client_config(self, memory_name: str):
        """Mock implementation."""
        pass

    async def add_retriever(self, name: str, config):
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

    async def get_retriever_config(self, retriever_name: str):
        """Mock implementation."""
        pass

    async def add_object_store(self, name: str, config: ObjectStoreBaseConfig):
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
        pass

    def get_user_manager(self):
        """Mock implementation."""
        mock_user = MagicMock()
        mock_user.get_id = MagicMock(return_value="test_user")
        return mock_user

    def get_function_dependencies(self, fn_name: str):
        """Mock implementation."""
        pass


class ToolTestRunner:
    """
    A test runner that enables isolated testing of AIQ tools without requiring
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
                if asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function(input_data)
                else:
                    result = tool_function(input_data)
            else:
                if asyncio.iscoroutinefunction(tool_function):
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
                elif tool_result.streaming_fn:
                    tool_function = tool_result.streaming_fn
                else:
                    raise ValueError("Tool function not found in FunctionInfo")
            elif callable(tool_result):
                tool_function = tool_result
            else:
                raise ValueError(f"Unexpected tool result type: {type(tool_result)}")

            # Execute the tool
            if input_data is not None:
                if asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function(input_data)
                else:
                    result = tool_function(input_data)
            else:
                if asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function()
                else:
                    result = tool_function()

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
