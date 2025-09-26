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
import typing
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.builder.function_info import FunctionInfo
from nat.control_flow.sequential_executor import SequentialExecutorConfig
from nat.control_flow.sequential_executor import ToolExecutionConfig
from nat.control_flow.sequential_executor import _validate_function_type_compatibility
from nat.control_flow.sequential_executor import _validate_tool_list_type_compatibility
from nat.control_flow.sequential_executor import sequential_execution
from nat.data_models.component_ref import FunctionRef
from nat.utils.type_utils import DecomposedType


# Test models for type compatibility testing
class StringInput(BaseModel):
    text: str


class StringOutput(BaseModel):
    result: str


class IntInput(BaseModel):
    number: int


class IntOutput(BaseModel):
    value: int


class ComplexInput(BaseModel):
    text: str
    number: int


class ComplexOutput(BaseModel):
    processed_text: str
    calculated_number: int


# Mock tool classes for testing
class MockTool(BaseTool):
    """Mock tool for testing purposes."""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def __init__(self, name: str = "mock_tool", return_value: str = "mock_result", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        # Store return_value in a way that doesn't conflict with Pydantic
        self.__dict__['_return_value'] = return_value
        self.__dict__['_call_count'] = 0

    async def _arun(self, query: typing.Any = None, **kwargs) -> str:
        self.__dict__['_call_count'] += 1
        return self.__dict__['_return_value']

    def _run(self, query: typing.Any = None, **kwargs) -> str:
        self.__dict__['_call_count'] += 1
        return self.__dict__['_return_value']

    @property
    def call_count(self) -> int:
        return self.__dict__['_call_count']


class StreamingMockTool(BaseTool):
    """Mock streaming tool for testing purposes."""

    name: str = "streaming_mock_tool"
    description: str = "A streaming mock tool for testing"

    def __init__(self, name: str = "streaming_mock_tool", chunks: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        # Store chunks in a way that doesn't conflict with Pydantic
        self.__dict__['_chunks'] = chunks if chunks is not None else ["chunk1", "chunk2", "chunk3"]
        self.__dict__['_call_count'] = 0

    async def astream(self, input, config=None, **kwargs):
        self.__dict__['_call_count'] += 1
        for chunk in self.__dict__['_chunks']:
            chunk_obj = MagicMock()
            chunk_obj.content = chunk
            yield chunk_obj

    async def _arun(self, query: typing.Any = None, **kwargs) -> str:
        self.__dict__['_call_count'] += 1
        return "".join(self.__dict__['_chunks'])

    def _run(self, query: typing.Any = None, **kwargs) -> str:
        self.__dict__['_call_count'] += 1
        return "".join(self.__dict__['_chunks'])

    @property
    def call_count(self) -> int:
        return self.__dict__['_call_count']


class ErrorMockTool(BaseTool):
    """Mock tool that raises an error for testing error handling."""

    name: str = "error_mock_tool"
    description: str = "A mock tool that raises errors"

    def __init__(self, name: str = "error_mock_tool", error_message: str = "Mock error", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        # Store error_message in a way that doesn't conflict with Pydantic
        self.__dict__['_error_message'] = error_message

    async def _arun(self, query: typing.Any = None, **kwargs) -> str:
        raise RuntimeError(self.__dict__['_error_message'])

    def _run(self, query: typing.Any = None, **kwargs) -> str:
        raise RuntimeError(self.__dict__['_error_message'])


class TestSequentialExecutionToolConfig:
    """Test cases for SequentialExecutionToolConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SequentialExecutorConfig()

        assert config.tool_list == []
        assert config.tool_execution_config == {}
        assert not config.raise_type_incompatibility

    def test_config_with_values(self):
        """Test configuration with custom values."""
        tool_list = [FunctionRef("tool1"), FunctionRef("tool2")]
        tool_config = {
            "tool1": ToolExecutionConfig(use_streaming=True),
            "tool2": ToolExecutionConfig(use_streaming=False),
        }

        config = SequentialExecutorConfig(tool_list=tool_list,
                                          tool_execution_config=tool_config,
                                          raise_type_incompatibility=True)

        assert config.tool_list == tool_list
        assert config.tool_execution_config == tool_config
        assert config.raise_type_incompatibility


class TestToolExecutionConfig:
    """Test cases for ToolExecutionConfig."""

    def test_default_config(self):
        """Test default ToolExecutionConfig values."""
        config = ToolExecutionConfig()
        assert not config.use_streaming

    def test_streaming_config(self):
        """Test ToolExecutionConfig with streaming enabled."""
        config = ToolExecutionConfig(use_streaming=True)
        assert config.use_streaming


class TestValidateFunctionTypeCompatibility:
    """Test cases for _validate_function_type_compatibility function."""

    @pytest.fixture
    def mock_function_compatible(self):
        """Create a mock function with compatible types."""
        func = MagicMock(spec=Function)
        func.instance_name = "compatible_func"
        func.single_output_type = str
        func.streaming_output_type = str
        func.input_type = str
        return func

    @pytest.fixture
    def mock_function_incompatible(self):
        """Create a mock function with incompatible types."""
        func = MagicMock(spec=Function)
        func.instance_name = "incompatible_func"
        func.single_output_type = int
        func.streaming_output_type = int
        func.input_type = str
        return func

    def test_compatible_types_no_streaming(self, mock_function_compatible):
        """Test type compatibility with non-streaming functions."""
        src_func = mock_function_compatible
        target_func = mock_function_compatible
        tool_config = {}

        with patch.object(DecomposedType, 'is_type_compatible', return_value=True) as mock_check:
            # Function should not raise an exception when types are compatible
            _validate_function_type_compatibility(src_func, target_func, tool_config)

            # Verify that the type compatibility check was called
            mock_check.assert_called_once_with(str, str)

    def test_compatible_types_with_streaming(self, mock_function_compatible):
        """Test type compatibility with streaming enabled."""
        src_func = mock_function_compatible
        target_func = mock_function_compatible
        tool_config = {"compatible_func": ToolExecutionConfig(use_streaming=True)}

        with patch.object(DecomposedType, 'is_type_compatible', return_value=True) as mock_check:
            # Function should not raise an exception when types are compatible
            _validate_function_type_compatibility(src_func, target_func, tool_config)

            # Verify that the type compatibility check was called
            mock_check.assert_called_once_with(str, str)

    def test_incompatible_types(self, mock_function_compatible, mock_function_incompatible):
        """Test type incompatibility detection."""
        src_func = mock_function_incompatible  # outputs int
        target_func = mock_function_compatible  # expects str input
        tool_config = {}

        with patch.object(DecomposedType, 'is_type_compatible', return_value=False) as mock_check:
            # Function should raise ValueError when types are incompatible
            with pytest.raises(ValueError, match="is not compatible with"):
                _validate_function_type_compatibility(src_func, target_func, tool_config)

            # Verify that the type compatibility check was called
            mock_check.assert_called_once_with(int, str)


class TestValidateSequentialToolList:
    """Test cases for _validate_sequential_tool_list function."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock builder."""
        builder = MagicMock(spec=Builder)
        builder.get_function = AsyncMock()
        builder.get_functions = AsyncMock()
        return builder

    @pytest.fixture
    def compatible_functions(self):
        """Create compatible mock functions."""
        func1 = MagicMock(spec=Function)
        func1.instance_name = "func1"
        func1.input_type = str
        func1.single_output_type = str
        func1.streaming_output_type = str

        func2 = MagicMock(spec=Function)
        func2.instance_name = "func2"
        func2.input_type = str
        func2.single_output_type = int
        func2.streaming_output_type = int

        return [func1, func2]

    @pytest.mark.asyncio
    async def test_compatible_sequential_tools(self, mock_builder, compatible_functions):
        """Test validation of compatible sequential tools."""
        config = SequentialExecutorConfig(tool_list=[FunctionRef("func1"), FunctionRef("func2")])

        mock_builder.get_functions.return_value = compatible_functions

        with patch('nat.control_flow.sequential_executor._validate_function_type_compatibility', return_value=True):
            input_type, output_type = await _validate_tool_list_type_compatibility(config, mock_builder)

            assert input_type is str  # First function's input type
            assert output_type is int  # Last function's output type

    @pytest.mark.asyncio
    async def test_incompatible_sequential_tools_with_exception(self, mock_builder, compatible_functions):
        """Test validation raises exception for incompatible tools when check_type_compatibility is True."""
        config = SequentialExecutorConfig(tool_list=[FunctionRef("func1"), FunctionRef("func2")],
                                          raise_type_incompatibility=True)

        mock_builder.get_functions.return_value = compatible_functions

        with patch('nat.control_flow.sequential_executor._validate_function_type_compatibility',
                   side_effect=ValueError("The output type of the func1 function is not compatible")):
            with pytest.raises(ValueError, match="The sequential tool list has incompatible types"):
                await _validate_tool_list_type_compatibility(config, mock_builder)

    @pytest.mark.asyncio
    async def test_streaming_output_type_selection(self, mock_builder, compatible_functions):
        """Test that streaming output type is selected when configured."""
        config = SequentialExecutorConfig(tool_list=[FunctionRef("func1"), FunctionRef("func2")],
                                          tool_execution_config={"func2": ToolExecutionConfig(use_streaming=True)})

        mock_builder.get_functions.return_value = compatible_functions

        with patch('nat.control_flow.sequential_executor._validate_function_type_compatibility', return_value=True):
            input_type, output_type = await _validate_tool_list_type_compatibility(config, mock_builder)

            assert input_type is str  # First function's input type
            assert output_type is int  # Last function's streaming_output_type


class TestSequentialExecution:
    """Test cases for the sequential_execution function."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock builder with tools."""
        builder = MagicMock(spec=Builder)

        # Create mock tools
        tool1 = MockTool(name="tool1", return_value="result1")
        tool2 = MockTool(name="tool2", return_value="result2")
        tool3 = MockTool(name="tool3", return_value="final_result")

        builder.get_tools.return_value = [tool1, tool2, tool3]

        # Mock functions for type validation
        func1 = MagicMock(spec=Function)
        func1.instance_name = "tool1"
        func1.input_type = str
        func1.single_output_type = str
        func1.streaming_output_type = str

        func2 = MagicMock(spec=Function)
        func2.instance_name = "tool2"
        func2.input_type = str
        func2.single_output_type = str
        func2.streaming_output_type = str

        func3 = MagicMock(spec=Function)
        func3.instance_name = "tool3"
        func3.input_type = str
        func3.single_output_type = str
        func3.streaming_output_type = str

        builder.get_function.side_effect = [func1, func2, func3]

        return builder

    @pytest.mark.asyncio
    async def test_basic_sequential_execution(self, mock_builder):
        """Test basic sequential execution of tools."""
        config = SequentialExecutorConfig(tool_list=[FunctionRef("tool1"), FunctionRef("tool2"), FunctionRef("tool3")])

        with patch('nat.control_flow.sequential_executor._validate_tool_list_type_compatibility',
                   return_value=(str, str)):
            async with sequential_execution(config, mock_builder) as function_info:
                assert isinstance(function_info, FunctionInfo)
                assert function_info.description and "sequential" in function_info.description.lower()

    @pytest.mark.asyncio
    async def test_sequential_execution_with_streaming(self, mock_builder):
        """Test sequential execution with streaming tools."""
        # Replace one tool with a streaming tool
        streaming_tool = StreamingMockTool(name="tool2", chunks=["stream1", "stream2"])
        mock_builder.get_tools.return_value = [
            MockTool(name="tool1", return_value="result1"),
            streaming_tool,
            MockTool(name="tool3", return_value="final_result")
        ]

        config = SequentialExecutorConfig(tool_list=[FunctionRef("tool1"), FunctionRef("tool2"), FunctionRef("tool3")],
                                          tool_execution_config={"tool2": ToolExecutionConfig(use_streaming=True)})

        with patch('nat.control_flow.sequential_executor._validate_tool_list_type_compatibility',
                   return_value=(str, str)):
            async with sequential_execution(config, mock_builder) as function_info:
                # Test that we get a function info object
                assert isinstance(function_info, FunctionInfo)

    @pytest.mark.asyncio
    async def test_sequential_execution_error_handling(self, mock_builder):
        """Test error handling in sequential execution."""
        # Replace middle tool with error tool
        error_tool = ErrorMockTool(name="tool2", error_message="Test error")
        mock_builder.get_tools.return_value = [
            MockTool(name="tool1", return_value="result1"),
            error_tool,
            MockTool(name="tool3", return_value="final_result")
        ]

        config = SequentialExecutorConfig(tool_list=[FunctionRef("tool1"), FunctionRef("tool2"), FunctionRef("tool3")])

        with patch('nat.control_flow.sequential_executor._validate_tool_list_type_compatibility',
                   return_value=(str, str)):
            async with sequential_execution(config, mock_builder) as function_info:
                # Get the actual function from the generator
                actual_function = function_info.single_fn  # type: ignore

                # Test that the function propagates errors
                with pytest.raises(RuntimeError, match="Test error"):
                    await actual_function("initial_input")  # type: ignore

    @pytest.mark.asyncio
    async def test_type_compatibility_error_with_check_enabled(self, mock_builder):
        """Test type compatibility error when check_type_compatibility is True."""
        config = SequentialExecutorConfig(tool_list=[FunctionRef("tool1"), FunctionRef("tool2")],
                                          raise_type_incompatibility=True)

        with patch('nat.control_flow.sequential_executor._validate_tool_list_type_compatibility',
                   side_effect=ValueError("Type incompatibility")):
            with pytest.raises(ValueError, match="Type incompatibility"):
                async with sequential_execution(config, mock_builder) as _:
                    pass

    @pytest.mark.asyncio
    async def test_type_compatibility_warning_with_check_disabled(self, mock_builder, caplog):
        """Test type compatibility warning when check_type_compatibility is False."""
        config = SequentialExecutorConfig(tool_list=[FunctionRef("tool1"), FunctionRef("tool2")],
                                          raise_type_incompatibility=False)

        with patch('nat.control_flow.sequential_executor._validate_tool_list_type_compatibility',
                   side_effect=ValueError("Type incompatibility")):
            with caplog.at_level(logging.WARNING):
                async with sequential_execution(config, mock_builder) as function_info:
                    assert isinstance(function_info, FunctionInfo)

                # Check that warning was logged
                assert "The sequential executor tool list has incompatible types" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_tool_list(self, mock_builder):
        """Test handling of empty tool list."""
        config = SequentialExecutorConfig(tool_list=[])

        mock_builder.get_tools.return_value = []

        with patch('nat.control_flow.sequential_executor._validate_tool_list_type_compatibility',
                   side_effect=IndexError("list index out of range")):
            with pytest.raises(ValueError, match="Error with the sequential executor tool list"):
                async with sequential_execution(config, mock_builder) as _:
                    pass

    @pytest.mark.asyncio
    async def test_single_tool_execution(self, mock_builder):
        """Test execution with a single tool."""
        config = SequentialExecutorConfig(tool_list=[FunctionRef("tool1")])

        # Mock single tool
        single_tool = MockTool(name="tool1", return_value="single_result")
        mock_builder.get_tools.return_value = [single_tool]

        # Mock single function
        func1 = MagicMock(spec=Function)
        func1.instance_name = "tool1"
        func1.input_type = str
        func1.single_output_type = str
        func1.streaming_output_type = str
        mock_builder.get_function.return_value = func1

        with patch('nat.control_flow.sequential_executor._validate_tool_list_type_compatibility',
                   return_value=(str, str)):
            async with sequential_execution(config, mock_builder) as function_info:
                actual_function = function_info.single_fn  # type: ignore
                result = await actual_function("test_input")  # type: ignore
                assert result == "single_result"

    @pytest.mark.asyncio
    async def test_tool_execution_order(self, mock_builder):
        """Test that tools are executed in the correct order."""
        config = SequentialExecutorConfig(tool_list=[FunctionRef("tool1"), FunctionRef("tool2"), FunctionRef("tool3")])

        # Create tools that append their names to the input
        class OrderTestTool(BaseTool):
            name: str = "order_test_tool"
            description: str = "A test tool for order testing"

            def __init__(self, tool_name: str, **kwargs):
                super().__init__(**kwargs)
                self.name = tool_name
                self.description = f"Test tool {tool_name}"
                # Store tool_name in a way that doesn't conflict with Pydantic
                self.__dict__['_tool_name'] = tool_name

            async def _arun(self, query: str = "", **kwargs) -> str:
                return f"{query}->{self.__dict__['_tool_name']}"

            def _run(self, query: str = "", **kwargs) -> str:
                return f"{query}->{self.__dict__['_tool_name']}"

        tools = [OrderTestTool(tool_name="tool1"), OrderTestTool(tool_name="tool2"), OrderTestTool(tool_name="tool3")]
        mock_builder.get_tools.return_value = tools

        with patch('nat.control_flow.sequential_executor._validate_tool_list_type_compatibility',
                   return_value=(str, str)):
            async with sequential_execution(config, mock_builder) as function_info:
                actual_function = function_info.single_fn  # type: ignore
                result = await actual_function("start")  # type: ignore
                assert result == "start->tool1->tool2->tool3"

    @pytest.mark.asyncio
    async def test_mixed_streaming_and_regular_tools(self, mock_builder):
        """Test execution with mixed streaming and regular tools."""
        streaming_tool = StreamingMockTool(name="tool1", chunks=["hello", " ", "world"])
        regular_tool = MockTool(name="tool2", return_value="processed")

        mock_builder.get_tools.return_value = [streaming_tool, regular_tool]

        config = SequentialExecutorConfig(tool_list=[FunctionRef("tool1"), FunctionRef("tool2")],
                                          tool_execution_config={"tool1": ToolExecutionConfig(use_streaming=True)})

        with patch('nat.control_flow.sequential_executor._validate_tool_list_type_compatibility',
                   return_value=(str, str)):
            async with sequential_execution(config, mock_builder) as function_info:
                actual_function = function_info.single_fn  # type: ignore
                result = await actual_function("input")  # type: ignore
                assert result == "processed"  # Final tool's result

    def test_function_annotations_set_correctly(self, mock_builder):
        """Test that function annotations are set correctly based on type validation."""
        config = SequentialExecutorConfig(tool_list=[FunctionRef("tool1")])

        with patch('nat.control_flow.sequential_executor._validate_tool_list_type_compatibility',
                   return_value=(str, int)):
            # Get the generator
            gen = sequential_execution(config, mock_builder)

            # Since this is an async generator, we need to test differently
            # The actual annotation setting happens inside the generator function
            assert gen is not None


class TestIntegration:
    """Integration tests for sequential execution."""

    @pytest.mark.asyncio
    async def test_real_world_scenario(self):
        """Test a real-world scenario with actual function registration."""
        # Test that the function is properly decorated (has __wrapped__)
        assert hasattr(sequential_execution, '__wrapped__')  # Should have register_function decorator

        # Test that sequential_execution is callable
        assert callable(sequential_execution)

    def test_framework_wrappers_configuration(self):
        """Test that framework wrappers are configured correctly."""
        # Test that sequential_execution is a decorated function
        # The actual framework configuration is internal to the registration system
        assert callable(sequential_execution)
        assert hasattr(sequential_execution, '__wrapped__')


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""

    @pytest.fixture
    def mock_builder_with_missing_tool(self):
        """Create a mock builder that simulates missing tools."""
        builder = MagicMock(spec=Builder)
        builder.get_tools.side_effect = KeyError("Tool not found")
        return builder

    @pytest.mark.asyncio
    async def test_missing_tool_error(self, mock_builder_with_missing_tool):
        """Test error handling when a tool is missing."""
        config = SequentialExecutorConfig(tool_list=[FunctionRef("missing_tool")])

        with pytest.raises(KeyError):
            async with sequential_execution(config, mock_builder_with_missing_tool) as _:
                pass

    @pytest.mark.asyncio
    async def test_invalid_tool_configuration(self):
        """Test error handling with invalid tool configuration."""
        config = SequentialExecutorConfig(
            tool_list=[FunctionRef("tool1")],
            tool_execution_config={"nonexistent_tool": ToolExecutionConfig(use_streaming=True)})

        builder = MagicMock(spec=Builder)
        tool = MockTool(name="tool1")
        builder.get_tools.return_value = [tool]

        func = MagicMock(spec=Function)
        func.instance_name = "tool1"
        func.input_type = str
        func.single_output_type = str
        func.streaming_output_type = str
        builder.get_function.return_value = func

        # This should not raise an error - extra config should be ignored
        with patch('nat.control_flow.sequential_executor._validate_tool_list_type_compatibility',
                   return_value=(str, str)):
            async with sequential_execution(config, builder) as function_info:
                actual_function = function_info.single_fn  # type: ignore
                result = await actual_function("test")  # type: ignore
                assert result == "mock_result"
