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

from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.plugins.adk.tool_wrapper import google_adk_tool_wrapper
from nat.plugins.adk.tool_wrapper import resolve_type

# ----------------------------
# Dummy Models for Testing
# ----------------------------


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    result: int


class InnerModel(BaseModel):
    x: int


class OuterModel(BaseModel):
    inner: InnerModel
    y: str


class NestedOutput(BaseModel):
    result: int


# ----------------------------
# Dummy Function Implementations
# ----------------------------


class DummyFunction:
    """Dummy function with simple input/output."""

    def __init__(self):
        self.description = "Dummy ADK function"
        self.config = type('Config', (), {'type': 'dummy_adk_func'})
        self.has_single_output = True
        self.has_streaming_output = False
        self.input_schema = DummyInput
        self.single_output_schema = DummyOutput
        self.streaming_output_schema = None

    async def acall_invoke(self, *args, **_kwargs):
        input_obj = args[0]
        return DummyOutput(result=input_obj.value * 3)


class DummyNestedFunction:
    """Dummy function using nested BaseModel for input."""

    def __init__(self):
        self.description = "Nested ADK function"
        self.config = type('Config', (), {'type': 'nested_adk_func'})
        self.has_single_output = True
        self.has_streaming_output = False
        self.input_schema = OuterModel
        self.single_output_schema = NestedOutput
        self.streaming_output_schema = None

    async def acall_invoke(self, *args, **_kwargs):
        outer = args[0]
        return NestedOutput(result=outer.inner.x + len(outer.y))


class DummyStreamingFunction:
    """Dummy function that simulates streaming output."""

    def __init__(self):
        self.description = "Streaming ADK function"
        self.config = type('Config', (), {'type': 'streaming_adk_func'})
        self.has_single_output = False
        self.has_streaming_output = True
        self.input_schema = DummyInput
        self.streaming_output_schema = DummyOutput
        self.single_output_schema = None

    async def acall_stream(self, *args, **_kwargs):
        """Simulate streaming output.

        Args:
            *args: Positional arguments, expects first arg to be DummyInput.
            **_kwargs: Keyword arguments (not used).

        Yields:
            DummyOutput: Streaming output items.
        """
        async for item in self._astream(args[0]):
            yield item

    async def _astream(self, value: Any):
        """Async generator to yield streaming output.

        Args:
            value (Any): Input value, expects DummyInput.
        Yields:
            DummyOutput: Streaming output items.
        """
        for i in range(2):
            yield DummyOutput(result=value.value + i)


# ----------------------------
# Pytest Unit Tests
# ----------------------------


def test_resolve_type():
    """Test the resolve_type function."""

    union_type = str | None
    resolved = resolve_type(union_type)
    assert resolved is str

    # Test with Optional type
    optional_type = int | None
    resolved = resolve_type(optional_type)
    assert resolved is int

    # Test with regular type
    regular_type = str
    resolved = resolve_type(regular_type)
    assert resolved is str


@patch('google.adk.tools.function_tool.FunctionTool')
@pytest.mark.asyncio
async def test_google_adk_tool_wrapper_simple_function(mock_function_tool):
    """Test the ADK tool wrapper with a simple function."""
    dummy_fn = DummyFunction()
    mock_builder = MagicMock()

    # Mock FunctionTool constructor
    mock_tool_instance = MagicMock()
    mock_function_tool.return_value = mock_tool_instance

    # Call the wrapper
    result = google_adk_tool_wrapper('dummy_adk_func', dummy_fn, mock_builder)

    # Verify FunctionTool was called
    assert mock_function_tool.called
    assert result == mock_tool_instance
    # Verify the callable was created with correct metadata
    call_args = mock_function_tool.call_args[0][0]
    assert call_args.__name__ == 'dummy_adk_func'
    assert call_args.__doc__ == "Dummy ADK function"


@patch('google.adk.tools.function_tool.FunctionTool')
@pytest.mark.asyncio
async def test_google_adk_tool_wrapper_nested_function(mock_function_tool):
    """Test the ADK tool wrapper with nested BaseModel input."""
    dummy_fn = DummyNestedFunction()
    mock_builder = MagicMock()

    mock_tool_instance = MagicMock()
    mock_function_tool.return_value = mock_tool_instance

    # Call the wrapper
    result = google_adk_tool_wrapper('nested_adk_func', dummy_fn, mock_builder)

    # Verify FunctionTool was called
    assert mock_function_tool.called
    assert result == mock_tool_instance

    # Verify the callable was created with correct metadata
    call_args = mock_function_tool.call_args[0][0]
    assert call_args.__name__ == 'nested_adk_func'
    assert call_args.__doc__ == "Nested ADK function"


@patch('google.adk.tools.function_tool.FunctionTool')
@pytest.mark.asyncio
async def test_google_adk_tool_wrapper_streaming_function(mock_function_tool):
    """Test the ADK tool wrapper with streaming function."""
    dummy_fn = DummyStreamingFunction()
    mock_builder = MagicMock()

    mock_tool_instance = MagicMock()
    mock_function_tool.return_value = mock_tool_instance

    # Call the wrapper
    result = google_adk_tool_wrapper('streaming_adk_func', dummy_fn, mock_builder)

    # Verify FunctionTool was called
    assert mock_function_tool.called
    assert result == mock_tool_instance

    # Verify the callable was created for streaming
    call_args = mock_function_tool.call_args[0][0]
    assert call_args.__name__ == 'streaming_adk_func'
    assert call_args.__doc__ == "Streaming ADK function"


@pytest.mark.asyncio
async def test_callable_ainvoke_functionality():
    """Test the callable_ainvoke wrapper functionality."""
    dummy_fn = DummyFunction()

    # Test the actual callable functionality
    with patch('google.adk.tools.function_tool.FunctionTool') as mock_function_tool:
        mock_tool_instance = MagicMock()
        mock_function_tool.return_value = mock_tool_instance

        google_adk_tool_wrapper('dummy_adk_func', dummy_fn, None)

        # Get the callable that was passed to FunctionTool
        callable_func = mock_function_tool.call_args[0][0]

        # Test calling it
        dummy_input = DummyInput(value=5)
        result = await callable_func(dummy_input)

        # Should call the original function's acall_invoke
        assert isinstance(result, DummyOutput)
        assert result.result == 15  # 5 * 3
    dummy_fn = DummyStreamingFunction()

    # Test the actual streaming callable functionality
    with patch('google.adk.tools.function_tool.FunctionTool') as mock_function_tool:
        mock_tool_instance = MagicMock()
        mock_function_tool.return_value = mock_tool_instance

        google_adk_tool_wrapper('streaming_adk_func', dummy_fn, None)

        # Get the callable that was passed to FunctionTool
        callable_func = mock_function_tool.call_args[0][0]

        # Test calling it with streaming
        dummy_input = DummyInput(value=10)
        results = []
        async for item in callable_func(dummy_input):
            results.append(item)

        # Should get 2 items from the streaming function
        assert len(results) == 2
        assert results[0].result == 10  # 10 + 0
        assert results[1].result == 11  # 10 + 1
