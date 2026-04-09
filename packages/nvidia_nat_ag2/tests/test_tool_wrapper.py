# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Test AG2 `tool_wrapper.py` — async tool wrapping for AG2."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.plugins.ag2.tool_wrapper import ag2_tool_wrapper


class MockInputSchema(BaseModel):
    """Mock input schema for tool wrapper."""

    query: str
    limit: int = 10


class TestAG2ToolWrapperBasic:
    """Test basic tool wrapper creation."""

    @pytest.fixture(name="mock_function")
    def fixture_mock_function(self):
        """Create a mock NeMo Agent Toolkit function for non-streaming."""
        mock_fn = Mock(spec=Function)
        mock_fn.description = "Test function description"
        mock_fn.input_schema = MockInputSchema
        mock_fn.has_streaming_output = False
        mock_fn.has_single_output = True
        mock_fn.acall_invoke = AsyncMock(return_value="test_result")
        mock_fn.acall_stream = AsyncMock()
        return mock_fn

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self):
        """Create a mock builder."""
        return Mock(spec=Builder)

    def test_tool_has_correct_name(self, mock_function, mock_builder):
        """Test that created Tool has the correct name."""
        tool = ag2_tool_wrapper("my_tool", mock_function, mock_builder)
        assert tool.name == "my_tool"

    def test_tool_has_correct_description(self, mock_function, mock_builder):
        """Test that created Tool has the correct description."""
        tool = ag2_tool_wrapper("my_tool", mock_function, mock_builder)
        assert tool.description == "Test function description"

    def test_tool_uses_name_when_no_description(self, mock_function, mock_builder):
        """Test that tool falls back to name when description is None."""
        mock_function.description = None
        tool = ag2_tool_wrapper("fallback_name", mock_function, mock_builder)
        assert tool.description == "fallback_name"

    def test_tool_has_json_schema(self, mock_function, mock_builder):
        """Test that created Tool has parameter schema from input_schema."""
        tool = ag2_tool_wrapper("my_tool", mock_function, mock_builder)
        schema = tool._func_schema
        assert schema is not None
        assert schema["function"]["parameters"] is not None

    def test_tool_no_schema_when_input_schema_none(self, mock_function, mock_builder):
        """Test that tool works when input_schema is None."""
        mock_function.input_schema = None
        tool = ag2_tool_wrapper("my_tool", mock_function, mock_builder)
        assert tool is not None

    def test_tool_func_is_callable(self, mock_function, mock_builder):
        """Test that the tool's underlying function is callable."""
        tool = ag2_tool_wrapper("my_tool", mock_function, mock_builder)
        assert callable(tool.func)


class TestAG2ToolWrapperAsync:
    """Test async tool execution — the core async improvement."""

    @pytest.fixture(name="mock_function")
    def fixture_mock_function(self):
        """Create a mock NeMo Agent Toolkit function."""
        mock_fn = Mock(spec=Function)
        mock_fn.description = "Async test function"
        mock_fn.input_schema = MockInputSchema
        mock_fn.has_streaming_output = False
        mock_fn.has_single_output = True
        mock_fn.acall_invoke = AsyncMock(return_value="async_result")
        mock_fn.acall_stream = AsyncMock()
        return mock_fn

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self):
        return Mock(spec=Builder)

    async def test_callable_ainvoke_returns_result(self, mock_function, mock_builder):
        """Test that the async invoke callable returns the correct result."""
        tool = ag2_tool_wrapper("my_tool", mock_function, mock_builder)

        # The tool's func is an async function — call and await it
        result = await tool.func(query="hello", limit=5)
        assert result == "async_result"
        mock_function.acall_invoke.assert_called_once_with(query="hello", limit=5)

    async def test_callable_ainvoke_is_coroutine(self, mock_function, mock_builder):
        """Test that the tool function is a proper coroutine function."""
        tool = ag2_tool_wrapper("my_tool", mock_function, mock_builder)
        assert asyncio.iscoroutinefunction(tool.func)

    async def test_callable_ainvoke_propagates_exceptions(self, mock_function, mock_builder):
        """Test that async invoke propagates exceptions."""
        mock_function.acall_invoke = AsyncMock(side_effect=ValueError("invoke failed"))
        tool = ag2_tool_wrapper("my_tool", mock_function, mock_builder)

        with pytest.raises(ValueError, match="invoke failed"):
            await tool.func(query="bad")


class TestAG2ToolWrapperStreaming:
    """Test streaming tool wrapping — collected into single result for AG2."""

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self):
        return Mock(spec=Builder)

    def _make_streaming_fn(self, stream_items):
        """Helper to create a mock function with streaming output."""
        mock_fn = Mock(spec=Function)
        mock_fn.description = "Streaming function"
        mock_fn.input_schema = MockInputSchema
        mock_fn.has_streaming_output = True
        mock_fn.has_single_output = False

        async def fake_stream(**kwargs):
            for item in stream_items:
                yield item

        mock_fn.acall_stream = fake_stream
        mock_fn.acall_invoke = AsyncMock()
        return mock_fn

    async def test_stream_collected_joins_strings(self, mock_builder):
        """Test that string stream chunks are joined."""
        mock_fn = self._make_streaming_fn(["Hello", " ", "World"])
        tool = ag2_tool_wrapper("stream_tool", mock_fn, mock_builder)

        result = await tool.func(query="test")
        assert result == "Hello World"

    async def test_stream_collected_returns_list_for_non_strings(self, mock_builder):
        """Test that non-string stream chunks are returned as list."""
        mock_fn = self._make_streaming_fn([{"a": 1}, {"b": 2}])
        tool = ag2_tool_wrapper("stream_tool", mock_fn, mock_builder)

        result = await tool.func(query="test")
        assert result == [{"a": 1}, {"b": 2}]

    async def test_stream_collected_returns_list_for_mixed_types(self, mock_builder):
        """Test that mixed string and non-string chunks are returned as a list (no joining)."""
        chunks = ["Hello", {"a": 1}, " ", 42, "World"]
        mock_fn = self._make_streaming_fn(chunks)
        tool = ag2_tool_wrapper("stream_tool", mock_fn, mock_builder)

        result = await tool.func(query="test")
        # Any non-string chunk disables joining — full list is returned, order preserved
        assert result == ["Hello", {"a": 1}, " ", 42, "World"]

    async def test_stream_collected_empty_returns_empty_string(self, mock_builder):
        """Test that empty stream returns empty string."""
        mock_fn = self._make_streaming_fn([])
        tool = ag2_tool_wrapper("stream_tool", mock_fn, mock_builder)

        result = await tool.func(query="test")
        assert result == ""

    async def test_stream_collected_is_coroutine(self, mock_builder):
        """Test that stream-collected function is a proper coroutine (not async generator)."""
        mock_fn = self._make_streaming_fn(["a"])
        tool = ag2_tool_wrapper("stream_tool", mock_fn, mock_builder)

        # Must be a coroutine function, NOT an async generator —
        # AG2's is_coroutine_callable checks for this
        assert asyncio.iscoroutinefunction(tool.func)

    def test_streaming_fn_selects_stream_path(self, mock_builder):
        """Test that streaming-only functions use the stream-collected path."""
        mock_fn = self._make_streaming_fn(["x"])
        tool = ag2_tool_wrapper("stream_tool", mock_fn, mock_builder)
        # The function should be named after the tool
        assert tool.func.__name__ == "stream_tool"

    def test_non_streaming_fn_selects_invoke_path(self, mock_builder):
        """Test that non-streaming functions use the invoke path."""
        mock_fn = Mock(spec=Function)
        mock_fn.description = "Non-streaming"
        mock_fn.input_schema = None
        mock_fn.has_streaming_output = False
        mock_fn.has_single_output = True
        mock_fn.acall_invoke = AsyncMock()
        mock_fn.acall_stream = AsyncMock()

        tool = ag2_tool_wrapper("invoke_tool", mock_fn, mock_builder)
        assert tool.func.__name__ == "invoke_tool"


class TestAG2ToolWrapperAG2Integration:
    """Test that tools work with AG2's async execution model."""

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self):
        return Mock(spec=Builder)

    async def test_ag2_a_execute_function_compatibility(self, mock_builder):
        """Test that tools work with AG2's a_execute_function pattern.

        AG2's a_execute_function does:
            if is_coroutine_callable(func):
                content = await func(**arguments)
            else:
                content = func(**arguments)
            if inspect.isawaitable(content):
                content = await content

        Our async callables must pass is_coroutine_callable check.
        """
        from autogen.fast_depends.utils import is_coroutine_callable

        mock_fn = Mock(spec=Function)
        mock_fn.description = "AG2 compatible tool"
        mock_fn.input_schema = None
        mock_fn.has_streaming_output = False
        mock_fn.has_single_output = True
        mock_fn.acall_invoke = AsyncMock(return_value="ag2_result")
        mock_fn.acall_stream = AsyncMock()

        tool = ag2_tool_wrapper("ag2_tool", mock_fn, mock_builder)

        # AG2 checks this to decide whether to await
        assert is_coroutine_callable(tool.func)

        # Simulate AG2's execution path
        result = await tool.func(query="test")
        assert result == "ag2_result"
