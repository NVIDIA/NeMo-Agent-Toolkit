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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.builder.function import Function
from nat.plugins.strands.tool_wrapper import _json_schema_from_pydantic
from nat.plugins.strands.tool_wrapper import _to_error_result
from nat.plugins.strands.tool_wrapper import _to_tool_result
from nat.plugins.strands.tool_wrapper import strands_tool_wrapper


class DummyInput(BaseModel):
    """Dummy input model for testing."""
    value: int


class DummyFunction:
    """Dummy function with simple input/output."""

    def __init__(self):
        self.description = "A dummy function"
        self.input_schema = DummyInput

    async def acall_invoke(self, **kwargs):
        return {"result": "success"}


class TestJsonSchemaFromPydantic:
    """Tests for _json_schema_from_pydantic function."""

    def test_json_schema_from_pydantic_basic(self):
        """Test basic JSON schema generation."""
        schema = _json_schema_from_pydantic(DummyInput)
        assert "json" in schema
        assert "properties" in schema["json"]

    def test_json_schema_from_pydantic_exception_handling(self):
        """Test exception handling in schema generation."""

        class BadModel:

            @staticmethod
            def model_json_schema():
                raise Exception("Schema generation failed")

        schema = _json_schema_from_pydantic(BadModel)  # type: ignore
        assert schema == {"json": {}}


class TestToToolResult:
    """Tests for _to_tool_result function."""

    def test_to_tool_result_with_dict(self):
        """Test _to_tool_result with dictionary value."""
        result = _to_tool_result("tool_123", {"key": "value"})
        assert result["toolUseId"] == "tool_123"
        assert result["status"] == "success"
        assert result["content"] == [{"json": {"key": "value"}}]

    def test_to_tool_result_with_string(self):
        """Test _to_tool_result with string value."""
        result = _to_tool_result("tool_789", "hello world")
        assert result["toolUseId"] == "tool_789"
        assert result["status"] == "success"
        assert result["content"] == [{"text": "hello world"}]


class TestToErrorResult:
    """Tests for _to_error_result function."""

    def test_to_error_result_with_exception(self):
        """Test _to_error_result with an exception."""
        error = ValueError("Something went wrong")
        result = _to_error_result("tool_error", error)

        assert result["toolUseId"] == "tool_error"
        assert result["status"] == "error"
        assert "ValueError" in result["content"][0]["text"]


class TestStrandsToolWrapper:
    """Tests for strands_tool_wrapper function."""

    @pytest.fixture
    def mock_function(self):
        """Create a mock Function object."""
        func = MagicMock(spec=Function)
        func.description = "Test function"
        func.input_schema = DummyInput
        return func

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder object."""
        return MagicMock()

    @patch("nat.plugins.strands.tool_wrapper.NATFunctionAgentTool")
    def test_strands_tool_wrapper_creation(self, mock_nat_tool, mock_function, mock_builder):
        """Test that strands_tool_wrapper creates NATFunctionAgentTool."""
        result = strands_tool_wrapper("test_tool", mock_function, mock_builder)

        # Verify that NATFunctionAgentTool was created and returned
        mock_nat_tool.assert_called_once()
        assert result is not None


class TestNATFunctionAgentTool:
    """Tests for NATFunctionAgentTool class."""

    @pytest.fixture
    def mock_function(self):
        """Create a mock Function object."""
        func = MagicMock(spec=Function)
        func.acall_invoke = AsyncMock(return_value="test result")
        return func

    @pytest.fixture
    def tool_spec(self):
        """Create a mock ToolSpec."""
        return {"name": "test_tool", "description": "Test tool", "input_schema": {"type": "object", "properties": {}}}

    def test_nat_function_agent_tool_initialization(self, mock_function, tool_spec):
        """Test NATFunctionAgentTool initialization."""
        from nat.plugins.strands.tool_wrapper import NATFunctionAgentTool

        tool = NATFunctionAgentTool("test_tool", "Test desc", tool_spec, mock_function)

        assert tool.tool_name == "test_tool"
        assert tool._fn == mock_function

    @pytest.mark.asyncio
    async def test_nat_function_agent_tool_stream_success(self, mock_function, tool_spec):
        """Test successful tool execution."""
        from nat.plugins.strands.tool_wrapper import NATFunctionAgentTool

        tool = NATFunctionAgentTool("test_tool", "Test desc", tool_spec, mock_function)

        # Mock the tool use - need to use get method properly
        tool_use = MagicMock()
        tool_use.get.return_value = {"param": "value"}
        tool_use.id = "tool_123"

        # Execute the tool
        results = []
        async for result in tool.stream(tool_use, {}):
            results.append(result)

        # Verify the function was called with the input
        mock_function.acall_invoke.assert_called_once()

        # Verify we got a result
        assert len(results) == 1
        # Check the result structure - it should contain tool_result
        result_event = results[0]
        assert "tool_result" in result_event
        tool_result = result_event["tool_result"]
        assert tool_result["status"] == "success"
        assert "content" in tool_result

    @pytest.mark.asyncio
    async def test_nat_function_agent_tool_stream_error(self, mock_function, tool_spec):
        """Test tool execution with error."""
        from nat.plugins.strands.tool_wrapper import NATFunctionAgentTool

        # Make the function raise an error
        mock_function.acall_invoke = AsyncMock(side_effect=ValueError("Test error"))

        tool = NATFunctionAgentTool("test_tool", "Test desc", tool_spec, mock_function)

        tool_use = MagicMock()
        tool_use.get.return_value = {"param": "value"}
        tool_use.id = "tool_456"

        # Execute the tool
        results = []
        async for result in tool.stream(tool_use, {}):
            results.append(result)

        # Should get an error result
        assert len(results) == 1
        result_event = results[0]
        assert "tool_result" in result_event
        tool_result = result_event["tool_result"]
        assert tool_result["status"] == "error"
        assert "ValueError" in tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_nat_function_agent_tool_streaming_function(self, tool_spec):
        """Test tool with streaming function."""
        from nat.plugins.strands.tool_wrapper import NATFunctionAgentTool

        # Create a mock streaming function
        mock_function = MagicMock(spec=Function)
        mock_function.has_streaming_output = True
        mock_function.has_single_output = False

        async def mock_stream(**kwargs):
            yield "chunk1"
            yield "chunk2"
            yield "final_chunk"

        mock_function.acall_stream = mock_stream

        tool = NATFunctionAgentTool("streaming_tool", "Streaming desc", tool_spec, mock_function)

        tool_use = MagicMock()
        tool_use.get.return_value = {"param": "value"}
        tool_use.__getitem__ = MagicMock(return_value="stream_tool_123")  # For toolUseId access

        # Execute the streaming tool
        results = []
        async for result in tool.stream(tool_use, {}):
            results.append(result)

        # Should get stream events plus final result
        assert len(results) > 1

        # Last result should be the final tool result
        final_result = results[-1]
        assert "tool_result" in final_result
        tool_result = final_result["tool_result"]
        assert tool_result["status"] == "success"
        # toolUseId comes from tool_use.get("toolUseId", "unknown")
        assert "toolUseId" in tool_result

    @pytest.mark.asyncio
    async def test_nat_function_agent_tool_streaming_with_error(self, tool_spec):
        """Test streaming tool with error."""
        from nat.plugins.strands.tool_wrapper import NATFunctionAgentTool

        mock_function = MagicMock(spec=Function)
        mock_function.has_streaming_output = True
        mock_function.has_single_output = False

        async def mock_stream_error(**kwargs):
            yield "chunk1"
            raise RuntimeError("Streaming error")

        mock_function.acall_stream = mock_stream_error

        tool = NATFunctionAgentTool("error_stream_tool", "Error stream desc", tool_spec, mock_function)

        tool_use = MagicMock()
        tool_use.get.return_value = {"param": "value"}
        tool_use.__getitem__ = MagicMock(return_value="error_stream_456")  # For toolUseId access

        # Execute the tool
        results = []
        async for result in tool.stream(tool_use, {}):
            results.append(result)

        # Should get stream events and then an error result
        assert len(results) >= 1
        # Last result should be the error
        final_result = results[-1]
        assert "tool_result" in final_result
        tool_result = final_result["tool_result"]
        assert tool_result["status"] == "error"
        assert "RuntimeError" in tool_result["content"][0]["text"]

    def test_nat_function_agent_tool_properties(self, mock_function):
        """Test NATFunctionAgentTool properties."""
        from nat.plugins.strands.tool_wrapper import NATFunctionAgentTool

        custom_tool_spec = {"name": "prop_tool", "description": "Property test", "inputSchema": {"type": "object"}}
        tool = NATFunctionAgentTool("prop_tool", "Property test", custom_tool_spec, mock_function)

        assert tool.tool_name == "prop_tool"
        # The tool_spec gets modified during construction, so check key fields
        assert tool.tool_spec["name"] == "prop_tool"
        assert tool.tool_spec["description"] == "Property test"
        assert "inputSchema" in tool.tool_spec
        assert tool.tool_type == "function"

    @pytest.mark.asyncio
    async def test_nat_function_agent_tool_empty_input(self, mock_function, tool_spec):
        """Test tool execution with empty input."""
        from nat.plugins.strands.tool_wrapper import NATFunctionAgentTool

        tool = NATFunctionAgentTool("empty_input_tool", "Empty input test", tool_spec, mock_function)

        tool_use = MagicMock()
        tool_use.get.return_value = None  # Empty input
        tool_use.__getitem__ = MagicMock(return_value="empty_123")  # For toolUseId access

        # Execute the tool
        results = []
        async for result in tool.stream(tool_use, {}):
            results.append(result)

        # Should still work with empty input
        assert len(results) == 1
        mock_function.acall_invoke.assert_called_once_with()  # Called with no args


class TestToolWrapperEdgeCases:
    """Tests for edge cases in tool wrapper functionality."""

    def test_strands_tool_wrapper_no_input_schema(self):
        """Test strands_tool_wrapper with no input schema."""
        mock_function = MagicMock(spec=Function)
        mock_function.input_schema = None
        mock_builder = MagicMock()

        with pytest.raises(ValueError, match="Tool 'no_schema_tool' must define an input schema"):
            strands_tool_wrapper("no_schema_tool", mock_function, mock_builder)

    def test_json_schema_from_pydantic_with_title(self):
        """Test _json_schema_from_pydantic removes title field."""

        class TestModel(BaseModel):
            value: str

        schema = _json_schema_from_pydantic(TestModel)

        # Should have json key but no title
        assert "json" in schema
        assert "title" not in schema["json"]
        assert "properties" in schema["json"]

    def test_to_tool_result_with_list(self):
        """Test _to_tool_result with list value."""
        result = _to_tool_result("list_tool_123", ["item1", "item2"])

        assert result["toolUseId"] == "list_tool_123"
        assert result["status"] == "success"
        assert result["content"] == [{"json": ["item1", "item2"]}]

    def test_to_tool_result_with_tuple(self):
        """Test _to_tool_result with tuple value."""
        result = _to_tool_result("tuple_tool_456", ("a", "b", "c"))

        assert result["toolUseId"] == "tuple_tool_456"
        assert result["status"] == "success"
        assert result["content"] == [{"json": ("a", "b", "c")}]

    def test_to_error_result_with_custom_exception(self):
        """Test _to_error_result with custom exception."""

        class CustomError(Exception):
            pass

        error = CustomError("Custom error message")
        result = _to_error_result("custom_error_789", error)

        assert result["toolUseId"] == "custom_error_789"
        assert result["status"] == "error"
        assert "CustomError: Custom error message" in result["content"][0]["text"]
