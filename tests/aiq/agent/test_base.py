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
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph

from aiq.agent.base import BaseAgent


class MockBaseAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing."""

    def __init__(self, detailed_logs=True):
        # Create simple mock objects without pydantic restrictions
        self.llm = Mock()
        self.tools = [Mock(), Mock()]
        self.tools[0].name = "Tool A"
        self.tools[1].name = "Tool B"
        self.callbacks = []
        self.detailed_logs = detailed_logs

    async def _build_graph(self, state_schema: type) -> CompiledGraph:
        """Mock implementation."""
        return Mock(spec=CompiledGraph)


@pytest.fixture
def base_agent():
    """Create a mock agent for testing with detailed logs enabled."""
    return MockBaseAgent(detailed_logs=True)


@pytest.fixture
def base_agent_no_logs():
    """Create a mock agent for testing with detailed logs disabled."""
    return MockBaseAgent(detailed_logs=False)


class TestStreamLLM:
    """Test the _stream_llm method."""

    async def test_successful_streaming(self, base_agent):
        """Test successful streaming without retries."""
        mock_runnable = Mock()
        mock_event1 = Mock()
        mock_event1.content = "Hello "
        mock_event2 = Mock()
        mock_event2.content = "world!"

        async def mock_astream(inputs, config=None):
            for event in [mock_event1, mock_event2]:
                yield event

        mock_runnable.astream = mock_astream

        inputs = {"messages": [HumanMessage(content="test")]}
        config = RunnableConfig(callbacks=[])

        result = await base_agent._stream_llm(mock_runnable, inputs, config)

        assert isinstance(result, AIMessage)
        assert result.content == "Hello world!"

    async def test_streaming_error_propagation(self, base_agent):
        """Test that streaming errors are propagated to the automatic retry system."""
        mock_runnable = Mock()

        async def mock_astream(inputs, config=None):
            raise Exception("Network error")
            yield  # Never executed but makes this an async generator

        mock_runnable.astream = mock_astream

        inputs = {"messages": [HumanMessage(content="test")]}

        # Error should be propagated (retry is handled automatically by underlying client)
        with pytest.raises(Exception, match="Network error"):
            await base_agent._stream_llm(mock_runnable, inputs)

    async def test_streaming_empty_content(self, base_agent):
        """Test streaming with empty content."""
        mock_runnable = Mock()
        mock_event = Mock()
        mock_event.content = ""

        async def mock_astream(inputs, config=None):
            yield mock_event

        mock_runnable.astream = mock_astream

        inputs = {"messages": [HumanMessage(content="test")]}

        result = await base_agent._stream_llm(mock_runnable, inputs)

        assert isinstance(result, AIMessage)
        assert result.content == ""


class TestCallLLM:
    """Test the _call_llm method."""

    async def test_successful_llm_call(self, base_agent):
        """Test successful LLM call."""
        messages = [HumanMessage(content="test")]
        mock_response = AIMessage(content="Response content")

        base_agent.llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await base_agent._call_llm(messages)

        assert isinstance(result, AIMessage)
        assert result.content == "Response content"
        base_agent.llm.ainvoke.assert_called_once_with(messages)

    async def test_llm_call_error_propagation(self, base_agent):
        """Test that LLM call errors are propagated to the automatic retry system."""
        messages = [HumanMessage(content="test")]

        base_agent.llm.ainvoke = AsyncMock(side_effect=Exception("API error"))

        # Error should be propagated (retry is handled automatically by underlying client)
        with pytest.raises(Exception, match="API error"):
            await base_agent._call_llm(messages)

    async def test_llm_call_content_conversion(self, base_agent):
        """Test that LLM response content is properly converted to string."""
        messages = [HumanMessage(content="test")]
        # Mock response that simulates non-string content that gets converted
        mock_response = Mock()
        mock_response.content = 123

        base_agent.llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await base_agent._call_llm(messages)

        assert isinstance(result, AIMessage)
        assert result.content == "123"


class TestCallTool:
    """Test the _call_tool method."""

    async def test_successful_tool_call(self, base_agent):
        """Test successful tool call."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}
        config = RunnableConfig(callbacks=[])

        tool.ainvoke = AsyncMock(return_value="Tool response")

        result = await base_agent._call_tool(tool, tool_input, config)

        assert isinstance(result, ToolMessage)
        assert result.content == "Tool response"
        assert result.name == tool.name
        assert result.tool_call_id == tool.name
        tool.ainvoke.assert_called_once_with(tool_input, config=config)

    async def test_tool_call_with_retries_success_on_second_attempt(self, base_agent):
        """Test that tool call succeeds on second attempt with retry logic."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}
        config = RunnableConfig(callbacks=[])

        tool.ainvoke = AsyncMock(side_effect=[Exception("Network error"), "Tool response"])

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await base_agent._call_tool(tool, tool_input, config, max_retries=2)

        assert isinstance(result, ToolMessage)
        assert result.content == "Tool response"
        assert tool.ainvoke.call_count == 2
        mock_sleep.assert_called_once_with(2)  # 2^1 = 2 seconds for first retry

    async def test_tool_call_all_retries_exhausted(self, base_agent):
        """Test that tool call returns error message when all retries are exhausted."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}

        tool.ainvoke = AsyncMock(side_effect=Exception("Persistent error"))

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await base_agent._call_tool(tool, tool_input, max_retries=2)

        assert isinstance(result, ToolMessage)
        assert "Tool call failed after all retry attempts" in result.content
        assert "Persistent error" in result.content
        assert tool.ainvoke.call_count == 2  # 2 total attempts with max_retries=2
        # Should have called sleep once: 2^1=2 (only first attempt fails and retries)
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_once_with(2)

    async def test_tool_call_none_response(self, base_agent):
        """Test handling of None response from tool."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}

        tool.ainvoke = AsyncMock(return_value=None)

        result = await base_agent._call_tool(tool, tool_input)

        assert isinstance(result, ToolMessage)
        assert "provided an empty response" in result.content
        assert result.name == tool.name

    async def test_tool_call_empty_string_response(self, base_agent):
        """Test handling of empty string response from tool."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}

        tool.ainvoke = AsyncMock(return_value="")

        result = await base_agent._call_tool(tool, tool_input)

        assert isinstance(result, ToolMessage)
        assert "provided an empty response" in result.content
        assert result.name == tool.name

    async def test_tool_call_zero_retries(self, base_agent):
        """Test behavior with zero retries."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}

        tool.ainvoke = AsyncMock(side_effect=Exception("Error"))

        result = await base_agent._call_tool(tool, tool_input, max_retries=0)

        # With max_retries=0, no attempts are made (range(1, 1) is empty)
        assert isinstance(result, ToolMessage)
        assert "Tool call failed after all retry attempts" in result.content
        assert tool.ainvoke.call_count == 0


class TestLogToolResponse:
    """Test the _log_tool_response method."""

    def test_log_tool_response_with_detailed_logs(self, base_agent, caplog):
        """Test logging when detailed_logs is True."""
        tool_name = "TestTool"
        tool_input = {"query": "test"}
        tool_response = "Short response"

        with caplog.at_level(logging.INFO):
            base_agent._log_tool_response(tool_name, tool_input, tool_response)

        assert "Calling tools: TestTool" in caplog.text
        assert "Short response" in caplog.text

    def test_log_tool_response_without_detailed_logs(self, base_agent_no_logs, caplog):
        """Test logging when detailed_logs is False."""
        tool_name = "TestTool"
        tool_input = {"query": "test"}
        tool_response = "Short response"

        with caplog.at_level(logging.INFO):
            base_agent_no_logs._log_tool_response(tool_name, tool_input, tool_response)

        assert "Calling tools: TestTool" not in caplog.text

    def test_log_tool_response_with_long_response(self, base_agent, caplog):
        """Test logging with response that exceeds max_chars."""
        tool_name = "TestTool"
        tool_input = {"query": "test"}
        tool_response = "x" * 1500  # Longer than default max_chars (1000)

        with caplog.at_level(logging.INFO):
            base_agent._log_tool_response(tool_name, tool_input, tool_response, max_chars=1000)

        assert "Calling tools: TestTool" in caplog.text
        assert "...(rest of response truncated)" in caplog.text
        assert len(caplog.text) < len(tool_response)

    def test_log_tool_response_with_custom_max_chars(self, base_agent, caplog):
        """Test logging with response that exceeds custom max_chars."""
        tool_name = "TestTool"
        tool_input = {"query": "test"}
        tool_response = "x" * 100

        with caplog.at_level(logging.INFO):
            base_agent._log_tool_response(tool_name, tool_input, tool_response, max_chars=50)

        assert "Calling tools: TestTool" in caplog.text
        assert "...(rest of response truncated)" in caplog.text

    def test_log_tool_response_with_complex_input(self, base_agent, caplog):
        """Test logging with complex tool input."""
        tool_name = "TestTool"
        tool_input = {"query": "test", "nested": {"key": "value"}}
        tool_response = "Response"

        with caplog.at_level(logging.INFO):
            base_agent._log_tool_response(tool_name, tool_input, tool_response)

        assert "Calling tools: TestTool" in caplog.text
        assert str(tool_input) in caplog.text


class TestParseJson:
    """Test the _parse_json method."""

    def test_parse_valid_json(self, base_agent):
        """Test parsing valid JSON."""
        json_string = '{"key": "value", "number": 42}'

        result = base_agent._parse_json(json_string)

        assert result == {"key": "value", "number": 42}

    def test_parse_empty_json(self, base_agent):
        """Test parsing empty JSON object."""
        json_string = '{}'

        result = base_agent._parse_json(json_string)

        assert result == {}

    def test_parse_json_array(self, base_agent):
        """Test parsing JSON array."""
        json_string = '[1, 2, 3]'

        result = base_agent._parse_json(json_string)

        assert result == [1, 2, 3]

    def test_parse_invalid_json(self, base_agent):
        """Test parsing invalid JSON."""
        json_string = '{"key": "value"'  # Missing closing brace

        result = base_agent._parse_json(json_string)

        assert "error" in result
        assert "JSON parsing failed" in result["error"]
        assert result["original_string"] == json_string

    def test_parse_malformed_json(self, base_agent):
        """Test parsing completely malformed JSON."""
        json_string = 'not json at all'

        result = base_agent._parse_json(json_string)

        assert "error" in result
        assert "JSON parsing failed" in result["error"]
        assert result["original_string"] == json_string

    def test_parse_json_with_unexpected_error(self, base_agent):
        """Test parsing JSON with unexpected error."""
        json_string = '{"key": "value"}'

        with patch('json.loads', side_effect=ValueError("Unexpected error")):
            result = base_agent._parse_json(json_string)

        assert "error" in result
        assert "Unexpected parsing error" in result["error"]
        assert result["original_string"] == json_string

    def test_parse_json_with_special_characters(self, base_agent):
        """Test parsing JSON with special characters."""
        json_string = '{"message": "Hello\\nWorld", "emoji": "😀"}'

        result = base_agent._parse_json(json_string)

        assert result == {"message": "Hello\nWorld", "emoji": "😀"}

    def test_parse_nested_json(self, base_agent):
        """Test parsing nested JSON."""
        json_string = '{"outer": {"inner": {"deep": "value"}}}'

        result = base_agent._parse_json(json_string)

        assert result == {"outer": {"inner": {"deep": "value"}}}


class TestBaseAgentIntegration:
    """Integration tests for BaseAgent methods."""

    def test_agent_initialization(self):
        """Test BaseAgent initialization."""
        agent = MockBaseAgent(detailed_logs=True)

        assert agent.llm is not None
        assert len(agent.tools) == 2
        assert agent.tools[0].name == "Tool A"
        assert agent.tools[1].name == "Tool B"
        assert agent.callbacks == []
        assert agent.detailed_logs is True

    async def test_error_handling_integration(self, base_agent):
        """Test that errors are properly handled through the automatic retry system."""
        messages = [HumanMessage(content="test")]
        base_agent.llm.ainvoke = AsyncMock(side_effect=Exception("Error"))

        # Errors should be propagated since retry is handled by the underlying client
        with pytest.raises(Exception, match="Error"):
            await base_agent._call_llm(messages)
