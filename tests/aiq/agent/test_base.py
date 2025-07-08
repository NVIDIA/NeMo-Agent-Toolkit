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


class TestStreamLLMWithRetry:
    """Test the _stream_llm_with_retry method."""

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

        result = await base_agent._stream_llm_with_retry(mock_runnable, inputs, config)

        assert result == "Hello world!"

    async def test_streaming_with_retry_success(self, base_agent):
        """Test streaming that fails once but succeeds on retry."""
        mock_runnable = Mock()
        mock_event = Mock()
        mock_event.content = "Success!"

        call_count = 0

        async def mock_astream(inputs, config=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Network error")
            else:
                yield mock_event

        mock_runnable.astream = mock_astream

        inputs = {"messages": [HumanMessage(content="test")]}

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await base_agent._stream_llm_with_retry(mock_runnable, inputs, max_retries=2)

        assert result == "Success!"
        assert call_count == 2
        mock_sleep.assert_called_once_with(2)  # 2^1 for first retry

    async def test_streaming_max_retries_exceeded(self, base_agent):
        """Test streaming that fails after max retries."""
        mock_runnable = Mock()

        async def failing_astream(inputs, config=None):
            raise Exception("Persistent error")

        mock_runnable.astream = failing_astream

        inputs = {"messages": [HumanMessage(content="test")]}

        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await base_agent._stream_llm_with_retry(mock_runnable, inputs, max_retries=2)

        assert "LLM streaming failed after 2 attempts" in result
        # The actual error message from the async function issue, not the original exception
        assert "async for" in result or "Persistent error" in result

    async def test_streaming_with_empty_response(self, base_agent):
        """Test streaming with empty response."""
        mock_runnable = Mock()

        async def mock_astream(inputs, config=None):
            return
            yield  # This will never execute, creating empty async generator

        mock_runnable.astream = mock_astream

        inputs = {"messages": [HumanMessage(content="test")]}

        result = await base_agent._stream_llm_with_retry(mock_runnable, inputs)

        assert result == ""


class TestCallLLMWithRetry:
    """Test the _call_llm_with_retry method."""

    async def test_successful_llm_call(self, base_agent):
        """Test successful LLM call without retries."""
        messages = [HumanMessage(content="test")]
        mock_response = AIMessage(content="Response content")

        base_agent.llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await base_agent._call_llm_with_retry(messages)

        assert result == "Response content"
        base_agent.llm.ainvoke.assert_called_once_with(messages)

    async def test_llm_call_with_retry_success(self, base_agent):
        """Test LLM call that fails once but succeeds on retry."""
        messages = [HumanMessage(content="test")]
        mock_response = AIMessage(content="Success!")

        base_agent.llm.ainvoke = AsyncMock(side_effect=[Exception("API error"), mock_response])

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await base_agent._call_llm_with_retry(messages, max_retries=2)

        assert result == "Success!"
        assert base_agent.llm.ainvoke.call_count == 2
        mock_sleep.assert_called_once_with(2)  # 2^1 for first retry

    async def test_llm_call_max_retries_exceeded(self, base_agent):
        """Test LLM call that fails after max retries."""
        messages = [HumanMessage(content="test")]
        base_agent.llm.ainvoke = AsyncMock(side_effect=Exception("Persistent error"))

        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await base_agent._call_llm_with_retry(messages, max_retries=2)

        assert "LLM call failed after 2 attempts" in result
        assert "Persistent error" in result
        assert base_agent.llm.ainvoke.call_count == 2

    async def test_llm_call_with_empty_content(self, base_agent):
        """Test LLM call that returns empty content."""
        messages = [HumanMessage(content="test")]
        mock_response = AIMessage(content="")

        base_agent.llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await base_agent._call_llm_with_retry(messages)

        assert result == ""

    async def test_llm_call_with_none_content(self, base_agent):
        """Test LLM call that returns None content."""
        messages = [HumanMessage(content="test")]
        mock_response = Mock()
        mock_response.content = None

        base_agent.llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await base_agent._call_llm_with_retry(messages)

        assert result == "None"


class TestCallToolWithRetry:
    """Test the _call_tool_with_retry method."""

    async def test_successful_tool_call(self, base_agent):
        """Test successful tool call without retries."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}
        config = RunnableConfig(callbacks=[])

        tool.ainvoke = AsyncMock(return_value="Tool response")

        result = await base_agent._call_tool_with_retry(tool, tool_input, config)

        assert result == "Tool response"
        tool.ainvoke.assert_called_once_with(tool_input, config=config)

    async def test_tool_call_with_retry_success(self, base_agent):
        """Test tool call that fails once but succeeds on retry."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}

        tool.ainvoke = AsyncMock(side_effect=[Exception("Tool error"), "Success!"])

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await base_agent._call_tool_with_retry(tool, tool_input, max_retries=2)

        assert result == "Success!"
        assert tool.ainvoke.call_count == 2
        mock_sleep.assert_called_once_with(2)  # 2^1 for first retry

    async def test_tool_call_max_retries_exceeded(self, base_agent):
        """Test tool call that fails after max retries."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}

        tool.ainvoke = AsyncMock(side_effect=Exception("Persistent error"))

        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await base_agent._call_tool_with_retry(tool, tool_input, max_retries=2)

        assert f"Tool {tool.name} failed after 2 attempts" in result
        assert "Persistent error" in result
        assert tool.ainvoke.call_count == 2

    async def test_tool_call_with_empty_string_response(self, base_agent):
        """Test tool call that returns empty string."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}

        tool.ainvoke = AsyncMock(return_value="")

        result = await base_agent._call_tool_with_retry(tool, tool_input)

        assert f"The tool {tool.name} provided an empty response." in result

    async def test_tool_call_with_none_response(self, base_agent):
        """Test tool call that returns None."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}

        tool.ainvoke = AsyncMock(return_value=None)

        result = await base_agent._call_tool_with_retry(tool, tool_input)

        # According to the implementation, None responses are returned as-is (not strings)
        assert result is None

    async def test_tool_call_with_non_string_response(self, base_agent):
        """Test tool call that returns non-string response."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}

        tool.ainvoke = AsyncMock(return_value={"result": "data"})

        result = await base_agent._call_tool_with_retry(tool, tool_input)

        assert result == {"result": "data"}

    async def test_tool_call_without_config(self, base_agent):
        """Test tool call without providing config parameter."""
        tool = base_agent.tools[0]  # Tool A
        tool_input = {"query": "test"}

        tool.ainvoke = AsyncMock(return_value="Tool response")

        result = await base_agent._call_tool_with_retry(tool, tool_input)

        assert result == "Tool response"
        tool.ainvoke.assert_called_once_with(tool_input, config=None)


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

    async def test_retry_methods_with_different_retry_counts(self, base_agent):
        """Test retry methods with different retry counts."""
        messages = [HumanMessage(content="test")]
        base_agent.llm.ainvoke = AsyncMock(side_effect=Exception("Error"))

        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await base_agent._call_llm_with_retry(messages, max_retries=1)

        assert "LLM call failed after 1 attempts" in result
        assert base_agent.llm.ainvoke.call_count == 1

    async def test_exponential_backoff_timing(self, base_agent):
        """Test that exponential backoff works correctly."""
        messages = [HumanMessage(content="test")]
        base_agent.llm.ainvoke = AsyncMock(side_effect=Exception("Error"))

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await base_agent._call_llm_with_retry(messages, max_retries=2)

        # With max_retries=2, there will be only 1 retry attempt (attempt 1)
        # So sleep is called once with 2^1 = 2
        mock_sleep.assert_called_once_with(2)

    async def test_zero_retries(self, base_agent):
        """Test behavior with zero retries."""
        messages = [HumanMessage(content="test")]
        base_agent.llm.ainvoke = AsyncMock(side_effect=Exception("Error"))

        result = await base_agent._call_llm_with_retry(messages, max_retries=0)

        # With max_retries=0, the loop doesn't execute, so it returns the fallback message
        assert "LLM call failed after all retry attempts" in result
