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

import json
from unittest.mock import MagicMock

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import InvocationNode
from nat.data_models.intermediate_step import StreamEventData
from nat.finetuning.utils.parsers.adk_parser import _extract_content
from nat.finetuning.utils.parsers.adk_parser import _parse_assistant_message
from nat.finetuning.utils.parsers.adk_parser import _parse_generic_message
from nat.finetuning.utils.parsers.adk_parser import _parse_input_message
from nat.finetuning.utils.parsers.adk_parser import _parse_tool_message
from nat.finetuning.utils.parsers.adk_parser import parse_to_openai_message


def create_intermediate_step(
    event_type: IntermediateStepType,
    payload_data: StreamEventData | None = None,
    name: str | None = None,
) -> IntermediateStep:
    """Helper to create IntermediateStep objects for testing."""
    invocation_node = InvocationNode(
        function_id="test_id",
        function_name="test_function",
        parent_id="root",
    )
    step_payload = IntermediateStepPayload(
        event_type=event_type,
        name=name,
        data=payload_data,
    )
    return IntermediateStep(
        parent_id="root",
        function_ancestry=invocation_node,
        payload=step_payload,
    )


class TestParseToOpenAIMessage:
    """Tests for parse_to_openai_message function."""

    def test_routes_llm_end_to_assistant_parser(self):
        """Test that LLM_END events are routed to assistant parser."""
        payload_message = MagicMock()
        payload_message.__getitem__ = lambda self, key: "Test content" if key == "content" else None
        payload_message.__contains__ = lambda self, key: key in ["content"]

        payload = MagicMock()
        payload.message = payload_message
        payload.__getitem__ = lambda self, key: payload_message if key == "message" else None
        payload.__contains__ = lambda self, key: key == "message"

        data = StreamEventData(payload=payload)
        step = create_intermediate_step(IntermediateStepType.LLM_END, payload_data=data)

        result = parse_to_openai_message(step)
        assert result["role"] == "assistant"

    def test_routes_tool_end_to_tool_parser(self):
        """Test that TOOL_END events are routed to tool parser."""
        data = StreamEventData(output="Tool output")
        step = create_intermediate_step(IntermediateStepType.TOOL_END, payload_data=data, name="my_tool")

        result = parse_to_openai_message(step)
        assert result["role"] == "function"
        assert result["content"] == "Tool output"
        assert result["name"] == "my_tool"

    def test_routes_llm_start_to_input_parser(self):
        """Test that LLM_START events are routed to input parser."""
        data = StreamEventData(payload=[{"role": "user", "content": "Hello"}])
        step = create_intermediate_step(IntermediateStepType.LLM_START, payload_data=data)

        result = parse_to_openai_message(step)
        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_routes_other_types_to_generic_parser(self):
        """Test that other event types are routed to generic parser."""
        data = StreamEventData(output="Some output")
        step = create_intermediate_step(IntermediateStepType.WORKFLOW_START, payload_data=data)

        result = parse_to_openai_message(step)
        assert result["role"] == "user"
        assert result["content"] == "Some output"


class TestParseInputMessage:
    """Tests for _parse_input_message function."""

    def test_parse_empty_payload(self):
        """Test parsing empty payload list."""
        data = StreamEventData(payload=[])
        step = create_intermediate_step(IntermediateStepType.LLM_START, payload_data=data)

        result = _parse_input_message(step)
        assert result == {"role": "user", "content": ""}

    def test_parse_single_dict_message_with_role_and_content(self):
        """Test parsing single dict message with role and content."""
        data = StreamEventData(payload=[{"role": "user", "content": "Hello world"}])
        step = create_intermediate_step(IntermediateStepType.LLM_START, payload_data=data)

        result = _parse_input_message(step)
        assert result == {"role": "user", "content": "Hello world"}

    def test_parse_single_dict_message_missing_role(self):
        """Test parsing dict missing role key."""
        data = StreamEventData(payload=[{"text": "Some text"}])
        step = create_intermediate_step(IntermediateStepType.LLM_START, payload_data=data)

        result = _parse_input_message(step)
        assert result["role"] == "user"
        assert "text" in result["content"]

    def test_parse_single_non_dict_message(self):
        """Test parsing non-dict message."""
        data = StreamEventData(payload=["Simple string message"])
        step = create_intermediate_step(IntermediateStepType.LLM_START, payload_data=data)

        result = _parse_input_message(step)
        assert result == {"role": "user", "content": "Simple string message"}

    def test_parse_multiple_messages(self):
        """Test parsing multiple messages returns a list."""
        data = StreamEventData(payload=[
            {
                "role": "system", "content": "System prompt"
            },
            {
                "role": "user", "content": "User message"
            },
        ])
        step = create_intermediate_step(IntermediateStepType.LLM_START, payload_data=data)

        result = _parse_input_message(step)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_parse_multiple_mixed_messages(self):
        """Test parsing multiple messages with mixed formats."""
        data = StreamEventData(payload=[
            {
                "role": "user", "content": "Valid message"
            },
            "String message",
            {
                "some_key": "no role or content"
            },
        ])
        step = create_intermediate_step(IntermediateStepType.LLM_START, payload_data=data)

        result = _parse_input_message(step)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "Valid message"}
        assert result[1] == {"role": "user", "content": "String message"}
        assert result[2]["role"] == "user"


class TestParseAssistantMessage:
    """Tests for _parse_assistant_message function."""

    def test_parse_assistant_with_content(self):
        """Test parsing assistant message with content."""
        payload_message = {"content": "Assistant response", "tool_calls": None}
        payload = MagicMock()
        payload.message = payload_message
        payload.__contains__ = lambda self, key: key in ["message"]

        data = StreamEventData(payload=payload)
        step = create_intermediate_step(IntermediateStepType.LLM_END, payload_data=data)

        result = _parse_assistant_message(step)
        assert result["role"] == "assistant"
        assert result["content"] == "Assistant response"

    def test_parse_assistant_with_tool_calls(self):
        """Test parsing assistant message with tool calls."""
        tool_calls = [{"id": "call_123", "function": {"name": "test_func", "arguments": "{}"}}]
        payload_message = {"content": "", "tool_calls": tool_calls}
        payload = MagicMock()
        payload.message = payload_message
        payload.__contains__ = lambda self, key: key in ["message"]

        data = StreamEventData(payload=payload)
        step = create_intermediate_step(IntermediateStepType.LLM_END, payload_data=data)

        result = _parse_assistant_message(step)
        assert result["role"] == "assistant"
        assert result["tool_calls"] == tool_calls

    def test_parse_assistant_with_logprobs(self):
        """Test parsing assistant message with logprobs."""
        logprobs_data = {"tokens": ["Hello"], "token_logprobs": [-0.5]}
        payload_message = {"content": "Hello", "tool_calls": None}
        payload = MagicMock()
        payload.message = payload_message
        payload.__contains__ = lambda self, key: key in ["message", "logprobs"]
        payload.__getitem__ = lambda self, key: logprobs_data if key == "logprobs" else payload_message

        data = StreamEventData(payload=payload)
        step = create_intermediate_step(IntermediateStepType.LLM_END, payload_data=data)

        result = _parse_assistant_message(step)
        assert result["role"] == "assistant"
        assert result["logprobs"] == logprobs_data

    def test_parse_assistant_with_none_content(self):
        """Test parsing assistant message with None content."""
        payload_message = {"content": None, "tool_calls": None}
        payload = MagicMock()
        payload.message = payload_message
        payload.__contains__ = lambda self, key: key in ["message"]

        data = StreamEventData(payload=payload)
        step = create_intermediate_step(IntermediateStepType.LLM_END, payload_data=data)

        result = _parse_assistant_message(step)
        assert result["role"] == "assistant"
        assert result["content"] == ""

    def test_parse_assistant_no_payload(self):
        """Test parsing assistant message with no payload."""
        step = create_intermediate_step(IntermediateStepType.LLM_END, payload_data=None)

        result = _parse_assistant_message(step)
        assert result == {"role": "assistant", "content": ""}

    def test_parse_assistant_empty_data(self):
        """Test parsing assistant message with empty data."""
        data = StreamEventData(payload=None)
        step = create_intermediate_step(IntermediateStepType.LLM_END, payload_data=data)

        result = _parse_assistant_message(step)
        assert result == {"role": "assistant", "content": ""}


class TestParseToolMessage:
    """Tests for _parse_tool_message function."""

    def test_parse_tool_with_output(self):
        """Test parsing tool message with output."""
        data = StreamEventData(output="Tool execution result")
        step = create_intermediate_step(IntermediateStepType.TOOL_END, payload_data=data, name="my_function")

        result = _parse_tool_message(step)
        assert result["role"] == "function"
        assert result["content"] == "Tool execution result"
        assert result["name"] == "my_function"

    def test_parse_tool_with_payload_fallback(self):
        """Test parsing tool message falls back to payload when no output."""
        data = StreamEventData(payload="Payload content")
        step = create_intermediate_step(IntermediateStepType.TOOL_END, payload_data=data, name="another_func")

        result = _parse_tool_message(step)
        assert result["role"] == "function"
        assert result["content"] == "Payload content"
        assert result["name"] == "another_func"

    def test_parse_tool_no_content(self):
        """Test parsing tool message with no content."""
        data = StreamEventData()
        step = create_intermediate_step(IntermediateStepType.TOOL_END, payload_data=data, name="empty_func")

        result = _parse_tool_message(step)
        assert result["role"] == "function"
        assert result["content"] == ""
        assert result["name"] == "empty_func"

    def test_parse_tool_no_name(self):
        """Test parsing tool message with no name."""
        data = StreamEventData(output="Result")
        step = create_intermediate_step(IntermediateStepType.TOOL_END, payload_data=data, name=None)

        result = _parse_tool_message(step)
        assert result["role"] == "function"
        assert result["content"] == "Result"
        assert "name" not in result

    def test_parse_tool_no_data(self):
        """Test parsing tool message with no data."""
        step = create_intermediate_step(IntermediateStepType.TOOL_END, payload_data=None, name="func")

        result = _parse_tool_message(step)
        assert result["role"] == "function"
        assert result["content"] == ""


class TestParseGenericMessage:
    """Tests for _parse_generic_message function."""

    def test_parse_generic_with_output(self):
        """Test parsing generic message with output."""
        data = StreamEventData(output="Output content")
        step = create_intermediate_step(IntermediateStepType.WORKFLOW_START, payload_data=data)

        result = _parse_generic_message(step)
        assert result["role"] == "user"
        assert result["content"] == "Output content"

    def test_parse_generic_with_input_fallback(self):
        """Test parsing generic message falls back to input."""
        data = StreamEventData(input="Input content")
        step = create_intermediate_step(IntermediateStepType.TASK_START, payload_data=data)

        result = _parse_generic_message(step)
        assert result["role"] == "user"
        assert result["content"] == "Input content"

    def test_parse_generic_with_chunk_fallback(self):
        """Test parsing generic message falls back to chunk."""
        data = StreamEventData(chunk="Chunk content")
        step = create_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, payload_data=data)

        result = _parse_generic_message(step)
        assert result["role"] == "user"
        assert result["content"] == "Chunk content"

    def test_parse_generic_no_content(self):
        """Test parsing generic message with no content."""
        data = StreamEventData()
        step = create_intermediate_step(IntermediateStepType.WORKFLOW_END, payload_data=data)

        result = _parse_generic_message(step)
        assert result["role"] == "user"
        assert result["content"] == ""

    def test_parse_generic_no_data(self):
        """Test parsing generic message with no data."""
        step = create_intermediate_step(IntermediateStepType.CUSTOM_START, payload_data=None)

        result = _parse_generic_message(step)
        assert result["role"] == "user"
        assert result["content"] == ""


class TestExtractContent:
    """Tests for _extract_content function."""

    def test_extract_string(self):
        """Test extracting content from string."""
        assert _extract_content("Simple string") == "Simple string"

    def test_extract_empty_string(self):
        """Test extracting empty string."""
        assert _extract_content("") == ""

    def test_extract_from_dict_with_content(self):
        """Test extracting content from dict with 'content' key."""
        data = {"content": "Message content"}
        assert _extract_content(data) == "Message content"

    def test_extract_from_dict_with_text(self):
        """Test extracting content from dict with 'text' key."""
        data = {"text": "Text content"}
        assert _extract_content(data) == "Text content"

    def test_extract_from_dict_with_message(self):
        """Test extracting content from dict with 'message' key."""
        data = {"message": "Message value"}
        assert _extract_content(data) == "Message value"

    def test_extract_from_dict_with_output(self):
        """Test extracting content from dict with 'output' key."""
        data = {"output": "Output value"}
        assert _extract_content(data) == "Output value"

    def test_extract_from_dict_fallback_to_json(self):
        """Test fallback to JSON for dict without known keys."""
        data = {"unknown_key": "value", "another": 123}
        result = _extract_content(data)
        assert "unknown_key" in result
        assert "value" in result

    def test_extract_from_dict_with_blocks(self):
        """Test extracting content from dict with blocks format."""
        data = {"blocks": [{"text": "First "}, {"text": "Second"}]}
        result = _extract_content(data)
        assert result == "First Second"

    def test_extract_from_dict_with_mixed_blocks(self):
        """Test extracting content from dict with mixed blocks."""
        data = {"blocks": [{"text": "Text"}, "plain string"]}
        result = _extract_content(data)
        assert "Text" in result
        assert "plain string" in result

    def test_extract_from_string_list(self):
        """Test extracting from list of strings."""
        data = ["First line", "Second line", "Third line"]
        result = _extract_content(data)
        assert result == "First line\nSecond line\nThird line"

    def test_extract_from_mixed_list(self):
        """Test extracting from list with non-strings falls back to JSON."""
        data = ["String", 123, {"key": "value"}]
        result = _extract_content(data)
        # Should convert to JSON
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == data

    def test_extract_from_object_with_content_attr(self):
        """Test extracting from object with content attribute."""
        mock_obj = MagicMock()
        mock_obj.content = "Object content"
        del mock_obj.text  # Remove text attr so content is used
        assert _extract_content(mock_obj) == "Object content"

    def test_extract_from_object_with_text_attr(self):
        """Test extracting from object with text attribute."""
        mock_obj = MagicMock(spec=["text"])
        mock_obj.text = "Object text"
        assert _extract_content(mock_obj) == "Object text"

    def test_extract_fallback_to_str(self):
        """Test fallback to str() for unknown types."""
        assert _extract_content(12345) == "12345"
        assert _extract_content(3.14) == "3.14"

    def test_extract_none(self):
        """Test extracting None."""
        assert _extract_content(None) == "None"

    def test_extract_boolean(self):
        """Test extracting boolean values."""
        assert _extract_content(True) == "True"
        assert _extract_content(False) == "False"

    def test_extract_nested_dict_content(self):
        """Test extracting from nested dict prefers top-level content key."""
        data = {"content": "Top level", "nested": {"content": "Nested"}}
        assert _extract_content(data) == "Top level"
