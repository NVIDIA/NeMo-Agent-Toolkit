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

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import FunctionMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.invocation_node import InvocationNode
from nat.plugins.langchain.langchain_parser import _extract_content
from nat.plugins.langchain.langchain_parser import _parse_dict_message
from nat.plugins.langchain.langchain_parser import _parse_langchain_message
from nat.plugins.langchain.langchain_parser import _parse_tool_message


class TestParseLangChainMessage:
    """Tests for _parse_langchain_message function."""

    def test_parse_human_message(self):
        """Test parsing HumanMessage."""
        msg = HumanMessage(content="Hello")
        result = _parse_langchain_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_parse_ai_message(self):
        """Test parsing AIMessage."""
        msg = AIMessage(content="Hi there!")
        result = _parse_langchain_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Hi there!"

    def test_parse_system_message(self):
        """Test parsing SystemMessage."""
        msg = SystemMessage(content="System prompt")
        result = _parse_langchain_message(msg)
        assert result["role"] == "system"
        assert result["content"] == "System prompt"

    def test_parse_tool_message(self):
        """Test parsing ToolMessage."""
        msg = ToolMessage(content="Tool result", tool_call_id="call_123")
        result = _parse_langchain_message(msg)
        assert result["role"] == "tool"
        assert result["content"] == "Tool result"
        assert result["tool_call_id"] == "call_123"

    def test_parse_function_message(self):
        """Test parsing FunctionMessage."""
        msg = FunctionMessage(content="Function result", name="my_function")
        result = _parse_langchain_message(msg)
        assert result["role"] == "function"
        assert result["content"] == "Function result"
        assert result["name"] == "my_function"

    def test_parse_ai_message_with_tool_calls(self):
        """Test parsing AIMessage with tool calls."""
        tool_calls = [{"id": "1", "function": {"name": "test"}}]
        msg = AIMessage(content="", additional_kwargs={"tool_calls": tool_calls})
        result = _parse_langchain_message(msg)
        assert result["role"] == "assistant"
        assert result["tool_calls"] == tool_calls

    def test_parse_ai_message_with_function_call(self):
        """Test parsing AIMessage with function call."""
        func_call = {"name": "test_func", "arguments": "{}"}
        msg = AIMessage(content="", additional_kwargs={"function_call": func_call})
        result = _parse_langchain_message(msg)
        assert result["role"] == "assistant"
        assert result["function_call"] == func_call

    def test_parse_empty_content(self):
        """Test parsing message with empty content."""
        msg = HumanMessage(content="")
        result = _parse_langchain_message(msg)
        assert result["role"] == "user"
        assert result["content"] == ""


class TestParseDictMessage:
    """Tests for _parse_dict_message function."""

    def test_parse_basic_dict(self):
        """Test parsing basic dictionary message."""
        msg_dict = {"role": "user", "content": "Test message"}
        result = _parse_dict_message(msg_dict)
        assert result["role"] == "user"
        assert result["content"] == "Test message"

    def test_parse_with_text_field(self):
        """Test parsing dict with 'text' instead of 'content'."""
        msg_dict = {"role": "assistant", "text": "Response"}
        result = _parse_dict_message(msg_dict)
        assert result["role"] == "assistant"
        assert result["content"] == "Response"

    def test_parse_default_role(self):
        """Test that default role is 'user'."""
        msg_dict = {"content": "No role specified"}
        result = _parse_dict_message(msg_dict)
        assert result["role"] == "user"

    def test_parse_with_optional_fields(self):
        """Test parsing with optional fields."""
        msg_dict = {
            "role": "assistant",
            "content": "Test",
            "tool_calls": [{
                "id": "1"
            }],
            "logprobs": {
                "tokens": []
            },
            "function_call": {
                "name": "test"
            }
        }
        result = _parse_dict_message(msg_dict)
        assert result["tool_calls"] == [{"id": "1"}]
        assert "logprobs" in result
        assert "function_call" in result

    def test_parse_empty_content(self):
        """Test parsing dict with no content field."""
        msg_dict = {"role": "user"}
        result = _parse_dict_message(msg_dict)
        assert result["content"] == ""


class TestExtractContent:
    """Tests for _extract_content function."""

    def test_extract_string(self):
        """Test extracting content from string."""
        assert _extract_content("Simple string") == "Simple string"

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

    def test_extract_from_dict_fallback_to_json(self):
        """Test fallback to JSON for dict without known keys."""
        data = {"unknown_key": "value"}
        result = _extract_content(data)
        assert "unknown_key" in result
        assert "value" in result

    def test_extract_from_string_list(self):
        """Test extracting from list of strings."""
        data = ["First line", "Second line", "Third line"]
        result = _extract_content(data)
        assert result == "First line\nSecond line\nThird line"

    def test_extract_from_mixed_list(self):
        """Test extracting from list with non-strings."""
        data = ["String", 123, {"key": "value"}]
        result = _extract_content(data)
        # Should convert to JSON
        assert isinstance(result, str)

    def test_extract_from_object_with_content_attr(self):
        """Test extracting from object with content attribute."""
        mock_obj = MagicMock()
        mock_obj.content = "Object content"
        assert _extract_content(mock_obj) == "Object content"

    def test_extract_fallback_to_str(self):
        """Test fallback to str() for unknown types."""
        data = 12345
        assert _extract_content(data) == "12345"

    def test_extract_empty_string(self):
        """Test extracting empty string."""
        assert _extract_content("") == ""

    def test_extract_none(self):
        """Test extracting None."""
        assert _extract_content(None) == "None"


class TestParseToolMessageCallId:
    """Tests for the tool_call_id carried on a TOOL_END tool message."""

    @staticmethod
    def _tool_end_step(metadata: Any, output: str = "tool output") -> IntermediateStep:
        """Build a real TOOL_END step, not a mock.

        A MagicMock metadata answers any attribute, so it cannot show whether a
        missing tool call id is handled.
        """
        return IntermediateStep(parent_id="root",
                                function_ancestry=InvocationNode(function_name="wf", function_id="wf-1"),
                                payload=IntermediateStepPayload(
                                    event_type=IntermediateStepType.TOOL_END,
                                    framework=LLMFrameworkEnum.LANGCHAIN,
                                    name="get_weather",
                                    metadata=metadata,
                                    data=StreamEventData(output=output),
                                ))

    @pytest.mark.parametrize("metadata", [TraceMetadata(), {}, None], ids=["model", "dict", "none"])
    def test_omits_tool_call_id_when_step_has_none(self, metadata: Any):
        """No recorded id means the key is omitted, not filled with a placeholder.

        TraceMetadata declares no tool_call_id, so this is the common case. A
        placeholder is shaped like a real id, which makes a trajectory look
        valid while every tool result points at the same call.
        """
        result = _parse_tool_message(self._tool_end_step(metadata))
        assert result["role"] == "tool"
        assert "tool_call_id" not in result

    def test_carries_a_recorded_id_from_mapping_metadata(self):
        """A real id on dict metadata is carried through unchanged."""
        result = _parse_tool_message(self._tool_end_step({"tool_call_id": "call_abc123"}))
        assert result["tool_call_id"] == "call_abc123"

    def test_carries_a_recorded_id_from_model_metadata(self):
        """TraceMetadata allows extra fields, so an id set there is carried through."""
        result = _parse_tool_message(self._tool_end_step(TraceMetadata(tool_call_id="call_xyz789")))
        assert result["tool_call_id"] == "call_xyz789"

    @pytest.mark.parametrize("value", [None, "", 0, 7, False], ids=["none", "empty", "zero", "int", "bool"])
    def test_rejects_an_unusable_recorded_id(self, value: Any):
        """Only a non-empty string is a usable id.

        OpenAI requires tool_call_id to be a string, so a null, blank or
        numeric value is dropped rather than emitted as an invalid id.
        """
        result = _parse_tool_message(self._tool_end_step({"tool_call_id": value}))
        assert "tool_call_id" not in result

    def test_distinct_tool_calls_do_not_share_an_id(self):
        """Two tool results must not both claim the same call."""
        first = _parse_tool_message(self._tool_end_step({"tool_call_id": "call_1"}, output="sunny"))
        second = _parse_tool_message(self._tool_end_step({"tool_call_id": "call_2"}, output="184.20"))
        assert first["tool_call_id"] != second["tool_call_id"]
