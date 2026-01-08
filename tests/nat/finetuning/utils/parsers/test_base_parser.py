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

from unittest.mock import MagicMock

import pytest

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStepState
from nat.data_models.intermediate_step import IntermediateStepType
from nat.finetuning.utils.parsers.base_parser import _validate_message_sequence
from nat.finetuning.utils.parsers.base_parser import parse_to_openai_messages


def create_mock_step(event_type, event_state, framework=None, data=None, name=None):
    """Helper function to create mock IntermediateStep objects."""
    step = MagicMock()
    step.event_type = event_type
    step.event_state = event_state
    step.framework = framework
    step.name = name
    step.data = data
    return step


class TestParseToOpenAIMessages:
    """Tests for parse_to_openai_messages function."""

    def test_empty_steps(self):
        """Test parsing empty list of steps."""
        result = parse_to_openai_messages([])
        assert result == []

    def test_skip_non_relevant_event_types(self):
        """Test that non-LLM/TOOL events are skipped."""
        step = create_mock_step(IntermediateStepType.WORKFLOW_START,
                                IntermediateStepState.START,
                                framework=LLMFrameworkEnum.LANGCHAIN)

        result = parse_to_openai_messages([step])
        assert len(result) == 0

    def test_skip_streaming_chunks(self):
        """Test that streaming chunks are skipped."""
        step = create_mock_step(
            IntermediateStepType.LLM_END,
            IntermediateStepState.CHUNK,  # Should be skipped
            framework=LLMFrameworkEnum.LANGCHAIN)

        result = parse_to_openai_messages([step])
        assert len(result) == 0

    def test_skip_llm_start_after_tool_end(self):
        """Test that LLM_START after TOOL_END is skipped."""
        steps = [
            create_mock_step(IntermediateStepType.TOOL_END,
                             IntermediateStepState.END,
                             framework=LLMFrameworkEnum.LANGCHAIN),
            create_mock_step(
                IntermediateStepType.LLM_START,  # Should be skipped
                IntermediateStepState.START,
                framework=LLMFrameworkEnum.LANGCHAIN),
        ]

        # Mock the data for tool_end
        steps[0].data = MagicMock()
        steps[0].data.output = "tool result"

        result = parse_to_openai_messages(steps)
        # Should only have tool message, no LLM_START
        assert len(result) == 1

    def test_unsupported_framework_is_skipped(self):
        """Test that unsupported framework is skipped."""
        step = create_mock_step(IntermediateStepType.LLM_END,
                                IntermediateStepState.END,
                                framework="unsupported_framework")

        result = parse_to_openai_messages([step])
        assert len(result) == 0

    def test_none_framework_is_skipped(self):
        """Test that None framework is skipped."""
        step = create_mock_step(IntermediateStepType.LLM_END, IntermediateStepState.END, framework=None)

        result = parse_to_openai_messages([step])
        assert len(result) == 0


class TestValidateMessageSequence:
    """Tests for _validate_message_sequence function."""

    def test_empty_messages(self):
        """Test validation of empty message list."""
        result = _validate_message_sequence([])
        assert result == []

    def test_valid_user_assistant_alternation(self):
        """Test valid user-assistant alternation."""
        messages = [{
            "role": "user", "content": "Hello"
        }, {
            "role": "assistant", "content": "Hi there!"
        }, {
            "role": "user", "content": "How are you?"
        }, {
            "role": "assistant", "content": "I'm doing well!"
        }]

        result = _validate_message_sequence(messages)
        assert result == messages

    def test_system_messages_at_beginning(self):
        """Test that system messages at beginning are valid."""
        messages = [{
            "role": "system", "content": "You are a helpful assistant"
        }, {
            "role": "user", "content": "Hello"
        }, {
            "role": "assistant", "content": "Hi!"
        }]

        result = _validate_message_sequence(messages)
        assert result == messages

    def test_system_message_after_non_system_raises_error(self):
        """Test that system message after non-system raises error."""
        messages = [{"role": "user", "content": "Hello"}, {"role": "system", "content": "Invalid system message"}]

        with pytest.raises(ValueError, match="System message found at position"):
            _validate_message_sequence(messages)

    def test_consecutive_user_messages_raises_error(self):
        """Test that consecutive user messages raise error."""
        messages = [{"role": "user", "content": "First message"}, {"role": "user", "content": "Second message"}]

        with pytest.raises(ValueError, match="Consecutive user messages"):
            _validate_message_sequence(messages)

    def test_consecutive_assistant_messages_raises_error(self):
        """Test that consecutive assistant messages raise error."""
        messages = [{
            "role": "user", "content": "Hello"
        }, {
            "role": "assistant", "content": "First response"
        }, {
            "role": "assistant", "content": "Second response"
        }]

        with pytest.raises(ValueError, match="Consecutive assistant messages"):
            _validate_message_sequence(messages)

    def test_non_user_messages_at_start_are_concatenated(self):
        """Test that non-user messages at start are concatenated into user message."""
        messages = [{
            "role": "tool", "content": "Tool result"
        }, {
            "role": "function", "content": "Function result"
        }, {
            "role": "assistant", "content": "Response"
        }]

        result = _validate_message_sequence(messages)
        # Should concatenate first two messages into a single user message
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert "[TOOL]" in result[0]["content"]
        assert "[FUNCTION]" in result[0]["content"]
        assert result[1]["role"] == "assistant"

    def test_user_and_non_user_messages_at_start_are_concatenated(self):
        """Test that user and non-user messages at start are concatenated."""
        messages = [{
            "role": "user", "content": "User message"
        }, {
            "role": "tool", "content": "Tool result"
        }, {
            "role": "assistant", "content": "Response"
        }]

        result = _validate_message_sequence(messages)
        # Should concatenate first two messages into a single user message
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert "User message" in result[0]["content"]
        assert "[TOOL]" in result[0]["content"]
        assert result[1]["role"] == "assistant"

    def test_valid_with_tool_messages(self):
        """Test valid sequence with tool messages."""
        messages = [{
            "role": "user", "content": "What's the weather?"
        }, {
            "role": "assistant", "content": "Let me check", "tool_calls": [{
                "id": "1"
            }]
        }, {
            "role": "tool", "content": "Sunny, 75°F", "tool_call_id": "1"
        }, {
            "role": "assistant", "content": "The weather is sunny!"
        }]

        result = _validate_message_sequence(messages)
        assert result == messages

    def test_system_then_non_user_at_start_are_concatenated(self):
        """Test that system message followed by non-user messages are handled."""
        messages = [{
            "role": "system", "content": "You are helpful"
        }, {
            "role": "tool", "content": "Tool result"
        }, {
            "role": "assistant", "content": "Response"
        }]

        result = _validate_message_sequence(messages)
        # System should remain, tool should be converted to user, assistant remains
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert "[TOOL]" in result[1]["content"]
        assert result[2]["role"] == "assistant"

    def test_empty_content_in_non_user_messages(self):
        """Test handling of empty content in non-user messages at start."""
        messages = [{
            "role": "tool", "content": ""
        }, {
            "role": "function", "content": "Function result"
        }, {
            "role": "assistant", "content": "Response"
        }]

        result = _validate_message_sequence(messages)
        # Should concatenate, but empty content tool shouldn't add much
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
