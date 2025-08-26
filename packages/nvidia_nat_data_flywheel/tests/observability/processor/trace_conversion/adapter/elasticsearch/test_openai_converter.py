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

# yapf: disable
from unittest.mock import patch

from nat.data_models.intermediate_step import ToolSchema
from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter import (
    FINISH_REASON_MAP,
)
from nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter import (
    ROLE_MAP,
)
from nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter import (
    convert_chat_response,
)
from nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter import (
    convert_langchain_openai,
)
from nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter import (
    convert_message_to_dfw,
)
from nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter import (
    convert_role,
)
from nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter import (
    create_message_by_role,
)
from nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter import (
    create_tool_calls,
)
from nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter import (
    validate_and_convert_tools,
)
from nat.plugins.data_flywheel.observability.schema.provider.openai_message import OpenAIMessage
from nat.plugins.data_flywheel.observability.schema.provider.openai_trace_source import OpenAITraceSource
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import AssistantMessage
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import DFWESRecord
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import FinishReason
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import Function
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import FunctionMessage
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import RequestTool
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import ResponseChoice
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import SystemMessage
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import ToolCall
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import ToolMessage
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import UserMessage
from nat.plugins.data_flywheel.observability.schema.trace_container import TraceContainer


class TestConvertRole:
    """Test suite for convert_role function."""

    def test_convert_role_basic_mappings(self):
        """Test basic role conversions."""
        assert convert_role("human") == "user"
        assert convert_role("user") == "user"
        assert convert_role("assistant") == "assistant"
        assert convert_role("ai") == "assistant"
        assert convert_role("system") == "system"
        assert convert_role("tool") == "tool"
        assert convert_role("function") == "function"
        assert convert_role("chain") == "function"

    def test_convert_role_fallback_to_default(self):
        """Test that unknown roles fall back to default user role."""
        assert convert_role("unknown_role") == "user"
        assert convert_role("") == "user"
        assert convert_role("custom") == "user"

    def test_convert_role_case_sensitivity(self):
        """Test that role conversion is case-sensitive."""
        assert convert_role("User") == "user"  # Should fallback to default
        assert convert_role("ASSISTANT") == "user"  # Should fallback to default
        assert convert_role("user") == "user"  # Should work correctly


class TestCreateMessageByRole:
    """Test suite for create_message_by_role function."""

    def test_create_user_message(self):
        """Test creating user messages."""
        result = create_message_by_role("user", "Hello, world!")
        assert isinstance(result, UserMessage)
        assert result.content == "Hello, world!"
        assert result.role == "user"

    def test_create_user_message_with_human_role(self):
        """Test creating user messages with 'human' role."""
        result = create_message_by_role("human", "Hello from human!")
        assert isinstance(result, UserMessage)
        assert result.content == "Hello from human!"
        assert result.role == "user"

    def test_create_system_message(self):
        """Test creating system messages."""
        result = create_message_by_role("system", "You are a helpful assistant.")
        assert isinstance(result, SystemMessage)
        assert result.content == "You are a helpful assistant."
        assert result.role == "system"

    def test_create_assistant_message_without_tool_calls(self):
        """Test creating assistant messages without tool calls."""
        result = create_message_by_role("assistant", "I can help you with that.")
        assert isinstance(result, AssistantMessage)
        assert result.content == "I can help you with that."
        assert result.role == "assistant"
        assert result.tool_calls is None

    def test_create_assistant_message_with_tool_calls(self):
        """Test creating assistant messages with tool calls."""
        tool_calls = [ToolCall(type="function", function=Function(name="test_func", arguments={}))]
        result = create_message_by_role("assistant", "Let me call a function.", tool_calls=tool_calls)
        assert isinstance(result, AssistantMessage)
        assert result.content is None  # Content should be None when tool_calls exist
        assert result.role == "assistant"
        assert result.tool_calls == tool_calls

    def test_create_assistant_message_with_ai_role(self):
        """Test creating assistant messages with 'ai' role."""
        result = create_message_by_role("ai", "AI response here.")
        assert isinstance(result, AssistantMessage)
        assert result.content == "AI response here."
        assert result.role == "assistant"

    def test_create_tool_message(self):
        """Test creating tool messages."""
        result = create_message_by_role("tool", "Tool execution result", tool_call_id="call_123")
        assert isinstance(result, ToolMessage)
        assert result.content == "Tool execution result"
        assert result.role == "tool"
        assert result.tool_call_id == "call_123"

    def test_create_tool_message_without_tool_call_id(self):
        """Test creating tool messages without tool_call_id."""
        result = create_message_by_role("tool", "Tool result")
        assert isinstance(result, ToolMessage)
        assert result.content == "Tool result"
        assert result.role == "tool"
        assert result.tool_call_id == ""

    def test_create_function_message(self):
        """Test creating function messages."""
        result = create_message_by_role("function", "Function result")
        assert isinstance(result, FunctionMessage)
        assert result.content == "Function result"
        assert result.role == "function"

    def test_create_function_message_with_chain_role(self):
        """Test creating function messages with 'chain' role."""
        result = create_message_by_role("chain", "Chain execution result")
        assert isinstance(result, FunctionMessage)
        assert result.content == "Chain execution result"
        assert result.role == "function"

    def test_create_message_with_unsupported_role(self):
        """Test that unsupported roles are converted to default user role."""
        # Unsupported roles fall back to default "user" role due to convert_role
        result = create_message_by_role("invalid_role", "content")
        assert isinstance(result, UserMessage)
        assert result.content == "content"
        assert result.role == "user"

    def test_create_message_with_none_content_for_required_roles(self):
        """Test that None content raises ValueError for roles that require content."""
        # User message requires content
        try:
            create_message_by_role("user", None)
            assert False, "Expected ValueError for None content in user message"
        except ValueError as e:
            assert "User message content cannot be None" in str(e)

        # System message requires content
        try:
            create_message_by_role("system", None)
            assert False, "Expected ValueError for None content in system message"
        except ValueError as e:
            assert "System message content cannot be None" in str(e)

        # Tool message requires content
        try:
            create_message_by_role("tool", None)
            assert False, "Expected ValueError for None content in tool message"
        except ValueError as e:
            assert "Tool message content cannot be None" in str(e)


class TestCreateToolCalls:
    """Test suite for create_tool_calls function."""

    def test_create_tool_calls_with_valid_data(self):
        """Test creating tool calls with valid data."""
        tool_calls_data = [{"function": {"name": "get_weather", "arguments": '{"location": "New York"}'}}]

        result = create_tool_calls(tool_calls_data)

        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].type_ == "function"
        assert result[0].function.name == "get_weather"
        assert result[0].function.arguments == {"location": "New York"}

    def test_create_tool_calls_with_dict_arguments(self):
        """Test creating tool calls when arguments are already a dict."""
        tool_calls_data = [{"function": {"name": "calculate", "arguments": {"x": 10, "y": 20}}}]

        result = create_tool_calls(tool_calls_data)

        assert len(result) == 1
        assert result[0].function.name == "calculate"
        assert result[0].function.arguments == {"x": 10, "y": 20}

    def test_create_tool_calls_with_invalid_json_arguments(self):
        """Test creating tool calls with invalid JSON arguments."""
        tool_calls_data = [{"function": {"name": "broken_func", "arguments": "invalid json {"}}]

        with patch(
                'nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter.logger'
        ) as mock_logger:
            result = create_tool_calls(tool_calls_data)

            assert len(result) == 1
            assert result[0].function.name == "broken_func"
            assert result[0].function.arguments == {}  # Should fallback to empty dict
            mock_logger.warning.assert_called_once()

    def test_create_tool_calls_with_missing_function_name(self):
        """Test creating tool calls with missing function name."""
        tool_calls_data = [{"function": {"arguments": '{"param": "value"}'}}]

        result = create_tool_calls(tool_calls_data)

        assert len(result) == 1
        assert result[0].function.name == "unknown"  # Should fallback to "unknown"
        assert result[0].function.arguments == {"param": "value"}

    def test_create_tool_calls_with_empty_function_name(self):
        """Test creating tool calls with empty function name."""
        tool_calls_data = [{"function": {"name": "", "arguments": "{}"}}]

        result = create_tool_calls(tool_calls_data)

        assert len(result) == 1
        assert result[0].function.name == "unknown"  # Should fallback to "unknown"

    def test_create_tool_calls_with_none_function_name(self):
        """Test creating tool calls with None function name."""
        tool_calls_data = [{"function": {"name": None, "arguments": "{}"}}]

        result = create_tool_calls(tool_calls_data)

        assert len(result) == 1
        assert result[0].function.name == "unknown"  # Should fallback to "unknown"

    def test_create_tool_calls_with_invalid_tool_call_structure(self):
        """Test creating tool calls with invalid structure."""
        # Non-dict tool call should be skipped
        tool_calls_data = ["invalid", {"function": {"name": "valid_func", "arguments": "{}"}}]

        result = create_tool_calls(tool_calls_data)

        assert len(result) == 1
        assert result[0].function.name == "valid_func"

    def test_create_tool_calls_with_invalid_function_structure(self):
        """Test creating tool calls with invalid function structure."""
        # Non-dict function should be skipped
        tool_calls_data = [{"function": "not_a_dict"}, {"function": {"name": "valid_func", "arguments": "{}"}}]

        result = create_tool_calls(tool_calls_data)

        assert len(result) == 1
        assert result[0].function.name == "valid_func"

    def test_create_tool_calls_with_empty_list(self):
        """Test creating tool calls with empty list."""
        result = create_tool_calls([])
        assert len(result) == 0

    def test_create_tool_calls_with_multiple_tool_calls(self):
        """Test creating multiple tool calls."""
        tool_calls_data = [{
            "function": {
                "name": "func1", "arguments": '{"a": 1}'
            }
        }, {
            "function": {
                "name": "func2", "arguments": '{"b": 2}'
            }
        }, {
            "function": {
                "name": "func3", "arguments": '{"c": 3}'
            }
        }]

        result = create_tool_calls(tool_calls_data)

        assert len(result) == 3
        assert result[0].function.name == "func1"
        assert result[1].function.name == "func2"
        assert result[2].function.name == "func3"
        assert result[0].function.arguments == {"a": 1}
        assert result[1].function.arguments == {"b": 2}
        assert result[2].function.arguments == {"c": 3}


class TestConvertMessageToDfw:
    """Test suite for convert_message_to_dfw function."""

    def test_convert_user_message(self):
        """Test converting user message."""
        message = OpenAIMessage(content="Hello, assistant!", type="user", response_metadata={}, additional_kwargs={})

        result = convert_message_to_dfw(message)

        assert isinstance(result, UserMessage)
        assert result.content == "Hello, assistant!"
        assert result.role == "user"

    def test_convert_message_with_content_in_response_metadata(self):
        """Test converting message with content in response_metadata."""
        message = OpenAIMessage(content="original_content",
                                type="user",
                                response_metadata={"content": "metadata_content"},
                                additional_kwargs={})

        result = convert_message_to_dfw(message)

        assert isinstance(result, UserMessage)
        assert result.content == "metadata_content"  # Should use response_metadata content

    def test_convert_assistant_message_with_tool_calls(self):
        """Test converting assistant message with tool calls."""
        message = OpenAIMessage(
            content="Let me help you",
            type="assistant",
            response_metadata={},
            additional_kwargs={"tool_calls": [{
                "function": {
                    "name": "search", "arguments": '{"query": "test"}'
                }
            }]})

        result = convert_message_to_dfw(message)

        assert isinstance(result, AssistantMessage)
        assert result.content is None  # Content should be None when tool calls exist
        assert result.role == "assistant"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"

    def test_convert_tool_message_with_tool_call_id(self):
        """Test converting tool message with tool_call_id."""
        message = OpenAIMessage(content="Tool execution result",
                                type="tool",
                                response_metadata={},
                                additional_kwargs={},
                                tool_call_id="call_12345")

        result = convert_message_to_dfw(message)

        assert isinstance(result, ToolMessage)
        assert result.content == "Tool execution result"
        assert result.role == "tool"
        assert result.tool_call_id == "call_12345"

    def test_convert_message_with_unknown_type_fallback(self):
        """Test converting message with unknown type falls back to default role."""
        message = OpenAIMessage(content="No type specified", type="unknown", response_metadata={}, additional_kwargs={})

        result = convert_message_to_dfw(message)

        assert isinstance(result, UserMessage)  # Should fallback to user
        assert result.content == "No type specified"

    def test_convert_message_with_human_type(self):
        """Test converting message with 'human' type."""
        message = OpenAIMessage(content="Human message", type="human", response_metadata={}, additional_kwargs={})

        result = convert_message_to_dfw(message)

        assert isinstance(result, UserMessage)
        assert result.content == "Human message"
        assert result.role == "user"


class TestValidateAndConvertTools:
    """Test suite for validate_and_convert_tools function."""

    def test_validate_and_convert_tools_with_valid_schema(self):
        """Test validating and converting valid tools schema."""
        tools_schema = [{
            "function": {
                "name": "get_weather",
                "description": "Get current weather information",
                "parameters": {
                    "type": "object", "properties": {
                        "location": {
                            "type": "string"
                        }
                    }, "required": ["location"]
                }
            }
        }]

        result = validate_and_convert_tools(tools_schema)

        assert len(result) == 1
        assert isinstance(result[0], RequestTool)
        assert result[0].type == "function"
        assert result[0].function.name == "get_weather"
        assert result[0].function.description == "Get current weather information"

    def test_validate_and_convert_tools_with_tool_schema_object(self):
        """Test validating and converting ToolSchema objects."""
        from nat.data_models.intermediate_step import ToolDetails
        from nat.data_models.intermediate_step import ToolParameters

        tool_schema = ToolSchema(type="function",
                                 function=ToolDetails(name="calculate",
                                                      description="Perform calculations",
                                                      parameters=ToolParameters(
                                                          properties={"expression": {
                                                              "type": "string"
                                                          }},
                                                          required=["expression"])))

        result = validate_and_convert_tools([tool_schema])

        assert len(result) == 1
        assert result[0].function.name == "calculate"
        assert result[0].function.description == "Perform calculations"

    def test_validate_and_convert_tools_with_invalid_tool_type(self):
        """Test validating tools with invalid tool type."""
        tools_schema = [
            "invalid_tool", {
                "function": {
                    "name": "valid", "description": "desc", "parameters": {
                        "properties": {}, "required": []
                    }
                }
            }
        ]

        with patch(
                'nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter.logger'
        ) as mock_logger:
            result = validate_and_convert_tools(tools_schema)

            assert len(result) == 1
            assert result[0].function.name == "valid"
            mock_logger.warning.assert_called()

    def test_validate_and_convert_tools_with_missing_function_key(self):
        """Test validating tools with missing 'function' key."""
        tools_schema = [{
            "type": "function"
        },
                        {
                            "function": {
                                "name": "valid",
                                "description": "desc",
                                "parameters": {
                                    "properties": {}, "required": []
                                }
                            }
                        }]

        with patch(
                'nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter.logger'
        ) as mock_logger:
            result = validate_and_convert_tools(tools_schema)

            assert len(result) == 1
            assert result[0].function.name == "valid"
            mock_logger.warning.assert_called()

    def test_validate_and_convert_tools_with_invalid_function_type(self):
        """Test validating tools with invalid function type."""
        tools_schema = [{
            "function": "not_a_dict"
        },
                        {
                            "function": {
                                "name": "valid",
                                "description": "desc",
                                "parameters": {
                                    "properties": {}, "required": []
                                }
                            }
                        }]

        with patch(
                'nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter.logger'
        ) as mock_logger:
            result = validate_and_convert_tools(tools_schema)

            assert len(result) == 1
            assert result[0].function.name == "valid"
            mock_logger.warning.assert_called()

    def test_validate_and_convert_tools_with_missing_required_fields(self):
        """Test validating tools with missing required fields."""
        tools_schema = [
            {
                "function": {
                    "name": "incomplete1"
                }
            },  # Missing description and parameters
            {
                "function": {
                    "name": "incomplete2", "description": "desc"
                }
            },  # Missing parameters
            {
                "function": {
                    "name": "complete", "description": "desc", "parameters": {
                        "properties": {}, "required": []
                    }
                }
            }  # Complete
        ]

        with patch(
                'nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter.logger'
        ) as mock_logger:
            result = validate_and_convert_tools(tools_schema)

            assert len(result) == 1
            assert result[0].function.name == "complete"
            assert mock_logger.warning.call_count == 2  # Two warnings for incomplete tools

    def test_validate_and_convert_tools_with_function_creation_error(self):
        """Test handling errors during FunctionDetails creation."""
        tools_schema = [{
            "function": {
                "name": "valid_name",
                "description": "valid_desc",
                "parameters": "invalid_parameters"  # Should be dict, not string
            }
        }]

        with patch(
                'nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter.logger'
        ) as mock_logger:
            result = validate_and_convert_tools(tools_schema)

            assert len(result) == 0  # Should return empty list due to creation error
            mock_logger.warning.assert_called()

    def test_validate_and_convert_tools_with_empty_list(self):
        """Test validating empty tools list."""
        result = validate_and_convert_tools([])
        assert len(result) == 0

    def test_validate_and_convert_tools_with_multiple_valid_tools(self):
        """Test validating multiple valid tools."""
        tools_schema = [{
            "function": {
                "name": "tool1", "description": "desc1", "parameters": {
                    "properties": {}, "required": []
                }
            }
        },
                        {
                            "function": {
                                "name": "tool2",
                                "description": "desc2",
                                "parameters": {
                                    "properties": {}, "required": []
                                }
                            }
                        },
                        {
                            "function": {
                                "name": "tool3",
                                "description": "desc3",
                                "parameters": {
                                    "properties": {}, "required": []
                                }
                            }
                        }]

        result = validate_and_convert_tools(tools_schema)

        assert len(result) == 3
        assert result[0].function.name == "tool1"
        assert result[1].function.name == "tool2"
        assert result[2].function.name == "tool3"


class TestConvertChatResponse:
    """Test suite for convert_chat_response function."""

    def test_convert_chat_response_basic(self):
        """Test converting basic chat response."""
        chat_response = {
            "message": {
                "content": "Hello, how can I help?",
                "response_metadata": {
                    "finish_reason": "stop"
                },
                "additional_kwargs": {}
            }
        }

        result = convert_chat_response(chat_response, "test_span", 0)

        assert isinstance(result, ResponseChoice)
        assert result.message.content == "Hello, how can I help?"
        assert result.message.role == "assistant"
        assert result.finish_reason == FinishReason.STOP
        assert result.index == 0

    def test_convert_chat_response_with_tool_calls(self):
        """Test converting chat response with tool calls."""
        chat_response = {
            "message": {
                "content": "Let me search for that",
                "response_metadata": {
                    "finish_reason": "tool_calls"
                },
                "additional_kwargs": {
                    "tool_calls": [{
                        "function": {
                            "name": "search", "arguments": '{"query": "test"}'
                        }
                    }]
                }
            }
        }

        result = convert_chat_response(chat_response, "test_span", 1)

        assert isinstance(result, ResponseChoice)
        assert result.message.content is None  # Content should be None when tool calls exist
        assert result.message.role == "assistant"
        assert result.finish_reason == FinishReason.TOOL_CALLS
        assert result.index == 1
        assert len(result.message.tool_calls) == 1
        assert result.message.tool_calls[0].function.name == "search"

    def test_convert_chat_response_with_length_finish_reason(self):
        """Test converting chat response with length finish reason."""
        chat_response = {
            "message": {
                "content": "Response cut off due to length",
                "response_metadata": {
                    "finish_reason": "length"
                },
                "additional_kwargs": {}
            }
        }

        result = convert_chat_response(chat_response, "test_span", 0)

        assert result.finish_reason == FinishReason.LENGTH

    def test_convert_chat_response_with_unknown_finish_reason(self):
        """Test converting chat response with unknown finish reason."""
        chat_response = {
            "message": {
                "content": "Response with unknown finish reason",
                "response_metadata": {
                    "finish_reason": "unknown_reason"
                },
                "additional_kwargs": {}
            }
        }

        result = convert_chat_response(chat_response, "test_span", 0)

        assert result.finish_reason is None  # Should be None for unmapped finish reasons

    def test_convert_chat_response_missing_message(self):
        """Test converting chat response with missing message."""
        chat_response = {}

        try:
            convert_chat_response(chat_response, "test_span", 0)
            assert False, "Expected ValueError for missing message"
        except (ValueError, TypeError) as e:
            # Either ValueError for missing message or TypeError for finish_reason handling
            assert "Chat response missing message" in str(e) or "unhashable type" in str(e)

    def test_convert_chat_response_with_none_message(self):
        """Test converting chat response with None message."""
        chat_response = {"message": None}

        try:
            convert_chat_response(chat_response, "test_span", 0)
            assert False, "Expected ValueError for None message"
        except ValueError as e:
            assert "Chat response missing message" in str(e)

    def test_convert_chat_response_with_none_additional_kwargs(self):
        """Test converting chat response with None additional_kwargs."""
        chat_response = {
            "message": {
                "content": "Response with None additional_kwargs",
                "response_metadata": {
                    "finish_reason": "stop"
                },
                "additional_kwargs": None
            }
        }

        result = convert_chat_response(chat_response, "test_span", 0)

        assert isinstance(result, ResponseChoice)
        assert result.message.content == "Response with None additional_kwargs"
        assert result.message.tool_calls is None

    def test_convert_chat_response_with_none_tool_calls(self):
        """Test converting chat response with None tool_calls."""
        chat_response = {
            "message": {
                "content": "Response with None tool_calls",
                "response_metadata": {
                    "finish_reason": "stop"
                },
                "additional_kwargs": {
                    "tool_calls": None
                }
            }
        }

        result = convert_chat_response(chat_response, "test_span", 0)

        assert isinstance(result, ResponseChoice)
        assert result.message.content == "Response with None tool_calls"
        assert result.message.tool_calls is None


class TestConvertLangchainOpenai:
    """Test suite for convert_langchain_openai function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client_id = "test_client_123"
        self.basic_span = Span(name="test_openai_span",
                               context=SpanContext(),
                               attributes={
                                   "nat.subspan.name": "gpt-4",
                                   "nat.function.name": "test_workload",
                                   "nat.event_timestamp": 1642780800,
                                   "llm.token_count.prompt": 100,
                                   "llm.token_count.completion": 50,
                                   "llm.token_count.total": 150,
                                   "nat.usage.num_llm_calls": 1,
                                   "nat.usage.seconds_between_calls": 0
                               })

    def test_convert_langchain_openai_basic_functionality(self):
        """Test basic conversion functionality."""
        from unittest.mock import MagicMock

        # Create test messages
        messages = [
            OpenAIMessage(content="Hello", type="user", response_metadata={}, additional_kwargs={}),
            OpenAIMessage(content="Hi there!", type="assistant", response_metadata={}, additional_kwargs={})
        ]

        # Create mock trace source to bypass validation
        source = MagicMock(spec=OpenAITraceSource)
        source.input_value = messages
        source.metadata = MagicMock()
        source.metadata.tools_schema = []
        source.metadata.chat_responses = [{
            "message": {
                "content": "Hi there!", "response_metadata": {
                    "finish_reason": "stop"
                }, "additional_kwargs": {}
            }
        }]
        source.client_id = self.client_id

        trace_container = TraceContainer(source=source, span=self.basic_span)

        result = convert_langchain_openai(trace_container)

        assert isinstance(result, DFWESRecord)
        assert len(result.request.messages) == 2
        assert isinstance(result.request.messages[0], UserMessage)
        assert isinstance(result.request.messages[1], AssistantMessage)
        assert result.request.model == "gpt-4"
        assert result.client_id == self.client_id
        assert result.timestamp == 1642780800

    def test_convert_langchain_openai_with_tools(self):
        """Test conversion with tools schema."""
        from unittest.mock import MagicMock

        messages = [
            OpenAIMessage(content="Use the weather tool", type="user", response_metadata={}, additional_kwargs={})
        ]

        tools_schema = [{
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object", "properties": {
                        "location": {
                            "type": "string"
                        }
                    }, "required": ["location"]
                }
            }
        }]

        # Create mock trace source to bypass validation
        source = MagicMock(spec=OpenAITraceSource)
        source.input_value = messages
        source.metadata = MagicMock()
        source.metadata.tools_schema = tools_schema
        source.metadata.chat_responses = [{
            "message": {
                "content": None,
                "response_metadata": {
                    "finish_reason": "tool_calls"
                },
                "additional_kwargs": {
                    "tool_calls": [{
                        "function": {
                            "name": "get_weather", "arguments": '{"location": "NY"}'
                        }
                    }]
                }
            }
        }]
        source.client_id = self.client_id

        trace_container = TraceContainer(source=source, span=self.basic_span)

        result = convert_langchain_openai(trace_container)

        assert len(result.request.tools) == 1
        assert result.request.tools[0].function.name == "get_weather"
        assert len(result.response.choices) == 1
        if result.response.choices[0].message.tool_calls is not None:
            assert len(result.response.choices[0].message.tool_calls) == 1

    def test_convert_langchain_openai_with_multiple_chat_responses(self):
        """Test conversion with multiple chat responses."""
        from unittest.mock import MagicMock

        messages = [
            OpenAIMessage(content="Generate multiple responses",
                          type="user",
                          response_metadata={},
                          additional_kwargs={})
        ]

        chat_responses = [{
            "message": {
                "content": "Response 1", "response_metadata": {
                    "finish_reason": "stop"
                }, "additional_kwargs": {}
            }
        },
                          {
                              "message": {
                                  "content": "Response 2",
                                  "response_metadata": {
                                      "finish_reason": "stop"
                                  },
                                  "additional_kwargs": {}
                              }
                          },
                          {
                              "message": {
                                  "content": "Response 3",
                                  "response_metadata": {
                                      "finish_reason": "length"
                                  },
                                  "additional_kwargs": {}
                              }
                          }]

        # Create mock trace source to bypass validation
        source = MagicMock(spec=OpenAITraceSource)
        source.input_value = messages
        source.metadata = MagicMock()
        source.metadata.tools_schema = []
        source.metadata.chat_responses = chat_responses
        source.client_id = self.client_id

        trace_container = TraceContainer(source=source, span=self.basic_span)

        result = convert_langchain_openai(trace_container)

        assert len(result.response.choices) == 3
        assert result.response.choices[0].index == 0
        assert result.response.choices[1].index == 1
        assert result.response.choices[2].index == 2
        assert result.response.choices[2].finish_reason == FinishReason.LENGTH

    def test_convert_langchain_openai_message_conversion_error(self):
        """Test handling of message conversion errors."""
        from unittest.mock import MagicMock

        # Create an invalid message that will cause conversion to fail
        invalid_message = OpenAIMessage(content=None, type="user", response_metadata={}, additional_kwargs={})

        # Create mock trace source to bypass validation
        source = MagicMock(spec=OpenAITraceSource)
        source.input_value = [invalid_message]
        source.metadata = MagicMock()
        source.metadata.tools_schema = []
        source.metadata.chat_responses = []
        source.client_id = self.client_id

        trace_container = TraceContainer(source=source, span=self.basic_span)

        try:
            convert_langchain_openai(trace_container)
            assert False, "Expected error for invalid message"
        except (ValueError, AssertionError) as e:
            # Either AssertionError for None content or ValueError from message conversion wrapper
            assert "User message content cannot be None" in str(
                e) or "Failed to convert message in trace source" in str(e)

    def test_convert_langchain_openai_chat_response_conversion_error(self):
        """Test handling of chat response conversion errors."""
        from unittest.mock import MagicMock

        messages = [OpenAIMessage(content="Test", type="user", response_metadata={}, additional_kwargs={})]

        # Invalid chat response (missing message)
        invalid_chat_responses = [{"invalid": "response"}]

        # Create mock trace source to bypass validation
        source = MagicMock(spec=OpenAITraceSource)
        source.input_value = messages
        source.metadata = MagicMock()
        source.metadata.tools_schema = []
        source.metadata.chat_responses = invalid_chat_responses
        source.client_id = self.client_id

        trace_container = TraceContainer(source=source, span=self.basic_span)

        try:
            convert_langchain_openai(trace_container)
            assert False, "Expected error for invalid chat response"
        except (ValueError, TypeError) as e:
            # Either TypeError for unhashable dict used as key or ValueError from chat response conversion wrapper
            assert "unhashable type" in str(e) or "Failed to convert chat response 0" in str(e)

    def test_convert_langchain_openai_no_chat_responses(self):
        """Test handling when there are no chat responses."""
        from unittest.mock import MagicMock

        messages = [OpenAIMessage(content="Test", type="user", response_metadata={}, additional_kwargs={})]

        # Create mock trace source to bypass validation
        source = MagicMock(spec=OpenAITraceSource)
        source.input_value = messages
        source.metadata = MagicMock()
        source.metadata.tools_schema = []
        source.metadata.chat_responses = []
        source.client_id = self.client_id

        trace_container = TraceContainer(source=source, span=self.basic_span)

        try:
            convert_langchain_openai(trace_container)
            assert False, "Expected ValueError for no chat responses"
        except ValueError as e:
            assert "No valid response choices found" in str(e)
            assert self.basic_span.name in str(e)

    def test_convert_langchain_openai_with_custom_response_id(self):
        """Test conversion with custom response ID in span attributes."""
        from unittest.mock import MagicMock

        messages = [OpenAIMessage(content="Test", type="user", response_metadata={}, additional_kwargs={})]

        # Create mock trace source to bypass validation
        source = MagicMock(spec=OpenAITraceSource)
        source.input_value = messages
        source.metadata = MagicMock()
        source.metadata.tools_schema = []
        source.metadata.chat_responses = [{
            "message": {
                "content": "Response", "response_metadata": {
                    "finish_reason": "stop"
                }, "additional_kwargs": {}
            }
        }]
        source.client_id = self.client_id

        # Add custom response ID to span
        span_with_response_id = Span(name="test_span",
                                     context=SpanContext(),
                                     attributes={
                                         **self.basic_span.attributes, "response.id": "custom_response_123"
                                     })

        trace_container = TraceContainer(source=source, span=span_with_response_id)

        result = convert_langchain_openai(trace_container)

        assert result.response.id == "custom_response_123"

    def test_convert_langchain_openai_dfw_record_creation_error(self):
        """Test handling of DFWESRecord creation errors."""
        from unittest.mock import MagicMock

        messages = [OpenAIMessage(content="Test", type="user", response_metadata={}, additional_kwargs={})]

        # Create mock trace source to bypass validation
        source = MagicMock(spec=OpenAITraceSource)
        source.input_value = messages
        source.metadata = MagicMock()
        source.metadata.tools_schema = []
        source.metadata.chat_responses = [{
            "message": {
                "content": "Response", "response_metadata": {
                    "finish_reason": "stop"
                }, "additional_kwargs": {}
            }
        }]
        source.client_id = self.client_id

        trace_container = TraceContainer(source=source, span=self.basic_span)

        # Mock DFWESRecord creation to fail
        with patch(
                'nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter.DFWESRecord',
                side_effect=Exception("Creation failed")):
            try:
                convert_langchain_openai(trace_container)
                assert False, "Expected ValueError for DFWESRecord creation failure"
            except ValueError as e:
                assert "Failed to create DFWESRecord" in str(e)
                assert self.basic_span.name in str(e)


class TestConstants:
    """Test suite for constants and mappings."""

    def test_role_map_completeness(self):
        """Test that ROLE_MAP contains expected mappings."""
        expected_roles = ["human", "user", "assistant", "ai", "system", "tool", "function", "chain"]

        for role in expected_roles:
            assert role in ROLE_MAP, f"Expected role '{role}' not found in ROLE_MAP"

        assert ROLE_MAP["human"] == "user"
        assert ROLE_MAP["ai"] == "assistant"
        assert ROLE_MAP["chain"] == "function"

    def test_finish_reason_map_completeness(self):
        """Test that FINISH_REASON_MAP contains expected mappings."""
        expected_reasons = ["tool_calls", "stop", "length"]

        for reason in expected_reasons:
            assert reason in FINISH_REASON_MAP, f"Expected finish reason '{reason}' not found in FINISH_REASON_MAP"

        assert FINISH_REASON_MAP["tool_calls"] == FinishReason.TOOL_CALLS
        assert FINISH_REASON_MAP["stop"] == FinishReason.STOP
        assert FINISH_REASON_MAP["length"] == FinishReason.LENGTH


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple functions."""

    def test_complete_conversion_workflow(self):
        """Test complete end-to-end conversion workflow."""
        # Create complex trace with user message, assistant response with tool calls
        user_message = OpenAIMessage(
            content="What's the weather in New York?",
            type="human",  # Test role conversion
            response_metadata={},
            additional_kwargs={})

        tool_schema = {
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string", "description": "The city and state"
                        },
                        "unit": {
                            "type": "string", "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }

        chat_response = {
            "message": {
                "content": None,  # No content when tool calls are present
                "response_metadata": {
                    "finish_reason": "tool_calls"
                },
                "additional_kwargs": {
                    "tool_calls": [{
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "New York, NY", "unit": "fahrenheit"}'
                        }
                    }]
                }
            }
        }

        # Create mock trace source to bypass validation
        from unittest.mock import MagicMock
        source = MagicMock(spec=OpenAITraceSource)
        source.input_value = [user_message]
        source.metadata = MagicMock()
        source.metadata.tools_schema = [tool_schema]
        source.metadata.chat_responses = [chat_response]
        source.client_id = "integration_test_client"

        span = Span(name="weather_query_span",
                    context=SpanContext(),
                    attributes={
                        "nat.subspan.name": "gpt-4-turbo",
                        "nat.function.name": "weather_assistant",
                        "nat.event_timestamp": 1642780800,
                        "llm.token_count.prompt": 45,
                        "llm.token_count.completion": 25,
                        "llm.token_count.total": 70,
                        "nat.usage.num_llm_calls": 1,
                        "nat.usage.seconds_between_calls": 0,
                        "response.id": "chatcmpl-weather123"
                    })

        trace_container = TraceContainer(source=source, span=span)

        result = convert_langchain_openai(trace_container)

        # Verify request structure
        assert len(result.request.messages) == 1
        assert isinstance(result.request.messages[0], UserMessage)
        assert result.request.messages[0].content == "What's the weather in New York?"
        assert result.request.messages[0].role == "user"  # Converted from "human"

        assert len(result.request.tools) == 1
        assert result.request.tools[0].function.name == "get_current_weather"
        assert result.request.model == "gpt-4-turbo"

        # Verify response structure
        assert len(result.response.choices) == 1
        choice = result.response.choices[0]
        assert choice.message.content is None
        assert choice.message.role == "assistant"
        assert choice.finish_reason == FinishReason.TOOL_CALLS
        assert len(choice.message.tool_calls) == 1

        tool_call = choice.message.tool_calls[0]
        assert tool_call.function.name == "get_current_weather"
        assert tool_call.function.arguments == {"location": "New York, NY", "unit": "fahrenheit"}

        # Verify metadata
        assert result.response.id == "chatcmpl-weather123"
        assert result.response.model == "gpt-4-turbo"
        assert result.timestamp == 1642780800
        assert result.client_id == "integration_test_client"
        assert result.workload_id == "weather_assistant"
