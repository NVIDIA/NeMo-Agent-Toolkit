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

import json
import logging

from nat.data_models.intermediate_step import ToolSchema
from nat.plugins.data_flywheel.observability.processor.dfw_record.common import extract_timestamp
from nat.plugins.data_flywheel.observability.processor.dfw_record.common import extract_usage_info
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import AssistantMessage
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import DFWESRecord
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import FinishReason
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import Function
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import FunctionDetails
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import FunctionMessage
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import Message
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import Request
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import RequestTool
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import Response
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import ResponseChoice
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import ResponseMessage
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import SystemMessage
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import ToolCall
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import ToolMessage
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import UserMessage
from nat.plugins.data_flywheel.observability.schema.langchain.langchain_message import LangChainMessage
from nat.plugins.data_flywheel.observability.schema.trace_source import TraceSource

logger = logging.getLogger(__name__)

DEFAULT_ROLE = "user"

# Role mapping from various role types to standard roles
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
    "chain": "function"
}

FINISH_REASON_MAP = {"tool_calls": FinishReason.TOOL_CALLS, "stop": FinishReason.STOP, "length": FinishReason.LENGTH}


def _convert_role(role: str) -> str:
    """Convert role to standard format with fallback."""
    return ROLE_MAP.get(role, DEFAULT_ROLE)


def _create_message_by_role(role: str, content: str, **kwargs) -> Message | None:
    """Factory function for creating messages by role.

    Args:
        role (str): The message role
        content (str): The message content
        **kwargs: Additional role-specific parameters

    Returns:
        Message | None: The appropriate message type for the role
    """
    role = _convert_role(role)

    match role:
        case "user":
            return UserMessage(content=content, role="user")
        case "system":
            return SystemMessage(content=content, role="system")
        case "assistant":
            tool_calls = kwargs.get("tool_calls", [])
            return AssistantMessage(content=content, role="assistant", tool_calls=tool_calls if tool_calls else None)
        case "tool":
            tool_call_id = kwargs.get("tool_call_id", "")
            return ToolMessage(content=content, role="tool", tool_call_id=tool_call_id)
        case "function":
            return FunctionMessage(content=content, role="function")
        case _:
            logger.warning("Unsupported message role: %s", role)


def _create_tool_calls(tool_calls_data: list) -> list[ToolCall]:
    """Create standardized tool calls from raw data.

    Args:
        tool_calls_data (list): Raw tool call data

    Returns:
        list[ToolCall]: List of validated tool calls
    """
    validated_tool_calls = []

    for tool_call in tool_calls_data:
        if not isinstance(tool_call, dict):
            continue

        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            continue

        # Parse function arguments safely
        function_args = {}
        try:
            raw_args = function.get("arguments", "{}")
            if isinstance(raw_args, str):
                function_args = json.loads(raw_args) or {}
            elif isinstance(raw_args, dict):
                function_args = raw_args
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in function arguments: %s", raw_args)
            function_args = {}

        validated_tool_calls.append(
            ToolCall(type="function",
                     function=Function(name=function.get("name", "unknown") or "unknown", arguments=function_args)))

    return validated_tool_calls


def _convert_message_to_dfw(message: LangChainMessage) -> Message | None:
    """Convert a message to appropriate DFW message type with improved structure.

    Args:
        message (LangChainMessage): The message to convert

    Returns:
        Message | None: The converted message
    """

    # Get content
    if "content" in message.response_metadata:
        content = message.response_metadata.get("content", None)
    else:
        content = message.content

    # Get role
    role = message.type or DEFAULT_ROLE

    # Handle tool calls for assistant messages
    tool_calls = []
    raw_tool_calls = message.additional_kwargs.get("tool_calls", [])
    if raw_tool_calls:
        tool_calls = _create_tool_calls(raw_tool_calls)

    # Get tool_call_id for tool messages
    tool_call_id = message.additional_kwargs.get("tool_call_id", "")

    return _create_message_by_role(role=role, content=str(content), tool_calls=tool_calls, tool_call_id=tool_call_id)


def _validate_and_convert_tools(tools_schema: list) -> list[RequestTool]:
    """Validate and convert tools schema to RequestTool format.

    Args:
        tools_schema (list): Raw tools schema

    Returns:
        list[RequestTool]: Validated request tools
    """
    request_tools = []

    for tool in tools_schema:
        if isinstance(tool, ToolSchema):
            tool = tool.model_dump()

        if not isinstance(tool, dict):
            logger.warning("Invalid tool schema: expected dict, got %s", type(tool))
            continue

        if "function" not in tool:
            logger.warning("Tool schema missing 'function' key: %s", tool)
            continue

        function_details = tool["function"]
        if not isinstance(function_details, dict):
            logger.warning("Tool function details must be dict: %s", function_details)
            continue

        # Validate required function fields
        required_fields = ["name", "description", "parameters"]
        if not all(field in function_details for field in required_fields):
            logger.warning("Tool function missing required fields %s: %s", required_fields, function_details)
            continue

        try:
            # Create FunctionDetails object from dict
            function_obj = FunctionDetails(**function_details)
            request_tools.append(RequestTool(type="function", function=function_obj))
        except Exception as e:
            logger.warning("Failed to create RequestTool: %s", str(e))
            continue

    return request_tools


def _convert_chat_response(chat_response: dict, span_name: str = "", index: int = 0) -> ResponseChoice | None:
    """Convert a chat response to a DFW payload with better error context.

    Args:
        chat_response (dict): The chat response to convert
        span_name (str): Span name for error context

    Returns:
        ResponseChoice | None: The converted chat response
    """
    message = chat_response.get("message", {})
    if message is None:
        logger.warning("Chat response missing message for span %s", span_name)
        return None

    # Get content
    content = message.get("content", "")

    # Get role and finish reason
    response_message = message.get("response_metadata", {})
    finish_reason = response_message.get("finish_reason", {})

    # Get tool calls using the centralized function
    validated_tool_calls = []
    additional_kwargs = message.get("additional_kwargs", {})
    if additional_kwargs is not None:
        tool_calls = additional_kwargs.get("tool_calls", [])
        if tool_calls is not None:
            validated_tool_calls = _create_tool_calls(tool_calls)

    # Map finish reason to enum
    mapped_finish_reason = FINISH_REASON_MAP.get(finish_reason)

    response_choice = ResponseChoice(message=ResponseMessage(
        content=content, role="assistant", tool_calls=validated_tool_calls if validated_tool_calls else None),
                                     finish_reason=mapped_finish_reason,
                                     index=index)

    return response_choice


def convert_langchain_openai(trace_source: TraceSource, client_id: str = "nat_test") -> DFWESRecord | None:

    # Convert messages
    messages = []
    for message in trace_source.source.input_value:
        msg_result = _convert_message_to_dfw(message)
        if msg_result is not None:
            messages.append(msg_result)

    # Get tools schema
    tools_schema = trace_source.source.metadata.tools_schema
    request_tools = _validate_and_convert_tools(tools_schema) if tools_schema else []

    # Construct a Request object
    model_name = str(trace_source.span.attributes.get("nat.subspan.name", "unknown"))

    # These parameters don't exist in current span structure, so set to None
    # The schema allows them to be optional
    temperature = None
    max_tokens = None

    request = Request(messages=messages,
                      model=model_name,
                      tools=request_tools if request_tools else None,
                      temperature=temperature,
                      max_tokens=max_tokens)

    # Transform chat responses
    response_choices = []
    chat_responses = trace_source.source.metadata.chat_responses or []
    for idx, chat_response in enumerate(chat_responses):
        response_choice = _convert_chat_response(chat_response, trace_source.span.name, index=idx)
        if response_choice is not None:
            response_choices.append(response_choice)

    # Require at least one response choice
    if not response_choices:
        logger.warning("No valid response choices found in span %s", trace_source.span.name)
        return None

    # Get timestamp with better error handling
    timestamp_int = extract_timestamp(trace_source.span)

    # Extract additional response metadata from span
    response_id = trace_source.span.attributes.get(
        "response.id") or f"response-{trace_source.span.name}-{timestamp_int}"
    response_object = "chat.completion"  # Standard OpenAI object type
    created_timestamp = timestamp_int  # Use same timestamp as the record

    # Extract usage information from span attributes using structured models
    usage_info = extract_usage_info(trace_source.span)
    responses = Response(choices=response_choices,
                         id=response_id,
                         object=response_object,
                         created=created_timestamp,
                         model=model_name,
                         usage=usage_info.model_dump() if usage_info else None)

    # Get workload_id
    workload_id = trace_source.span.attributes.get("nat.function.name", "unknown")

    try:
        dfw_payload = DFWESRecord(request=request,
                                  response=responses,
                                  timestamp=timestamp_int,
                                  workload_id=str(workload_id),
                                  client_id=client_id,
                                  error_details=None)
        logger.debug("Successfully converted span %s to DFW record", trace_source.span.name)
        return dfw_payload
    except Exception as e:
        logger.error("Failed to create DFW record for span %s: %s", trace_source.span.name, str(e))
        return None
