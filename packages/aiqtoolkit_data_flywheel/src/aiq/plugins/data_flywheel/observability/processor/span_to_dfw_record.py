# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import typing

from aiq.data_models.intermediate_step import TokenUsageBaseModel
from aiq.data_models.intermediate_step import UsageInfo
from aiq.data_models.span import Span
from aiq.plugins.data_flywheel.observability.schema.dfw_record import AssistantMessage
from aiq.plugins.data_flywheel.observability.schema.dfw_record import DFWRecord
from aiq.plugins.data_flywheel.observability.schema.dfw_record import FinishReason
from aiq.plugins.data_flywheel.observability.schema.dfw_record import Function
from aiq.plugins.data_flywheel.observability.schema.dfw_record import FunctionDetails
from aiq.plugins.data_flywheel.observability.schema.dfw_record import FunctionMessage
from aiq.plugins.data_flywheel.observability.schema.dfw_record import Message
from aiq.plugins.data_flywheel.observability.schema.dfw_record import Request
from aiq.plugins.data_flywheel.observability.schema.dfw_record import RequestTool
from aiq.plugins.data_flywheel.observability.schema.dfw_record import Response
from aiq.plugins.data_flywheel.observability.schema.dfw_record import ResponseChoice
from aiq.plugins.data_flywheel.observability.schema.dfw_record import ResponseMessage
from aiq.plugins.data_flywheel.observability.schema.dfw_record import SystemMessage
from aiq.plugins.data_flywheel.observability.schema.dfw_record import ToolCall
from aiq.plugins.data_flywheel.observability.schema.dfw_record import ToolMessage
from aiq.plugins.data_flywheel.observability.schema.dfw_record import UserMessage

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CLIENT_ID = "aiq-toolkit"
DEFAULT_ROLE = "user"
DEFAULT_FUNCTION_NAME = "unknown"
UNKNOWN_MODEL = "unknown"

# Required span attributes
REQUIRED_ATTRIBUTES = ["input.value", "aiq.metadata"]

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

# Finish reason mapping
FINISH_REASON_MAP = {"tool_calls": FinishReason.TOOL_CALLS, "stop": FinishReason.STOP, "length": FinishReason.LENGTH}


def _validate_span_attributes(span: Span) -> bool:
    """Validate that span has required attributes for DFW conversion.

    Args:
        span (Span): The span to validate

    Returns:
        bool: True if span has required attributes, False otherwise
    """
    if span.attributes is None:
        return False

    # Check for input.value which is essential for message conversion
    if "input.value" not in span.attributes:
        return False

    return True


def _get_structured_attribute(span: Span, attribute_name: str, default_value: typing.Any = "{}") -> dict | list | None:
    """Get a structured attribute from a span with improved error handling.

    Args:
        span (Span): The span to get the attribute from
        attribute_name (str): The name of the attribute to get
        default_value (typing.Any): The default value to return if the attribute is not found

    Returns:
        dict | list | None: The parsed attribute value or None if parsing fails
    """
    try:
        serialized_attribute = span.attributes.get(attribute_name, default_value)
        if isinstance(serialized_attribute, (dict, list)):
            return serialized_attribute
        deserialized_attribute = json.loads(serialized_attribute)
        return deserialized_attribute
    except (json.JSONDecodeError, TypeError) as e:
        logger.error("Failed to parse attribute %s for span %s: %s", attribute_name, span.name, str(e))
        return None


def _convert_role(role: str) -> str:
    """Convert role to standard format with fallback."""
    return ROLE_MAP.get(role, DEFAULT_ROLE)


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
                     function=Function(name=function.get("name", DEFAULT_FUNCTION_NAME) or DEFAULT_FUNCTION_NAME,
                                       arguments=function_args)))

    return validated_tool_calls


def _create_message_by_role(role: str, content: str, **kwargs) -> Message:
    """Factory function for creating messages by role.

    Args:
        role (str): The message role
        content (str): The message content
        **kwargs: Additional role-specific parameters

    Returns:
        Message: The appropriate message type for the role
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

    # elif role == "assistant":
    #     tool_calls = kwargs.get("tool_calls", [])
    #     return AssistantMessage(content=content, role="assistant", tool_calls=tool_calls if tool_calls else None)
    # elif role == "tool":
    #     tool_call_id = kwargs.get("tool_call_id", "")
    #     return ToolMessage(content=content, role="tool", tool_call_id=tool_call_id)
    # elif role == "function":
    #     return FunctionMessage(content=content, role="function")

    # # Default fallback
    # return UserMessage(content=content, role="user")


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


def _convert_message_to_dfw(message: dict) -> Message | None:
    """Convert a message to appropriate DFW message type with improved structure.

    Args:
        message (dict): The message to convert

    Returns:
        Message | None: The converted message
    """
    if not isinstance(message, dict):
        return None

    response_metadata = message.get("response_metadata", {})

    # Get content
    if "content" in response_metadata:
        content = response_metadata.get("content", None)
    else:
        content = message.get("content", "") or ""

    # Get role
    role = response_metadata.get("role", message.get("type", DEFAULT_ROLE))

    # Handle tool calls for assistant messages
    additional_kwargs = message.get("additional_kwargs", {})
    tool_calls = []
    if additional_kwargs and "tool_calls" in additional_kwargs:
        raw_tool_calls = additional_kwargs.get("tool_calls", [])
        if raw_tool_calls:
            tool_calls = _create_tool_calls(raw_tool_calls)

    # Get tool_call_id for tool messages
    tool_call_id = message.get("tool_call_id", "")

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


def span_to_dfw_record(span: Span, client_id: str = DEFAULT_CLIENT_ID) -> DFWRecord | None:
    """Convert a span to a DFW payload with improved validation and error handling.

    Args:
        span (Span): A span from the local AIQ library
        client_id (str): Client identifier for the DFW record

    Returns:
        DFWRecord | None: The converted DFW payload that is compatible with the NeMo Data Flywheel
    """
    # Validate span has required attributes
    if not _validate_span_attributes(span):
        logger.warning("Span %s missing required attributes for DFW conversion", span.name)
        return None

    # Transform request messages
    message_content_list: dict | list | None = _get_structured_attribute(span, "input.value")

    if not isinstance(message_content_list, list):
        logger.warning("Failed to extract messages from span %s: expected list, got %s",
                       span.name,
                       type(message_content_list))
        return None

    # Convert messages
    messages = []
    for message in message_content_list:
        msg_result = _convert_message_to_dfw(message)
        if msg_result is not None:
            messages.append(msg_result)

    if not messages:
        logger.warning("No valid messages found in span %s", span.name)
        return None

    # Get metadata and tools with validation
    metadata: dict | list | None = _get_structured_attribute(span, "aiq.metadata")

    if not isinstance(metadata, dict):
        logger.warning("Failed to extract metadata from span %s: expected dict, got %s", span.name, type(metadata))
        return None

    tools_schema = metadata.get("tools_schema", [])
    request_tools = _validate_and_convert_tools(tools_schema) if tools_schema else []

    # Construct a Request object
    model_name = str(span.attributes.get("aiq.subspan.name", UNKNOWN_MODEL))

    # These parameters don't exist in current span structure, so set to None
    # The schema allows them to be optional
    temperature = None
    max_tokens = None

    request = Request(messages=messages,
                      model=model_name,
                      tools=request_tools if request_tools else None,
                      temperature=temperature,
                      max_tokens=max_tokens)

    # Filter out unsupported message types for DFW
    unsupported_types = (ToolMessage, FunctionMessage)
    if (request.messages is not None) and (any(isinstance(msg, unsupported_types) for msg in request.messages)):
        logger.info("Span %s contains unsupported message types, skipping DFW conversion", span.name)
        return None

    # Transform response messages
    response_choices = []
    chat_responses = metadata.get("chat_responses", []) or []
    for idx, chat_response in enumerate(chat_responses):
        response_choice = _convert_chat_response(chat_response, span.name, index=idx)
        if response_choice is not None:
            response_choices.append(response_choice)

    # Require at least one response choice
    if not response_choices:
        logger.warning("No valid response choices found in span %s", span.name)
        return None

    # Get timestamp with better error handling
    timestamp = span.attributes.get("aiq.event_timestamp", 0)
    try:
        timestamp_int = int(float(str(timestamp)))
    except (ValueError, TypeError) as e:
        logger.warning("Invalid timestamp in span %s: %s, using 0", span.name, str(e))
        timestamp_int = 0

    # Extract additional response metadata from span
    response_id = span.attributes.get("response.id") or f"response-{span.name}-{timestamp_int}"
    response_object = "chat.completion"  # Standard OpenAI object type
    created_timestamp = timestamp_int  # Use same timestamp as the record

    # Extract usage information from span attributes using structured models
    token_usage = TokenUsageBaseModel(prompt_tokens=span.attributes.get("llm.token_count.prompt", 0),
                                      completion_tokens=span.attributes.get("llm.token_count.completion", 0),
                                      total_tokens=span.attributes.get("llm.token_count.total", 0))

    # Get additional usage metrics from span attributes
    num_llm_calls = span.attributes.get("aiq.usage.num_llm_calls", 0)
    seconds_between_calls = span.attributes.get("aiq.usage.seconds_between_calls", 0)

    usage_info = UsageInfo(token_usage=token_usage,
                           num_llm_calls=num_llm_calls,
                           seconds_between_calls=seconds_between_calls)

    responses = Response(choices=response_choices,
                         id=response_id,
                         object=response_object,
                         created=created_timestamp,
                         model=model_name,
                         usage=usage_info.model_dump() if usage_info else None)

    # Get workload_id
    workload_id = span.attributes.get("aiq.function.name", UNKNOWN_MODEL)

    try:
        dfw_payload = DFWRecord(request=request,
                                response=responses,
                                timestamp=timestamp_int,
                                workload_id=str(workload_id),
                                client_id=client_id,
                                error_details=None)
        logger.debug("Successfully converted span %s to DFW record", span.name)
        return dfw_payload
    except Exception as e:
        logger.error("Failed to create DFW record for span %s: %s", span.name, str(e))
        return None
