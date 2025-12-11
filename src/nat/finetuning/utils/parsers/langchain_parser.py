# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from langchain_core.messages import FunctionMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.finetuning.utils.parsers.common import extract_content
from nat.finetuning.utils.parsers.common import parse_generic_message

# Re-export for backwards compatibility and internal use
_extract_content = extract_content
_parse_generic_message = parse_generic_message


def parse_to_openai_message(message: IntermediateStep) -> dict | list[dict]:
    """
    Convert IntermediateStep to OpenAI-compatible message dictionary.

    Args:
        message: An IntermediateStep object representing a single message.
        previous_message: Previous message for context (reserved for future).

    Returns:
        A dictionary formatted for OpenAI API consumption.
    """

    # Handle different event types to determine role and extract content
    if message.event_type == IntermediateStepType.LLM_END:
        # Assistant message with potential tool calls
        result = _parse_assistant_message(message)
    elif message.event_type == IntermediateStepType.TOOL_END:
        # Tool response message
        result = _parse_tool_message(message)
    elif message.event_type == IntermediateStepType.LLM_START:
        # Extract user/system messages from the input
        result = _parse_input_message(message)
        # drop logprobs field if exists
        if "logprobs" in result:
            del result["logprobs"]
    else:
        # For other types, try to infer from the data
        result = _parse_generic_message(message)

    return result


def _parse_assistant_message(message: IntermediateStep) -> dict:
    """Parse an assistant message from LLM_END event."""
    result = {"role": "assistant"}
    # Get the generation from payload if available
    if message.data and message.data.payload:
        payload = message.data.payload
        msg = None
        if isinstance(payload, dict) and "message" in payload:
            # Handle dict payloads
            try:
                msg = AIMessage(**payload["message"])
            except Exception as _:
                try:
                    msg = AIMessageChunk(**payload["message"])
                except Exception as _:
                    msg = None

        # Handle ChatGeneration objects from LangChain
        if hasattr(payload, 'message'):
            msg = payload.message
        if msg:
            # Extract content
            if isinstance(msg, AIMessage):
                result["content"] = msg.content or ""

                # Extract tool calls if present
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    result["tool_calls"] = msg.tool_calls
                elif 'tool_calls' in msg.additional_kwargs:
                    tool_calls = msg.additional_kwargs['tool_calls']
                    result["tool_calls"] = tool_calls

                # Extract function call if present
                if hasattr(msg, 'function_call') and msg.function_call:
                    result["function_call"] = msg.function_call
                elif 'function_call' in msg.additional_kwargs:
                    func_call = msg.additional_kwargs['function_call']
                    result["function_call"] = func_call
            else:
                # Fallback to extracting content as string
                result["content"] = str(getattr(msg, 'content', msg))

            # Extract logprobs if available
            gen_info = getattr(msg, 'response_metadata', None)
            if gen_info and 'logprobs' in gen_info:
                result["logprobs"] = gen_info['logprobs']

    elif message.data and message.data.output:
        # Fallback to output field
        result["content"] = _extract_content(message.data.output)
    else:
        result["content"] = ""

    # Check for logprobs in data field
    logprobs = (getattr(message.data, 'logprobs', None) if message.data else None)
    if logprobs:
        result["logprobs"] = logprobs

    # if not logprobs, set to empty dict to avoid issues downstream
    if "logprobs" not in result:
        result["logprobs"] = {}

    return result


def _parse_tool_message(message: IntermediateStep) -> dict:
    """Parse a tool response message from TOOL_END event."""
    result = {"role": "tool"}

    # Extract tool output as content
    if message.data:
        if message.data.output:
            result["content"] = _extract_content(message.data.output)
        elif message.data.payload:
            result["content"] = _extract_content(message.data.payload)
        else:
            result["content"] = ""
    else:
        result["content"] = ""

    # Add tool_call_id if available from metadata or UUID
    if message.metadata and hasattr(message.metadata, 'tool_call_id'):
        result["tool_call_id"] = message.metadata.tool_call_id
    else:
        result["tool_call_id"] = 0

    return result


def _parse_input_message(message: IntermediateStep) -> dict | list[dict]:
    """Parse user or system messages from LLM_START event."""
    if not message.data or not message.data.input:
        return {"role": "user", "content": ""}

    input_data = message.data.input

    # Handle list of messages
    if isinstance(input_data, list) and len(input_data) > 0:
        # Get the last message in the list
        messages = []
        for msg in input_data:
            last_msg = msg
            # Handle BaseMessage objects
            if isinstance(last_msg, BaseMessage):
                messages.append(_parse_langchain_message(last_msg))
            # Handle dict messages
            elif isinstance(last_msg, dict):
                messages.append(_parse_dict_message(last_msg))
            # Handle string messages
            elif isinstance(last_msg, str):
                messages.append({"role": "user", "content": last_msg})
            else:
                messages.append({"role": "user", "content": str(last_msg)})
        return messages
    # Handle single message
    elif isinstance(input_data, BaseMessage):
        return _parse_langchain_message(input_data)
    elif isinstance(input_data, dict):
        return _parse_dict_message(input_data)
    else:
        return {"role": "user", "content": _extract_content(input_data)}


def _parse_langchain_message(msg: BaseMessage) -> dict:
    """Parse a LangChain BaseMessage object."""
    result = {}

    # Determine role based on message type
    if isinstance(msg, HumanMessage):
        result["role"] = "user"
    elif isinstance(msg, AIMessage):
        result["role"] = "assistant"
    elif isinstance(msg, SystemMessage):
        result["role"] = "system"
    elif isinstance(msg, ToolMessage):
        result["role"] = "tool"
        # Add tool_call_id if present
        if hasattr(msg, 'tool_call_id'):
            result["tool_call_id"] = msg.tool_call_id
    elif isinstance(msg, FunctionMessage):
        result["role"] = "function"
        # Add name if present
        if hasattr(msg, 'name'):
            result["name"] = msg.name
    else:
        # Default to user role for unknown message types
        result["role"] = "user"

    # Extract content
    result["content"] = msg.content or ""

    # Handle tool calls for AI messages
    if isinstance(msg, AIMessage):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            result["tool_calls"] = msg.tool_calls
        elif 'tool_calls' in msg.additional_kwargs:
            result["tool_calls"] = msg.additional_kwargs['tool_calls']

        if hasattr(msg, 'function_call') and msg.function_call:
            result["function_call"] = msg.function_call
        elif 'function_call' in msg.additional_kwargs:
            result["function_call"] = msg.additional_kwargs['function_call']

    return result


def _parse_dict_message(msg_dict: dict) -> dict:
    """Parse a dictionary-based message."""
    result = {}

    # Extract role
    #result["role"] = msg_dict.get("role", "user")
    if "role" in msg_dict:
        role = msg_dict["role"]
    elif "type" in msg_dict:
        role = msg_dict["type"]
    else:
        role = "user"

    if role == 'ai':
        role = 'assistant'
    elif role == 'human':
        role = 'user'

    result["role"] = role

    # Extract content
    if "content" in msg_dict:
        result["content"] = msg_dict["content"]
    elif "text" in msg_dict:
        result["content"] = msg_dict["text"]
    else:
        result["content"] = ""

    # Copy over optional fields
    optional_fields = ["tool_calls", "tool_call_id", "function_call", "name", "logprobs"]
    for field in optional_fields:
        if field in msg_dict:
            result[field] = msg_dict[field]

    return result
