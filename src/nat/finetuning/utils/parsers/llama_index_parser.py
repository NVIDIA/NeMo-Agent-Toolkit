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

from llama_index.core.llms import ChatResponse

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.finetuning.utils.parsers.common import extract_content
from nat.finetuning.utils.parsers.common import parse_generic_message

# Re-export for backwards compatibility and internal use
_extract_content = extract_content
_parse_generic_message = parse_generic_message


def parse_to_openai_message(message: IntermediateStep) -> dict:  # noqa: ARG001
    """
    Convert IntermediateStep to OpenAI-compatible message dictionary.

    Args:
        message: An IntermediateStep object representing a single message.
        previous_message: Previous message for context (reserved for future).

    Returns:
        A dictionary formatted for OpenAI API consumption.
    """
    result = {}

    # Handle different event types to determine role and extract content
    if message.event_type == IntermediateStepType.LLM_END:
        # Assistant message from ChatResponse
        result = _parse_assistant_message(message)
    elif message.event_type == IntermediateStepType.TOOL_END:
        # Tool/Function response message
        result = _parse_tool_message(message)
    elif message.event_type == IntermediateStepType.LLM_START:
        # Extract user/system messages from the input
        result = _parse_input_message(message)
    else:
        # For other types, try to infer from the data
        result = _parse_generic_message(message)

    return result


def _parse_assistant_message(message: IntermediateStep) -> dict:
    """Parse an assistant message from LLM_END event."""
    result = {"role": "assistant"}

    # Get the ChatResponse from payload if available
    if message.data and message.data.payload:
        payload = message.data.payload

        # Handle ChatResponse objects from LlamaIndex
        if isinstance(payload, ChatResponse):
            # Extract content from message blocks
            content = ""
            msg = getattr(payload, 'message', None)
            if msg and hasattr(msg, 'blocks'):
                try:
                    content = ''.join(block.text for block in msg.blocks)
                except (AttributeError, TypeError):
                    # Fallback to str representation
                    content = str(msg) if msg else ""
            elif msg:
                # Direct message content
                content = str(msg)
            result["content"] = content

            # Check for tool calls in additional_kwargs
            if (hasattr(payload, 'message') and hasattr(payload.message, 'additional_kwargs')):
                additional_kwargs = payload.message.additional_kwargs
                if 'tool_calls' in additional_kwargs:
                    result["tool_calls"] = additional_kwargs['tool_calls']
                if 'function_call' in additional_kwargs:
                    func_call = additional_kwargs['function_call']
                    result["function_call"] = func_call

            # Extract logprobs if available
            raw_attr = getattr(payload, 'raw', None)
            try:
                choice = raw_attr.choices[0] if raw_attr and hasattr(raw_attr, 'choices') else None
                if choice and hasattr(choice, 'logprobs') and choice.logprobs:
                    result["logprobs"] = choice.logprobs
            except (AttributeError, IndexError):
                pass

    elif message.data and message.data.output:
        # Fallback to output field
        result["content"] = _extract_content(message.data.output)
    else:
        result["content"] = ""

    # if not logprobs, set to empty dict to avoid issues downstream
    if "logprobs" not in result:
        result["logprobs"] = {}

    return result


def _parse_tool_message(message: IntermediateStep) -> dict:
    """Parse a tool/function response message from TOOL_END event."""
    result = {"role": "function"}

    # Extract function output as content
    if message.data:
        if message.data.output:
            result["content"] = _extract_content(message.data.output)
        elif message.data.payload:
            result["content"] = _extract_content(message.data.payload)
        else:
            result["content"] = ""
    else:
        result["content"] = ""

    # Add function name if available
    if message.name:
        result["name"] = message.name

    return result


def _parse_input_message(message: IntermediateStep) -> dict:
    """Parse user or system messages from LLM_START event."""
    if not message.data or not message.data.input:
        return {"role": "user", "content": ""}

    input_data = message.data.input

    # LlamaIndex typically stores messages as strings in the input
    if isinstance(input_data, str):
        # Check if it looks like a system message (heuristic)
        lower_input = input_data.lower()
        if (lower_input.startswith("system:") or "system prompt" in lower_input):
            return {"role": "system", "content": input_data}
        else:
            return {"role": "user", "content": input_data}

    # Handle list of messages (from EventPayload.MESSAGES)
    elif isinstance(input_data, list) and len(input_data) > 0:
        # Get the last message in the list
        last_msg = input_data[-1]

        # Try to parse the message
        if hasattr(last_msg, 'role') and hasattr(last_msg, 'content'):
            # LlamaIndex ChatMessage object
            role = str(last_msg.role).lower()
            # Map LlamaIndex roles to OpenAI roles
            role_mapping = {
                'user': 'user',
                'assistant': 'assistant',
                'system': 'system',
                'human': 'user',
                'ai': 'assistant',
                'chatbot': 'assistant'
            }
            role = role_mapping.get(role, 'user')
            return {"role": role, "content": str(last_msg.content)}
        else:
            # Convert to string if not a message object
            return {"role": "user", "content": str(last_msg)}

    # Handle dict messages
    elif isinstance(input_data, dict):
        return _parse_dict_message(input_data)

    else:
        return {"role": "user", "content": _extract_content(input_data)}


def _parse_dict_message(msg_dict: dict) -> dict:
    """Parse a dictionary-based message."""
    result = {}

    # Extract role
    result["role"] = msg_dict.get("role", "user")

    # Extract content
    if "content" in msg_dict:
        result["content"] = msg_dict["content"]
    elif "text" in msg_dict:
        result["content"] = msg_dict["text"]
    elif "blocks" in msg_dict:
        # Handle LlamaIndex block format
        blocks = msg_dict["blocks"]
        if isinstance(blocks, list):
            content_parts = []
            for block in blocks:
                if isinstance(block, dict) and "text" in block:
                    content_parts.append(block["text"])
                elif hasattr(block, 'text'):
                    content_parts.append(block.text)
            result["content"] = ''.join(content_parts)
        else:
            result["content"] = ""
    else:
        result["content"] = ""

    # Copy over optional fields
    optional_fields = ["tool_calls", "function_call", "name", "logprobs"]
    for field in optional_fields:
        if field in msg_dict:
            result[field] = msg_dict[field]

    return result
