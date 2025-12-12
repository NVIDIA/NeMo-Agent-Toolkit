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

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.finetuning.utils.parsers.common import extract_content
from nat.finetuning.utils.parsers.common import parse_generic_message

logger = logging.getLogger(__name__)

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


def _parse_input_message(message: IntermediateStep) -> dict | list[dict]:
    """Parse user or system messages from LLM_START event."""

    messages = message.data.payload

    if len(messages) == 0:
        return {"role": "user", "content": ""}
    elif len(messages) == 1:
        if not isinstance(messages[0], dict):
            return {"role": "user", "content": str(messages[0])}

        if not ("role" in messages[0] and "content" in messages[0]):
            return {"role": "user", "content": json.dumps(messages[0])}

        return messages[0]
    else:
        parsed_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                parsed_messages.append({"role": "user", "content": str(msg)})
            elif not ("role" in msg and "content" in msg):
                parsed_messages.append({"role": "user", "content": json.dumps(msg)})
            else:
                parsed_messages.append(msg)
        return parsed_messages


def _parse_assistant_message(message: IntermediateStep) -> dict:
    """Parse an assistant message from LLM_END event."""
    result = {"role": "assistant"}

    # Get the ChatResponse from payload if available
    try:
        if message.data and message.data.payload:
            pass
            payload = message.data.payload
            payload_message = getattr(payload, 'message', None)

            if "logprobs" in payload:
                result["logprobs"] = payload["logprobs"]
            else:
                logger.warning("No logprobs found in LLM_END message payload.")

            if "content" in payload_message and payload_message["content"] is not None:
                result["content"] = _extract_content(payload_message["content"])
            else:
                result["content"] = ""

            if "tool_calls" in payload_message and payload_message["tool_calls"] is not None:
                result["tool_calls"] = payload_message["tool_calls"]

        else:
            logger.warning("No payload found in LLM_END message data.")
            return {"role": "assistant", "content": ""}
    except Exception as _:
        logger.exception("Error parsing assistant message from LLM_END event.")
        return {"role": "assistant", "content": ""}

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
