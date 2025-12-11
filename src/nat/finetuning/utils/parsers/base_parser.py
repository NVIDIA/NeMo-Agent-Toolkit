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

import logging

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepState
from nat.data_models.intermediate_step import IntermediateStepType
from nat.finetuning.utils.parsers import adk_parser
from nat.finetuning.utils.parsers import langchain_parser
from nat.finetuning.utils.parsers import llama_index_parser

logger = logging.getLogger(__name__)


def parse_to_openai_messages(steps: list[IntermediateStep]) -> list[dict]:
    """
    Convert IntermediateStep objects to OpenAI-compatible messages.

    Args:
        steps: List of IntermediateStep objects representing the conversation.

    Returns:
        List of dictionaries formatted for OpenAI API consumption.

    Raises:
        ValueError: If unsupported type or invalid sequence.
    """

    messages = []

    # Track the last event type to handle special cases
    last_event_type = None
    message_content_hashes = set()
    for message in steps:
        # Skip LLM_START events that come after TOOL_END events
        # These represent the assistant processing tool results internally
        if message.event_type not in [
                IntermediateStepType.LLM_END, IntermediateStepType.LLM_START, IntermediateStepType.TOOL_END
        ]:
            continue

        if (message.event_type == IntermediateStepType.LLM_START and last_event_type == IntermediateStepType.TOOL_END):
            continue

        # Skip streaming chunks
        if message.event_state not in [IntermediateStepState.START, IntermediateStepState.END]:
            continue

        # Parse the message based on framework
        if message.framework == LLMFrameworkEnum.LANGCHAIN:
            parsed_msg = langchain_parser.parse_to_openai_message(message=message)
        elif message.framework == LLMFrameworkEnum.LLAMA_INDEX:
            parsed_msg = llama_index_parser.parse_to_openai_message(message=message)
        elif message.framework == LLMFrameworkEnum.ADK:
            parsed_msg = adk_parser.parse_to_openai_message(message=message)
        else:
            if message.framework is not None:
                logger.warning(f"Unsupported framework: {message.framework} for message {message}")
            continue

        # Add the parsed message
        if message.event_type == IntermediateStepType.LLM_START:
            # LLM_START messages may contain multiple messages (e.g., tools called by the LLM)
            # We deduplicate previously seen messages if sharing message history to the model
            if isinstance(parsed_msg, list):
                for msg in parsed_msg:
                    content_hash = hash(msg["role"] + ": " + msg["content"])
                    if content_hash not in message_content_hashes:
                        messages.append(msg)
                        message_content_hashes.add(content_hash)
            else:
                content_hash = hash(parsed_msg["role"] + ": " + parsed_msg["content"])
                messages.append(parsed_msg)
                message_content_hashes.add(content_hash)
        else:
            assert not isinstance(parsed_msg, list), "TOOL_END or LLM_END should not produce multiple messages"
            content_hash = hash(parsed_msg["role"] + ": " + parsed_msg["content"])
            message_content_hashes.add(content_hash)
            messages.append(parsed_msg)

        last_event_type = message.event_type

    # Validate and fix the message sequence
    try:
        messages = _validate_message_sequence(messages)
    except Exception as _:
        logger.exception("Error validating message sequence.")
        raise

    return messages


def _validate_message_sequence(messages: list[dict]) -> list[dict]:
    """
    Validate and fix the message sequence to follow OpenAI's expected format.

    Rules:

    - System messages can only appear at the beginning
    - After system messages, must alternate between user/tool and assistant
    - Cannot have consecutive user messages or consecutive assistant messages
    - If first non-system messages are not user messages, they will be
      concatenated into a single user message (with a warning)

    Args:
        messages: List of parsed OpenAI messages

    Returns:
        list[dict]: The validated (and potentially fixed) message list

    Raises:
        ValueError: If the message sequence is invalid.
    """
    if not messages:
        return messages

    # Check system messages are only at the beginning
    found_non_system = False
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            if found_non_system:
                raise ValueError(f"System message found at position {i} after "
                                 "non-system messages. System messages must only "
                                 "appear at the beginning.")
        else:
            found_non_system = True

    # Find first non-system message
    first_non_system_idx = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            first_non_system_idx = i
            break

    # Fix non-user messages at the start of trajectory
    # Collect all non-system messages before the first assistant message
    if first_non_system_idx < len(messages):
        # Find the first assistant message
        first_assistant_idx = None
        for i in range(first_non_system_idx, len(messages)):
            if messages[i].get("role") == "assistant":
                first_assistant_idx = i
                break

        # Check if we need to fix the start of the trajectory
        if first_assistant_idx is not None:
            messages_to_concatenate = []
            for i in range(first_non_system_idx, first_assistant_idx):
                msg = messages[i]
                role = msg.get("role")
                if role != "user":
                    # This message should be concatenated
                    messages_to_concatenate.append((i, msg))

            if messages_to_concatenate:
                # Collect all content from non-user messages at the start
                content_parts = []
                indices_to_remove = []

                for i in range(first_non_system_idx, first_assistant_idx):
                    msg = messages[i]
                    role = msg.get("role")
                    content = msg.get("content", "")

                    if role not in ["user"]:
                        # Non-user message that needs to be consolidated
                        if content:
                            content_parts.append(f"[{role.upper()}]: {content}")
                        indices_to_remove.append(i)
                    else:
                        # User message - include its content
                        if content:
                            content_parts.append(content)
                        indices_to_remove.append(i)

                # Create a single user message with concatenated content
                if content_parts:
                    concatenated_content = "\n\n".join(content_parts)
                    new_user_message = {"role": "user", "content": concatenated_content}

                    # Log warning about the modification
                    logger.warning(
                        "Trajectory had %d non-user messages at the start "
                        "before the first assistant message. "
                        "Concatenated these into a single user message. "
                        "Original roles: %s",
                        len(messages_to_concatenate), [msg.get("role") for _, msg in messages_to_concatenate])

                    # Remove the old messages and insert the new one
                    # Remove in reverse order to maintain indices
                    for idx in reversed(indices_to_remove):
                        messages.pop(idx)

                    # Insert the new user message
                    messages.insert(first_non_system_idx, new_user_message)

    # Recalculate first_non_system_idx after potential modifications
    first_non_system_idx = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            first_non_system_idx = i
            break

    # Validate alternating pattern after system messages
    if first_non_system_idx < len(messages):
        prev_role = None
        for i in range(first_non_system_idx, len(messages)):
            role = messages[i].get("role")

            if prev_role:
                # Check for invalid consecutive roles
                if role == "user" and prev_role == "user":
                    raise ValueError(f"Consecutive user messages at positions {i-1} "
                                     f"and {i}. User messages must be followed by "
                                     "assistant messages.")
                elif role == "assistant" and prev_role == "assistant":
                    raise ValueError(f"Consecutive assistant messages at positions "
                                     f"{i-1} and {i}. Assistant messages must be "
                                     "followed by user or tool messages.")

            prev_role = role

    return messages
