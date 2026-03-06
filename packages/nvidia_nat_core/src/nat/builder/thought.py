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

import logging
import typing
import uuid

from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData

if typing.TYPE_CHECKING:
    from nat.builder.context import Context

logger = logging.getLogger(__name__)


def emit_thought(context: "Context", thought_text: str, name: str | None = None) -> str:
    """Emit a complete custom thought that appears in the UI's thought process display.

    This is useful for showing discrete progress steps or status updates during
    function/tool execution.

    Args:
        context: The NAT context object (obtain via Context.get())
        thought_text: The thought text to display in the UI
        name: Optional name for the thought (defaults to "custom_thought")

    Returns:
        The UUID of the emitted thought
    """
    thought_uuid = str(uuid.uuid4())
    thought_name = name or "custom_thought"

    # Emit START event
    context.intermediate_step_manager.push_intermediate_step(
        IntermediateStepPayload(UUID=thought_uuid,
                                event_type=IntermediateStepType.SPAN_START,
                                name=thought_name,
                                data=StreamEventData(input=None),
                                metadata={"thought_text": thought_text}))

    # Immediately emit END event (complete thought)
    context.intermediate_step_manager.push_intermediate_step(
        IntermediateStepPayload(UUID=thought_uuid,
                                event_type=IntermediateStepType.SPAN_END,
                                name=thought_name,
                                data=StreamEventData(output=None),
                                metadata={"thought_text": thought_text}))

    return thought_uuid


def emit_thought_start(context: "Context", thought_text: str, name: str | None = None) -> str:
    """Start emitting a streaming thought that can be updated with chunks.

    Use this for long-running operations where you want to show progressive updates.
    Follow up with emit_thought_chunk() for updates and emit_thought_end() to complete.

    Args:
        context: The NAT context object (obtain via Context.get())
        thought_text: The initial thought text to display
        name: Optional name for the thought (defaults to "custom_thought")

    Returns:
        The UUID of the started thought (use this for chunks and end)
    """
    thought_uuid = str(uuid.uuid4())
    thought_name = name or "custom_thought"

    context.intermediate_step_manager.push_intermediate_step(
        IntermediateStepPayload(UUID=thought_uuid,
                                event_type=IntermediateStepType.SPAN_START,
                                name=thought_name,
                                data=StreamEventData(input=None),
                                metadata={"thought_text": thought_text}))

    return thought_uuid


def emit_thought_chunk(context: "Context", thought_uuid: str, thought_text: str) -> None:
    """Emit an update to a streaming thought started with emit_thought_start().

    This updates the thought text in the UI, useful for showing progress updates.

    Args:
        context: The NAT context object (obtain via Context.get())
        thought_uuid: The UUID returned from emit_thought_start()
        thought_text: The updated thought text to display
    """
    context.intermediate_step_manager.push_intermediate_step(
        IntermediateStepPayload(UUID=thought_uuid,
                                event_type=IntermediateStepType.SPAN_CHUNK,
                                name="custom_thought",
                                data=StreamEventData(chunk=thought_text),
                                metadata={"thought_text": thought_text}))


def emit_thought_end(context: "Context", thought_uuid: str, thought_text: str | None = None) -> None:
    """Complete a streaming thought started with emit_thought_start().

    Args:
        context: The NAT context object (obtain via Context.get())
        thought_uuid: The UUID returned from emit_thought_start()
        thought_text: Optional final thought text (if None, keeps the last chunk text)
    """
    context.intermediate_step_manager.push_intermediate_step(
        IntermediateStepPayload(UUID=thought_uuid,
                                event_type=IntermediateStepType.SPAN_END,
                                name="custom_thought",
                                data=StreamEventData(output=None),
                                metadata={"thought_text": thought_text} if thought_text else {}))
