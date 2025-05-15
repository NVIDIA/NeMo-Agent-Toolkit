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

import dataclasses
import logging
import typing

from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepState
from aiq.data_models.invocation_node import InvocationNode
from aiq.utils.reactive.observable import OnComplete
from aiq.utils.reactive.observable import OnError
from aiq.utils.reactive.observable import OnNext
from aiq.utils.reactive.subscription import Subscription

if typing.TYPE_CHECKING:
    from aiq.builder.context import AIQContextState

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OpenStep:
    step_id: str
    step_name: str
    step_type: str
    step_parent_id: str | None


class IntermediateStepManager:
    """
    Manages updates to the AIQ Toolkit Event Stream for intermediate steps
    """

    def __init__(self, context_state: "AIQContextState"):  # noqa: F821
        self._context_state = context_state

        self._outstanding_start_steps: dict[str, OpenStep] = {}

    def push_intermediate_step(self, payload: IntermediateStepPayload) -> None:
        """
        Pushes an intermediate step to the AIQ Toolkit Event Stream
        """

        if not isinstance(payload, IntermediateStepPayload):
            raise TypeError(f"Payload must be of type IntermediateStepPayload, not {type(payload)}")

        active_span_id_stack = self._context_state.active_span_id_stack.get()

        if (payload.event_state == IntermediateStepState.START):

            parent_step_id = active_span_id_stack[-1]

            # Note, this must not mutate the active_span_id_stack in place
            active_span_id_stack = active_span_id_stack + [payload.UUID]
            self._context_state.active_span_id_stack.set(active_span_id_stack)

            self._outstanding_start_steps[payload.UUID] = OpenStep(step_id=payload.UUID,
                                                                   step_name=payload.name or payload.UUID,
                                                                   step_type=payload.event_type,
                                                                   step_parent_id=parent_step_id)

        elif (payload.event_state == IntermediateStepState.END):

            # Remove the current step from the outstanding steps
            open_step = self._outstanding_start_steps.pop(payload.UUID, None)

            if (open_step is None):
                logger.warning("Step id %s not found in outstanding start steps", payload.UUID)
                return

            # Remove the current step from the active span id stack. Look for the step id in the stack and remove it to
            # correct errors
            current_step_index = active_span_id_stack.index(payload.UUID)

            if (current_step_index is not None):
                if (current_step_index != len(active_span_id_stack) - 1):
                    logger.warning(
                        "Step id %s not the last step in the stack. "
                        "Removing it from the stack but this is likely an error",
                        payload.UUID)

                active_span_id_stack = active_span_id_stack[:current_step_index]
                self._context_state.active_span_id_stack.set(active_span_id_stack)

            parent_step_id = open_step.step_parent_id

        elif (payload.event_state == IntermediateStepState.CHUNK):

            # Get the current step from the outstanding steps
            open_step = self._outstanding_start_steps.get(payload.UUID, None)

            # Generate a warning if the parent step id is not set to the current step id
            if (open_step is None):
                logger.warning(
                    "Created a chunk for step %s, but no matching start step was found. "
                    "Chunks must be created with the same ID as the start step.",
                    payload.UUID)
                return

            parent_step_id = open_step.step_parent_id

        active_function = self._context_state.active_function.get()

        function_ancestry = InvocationNode(function_name=active_function.function_name,
                                           function_id=active_function.function_id,
                                           parent_id=parent_step_id,
                                           parent_name=active_function.parent_name)

        intermediate_step = IntermediateStep(function_ancestry=function_ancestry, payload=payload)

        self._context_state.event_stream.get().on_next(intermediate_step)

    def subscribe(self,
                  on_next: OnNext[IntermediateStep],
                  on_error: OnError = None,
                  on_complete: OnComplete = None) -> Subscription:
        """
        Subscribes to the AIQ Toolkit Event Stream for intermediate steps
        """

        return self._context_state.event_stream.get().subscribe(on_next, on_error, on_complete)
