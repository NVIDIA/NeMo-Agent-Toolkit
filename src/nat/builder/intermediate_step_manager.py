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
import weakref
from typing import ClassVar

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepState
from nat.utils.reactive.observable import OnComplete
from nat.utils.reactive.observable import OnError
from nat.utils.reactive.observable import OnNext
from nat.utils.reactive.subscription import Subscription

if typing.TYPE_CHECKING:
    from nat.builder.context import ContextState

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OpenStep:
    step_id: str
    step_name: str
    step_type: str
    step_parent_id: str
    prev_stack: list[str]
    active_stack: list[str]


class IntermediateStepManager:
    """
    Manages updates to the NAT Event Stream for intermediate steps
    """

    # Class-level tracking for debugging and monitoring
    _instance_count: ClassVar[int] = 0
    _active_instances: ClassVar[set[weakref.ref]] = set()

    def __init__(self, context_state: "ContextState"):  # noqa: F821
        self._context_state = context_state

        self._outstanding_start_steps: dict[str, OpenStep] = {}

        # Track instance creation
        IntermediateStepManager._instance_count += 1
        IntermediateStepManager._active_instances.add(weakref.ref(self, self._cleanup_instance_tracking))

    def push_intermediate_step(self, payload: IntermediateStepPayload) -> None:
        """
        Pushes an intermediate step to the NAT Event Stream
        """

        if not isinstance(payload, IntermediateStepPayload):
            raise TypeError(f"Payload must be of type IntermediateStepPayload, not {type(payload)}")

        active_span_id_stack = self._context_state.active_span_id_stack.get()

        if (payload.event_state == IntermediateStepState.START):

            prev_stack = active_span_id_stack

            parent_step_id = active_span_id_stack[-1]

            # Note, this must not mutate the active_span_id_stack in place
            active_span_id_stack = active_span_id_stack + [payload.UUID]
            self._context_state.active_span_id_stack.set(active_span_id_stack)

            self._outstanding_start_steps[payload.UUID] = OpenStep(step_id=payload.UUID,
                                                                   step_name=payload.name or payload.UUID,
                                                                   step_type=payload.event_type,
                                                                   step_parent_id=parent_step_id,
                                                                   prev_stack=prev_stack,
                                                                   active_stack=active_span_id_stack)

            logger.debug("Pushed start step %s, name %s, type %s, parent %s, stack id %s",
                         payload.UUID,
                         payload.name,
                         payload.event_type,
                         parent_step_id,
                         id(active_span_id_stack))

        elif (payload.event_state == IntermediateStepState.END):

            # Remove the current step from the outstanding steps
            open_step = self._outstanding_start_steps.pop(payload.UUID, None)

            if (open_step is None):
                logger.warning(
                    "Step id %s not found in outstanding start steps. "
                    "This may occur if the step was started in a different context or already completed.",
                    payload.UUID)
                return

            parent_step_id = open_step.step_parent_id

            # Get the current and previous active span id stack.
            curr_stack = open_step.active_stack
            prev_stack = open_step.prev_stack

            # To restore the stack, we need to handle two scenarios:
            # 1. This function is called from a coroutine. In this case, the context variable will be the same as the
            #    one used in START. So we can just set the context variable to the previous stack.
            # 2. This function is called from a task. In this case, the context variable will be separate from the one
            #    used in START so calling set() will have no effect. However, we still have a reference to the list used
            #    in START. So we update the reference to be equal to the old one.. So we need to update the current
            #    reference stack to be equal to the previous stack.

            # Scenario 1: Restore the previous active span id stack in case we are in a coroutine. Dont use reset here
            # since we can be in different contexts
            self._context_state.active_span_id_stack.set(prev_stack)

            pop_count = 0

            # Scenario 2: Remove all steps from the current stack until we reach the parent step id to make it equal to
            # the previous stack. In the coroutine case, this will not have any effect.
            while (curr_stack[-1] != parent_step_id):
                curr_stack.pop()
                pop_count += 1

            if (pop_count != 1):
                logger.warning(
                    "Step id %s not the last step in the stack. "
                    "Removing it from the stack but this is likely an error",
                    payload.UUID)

            # Verify that the stack is now equal to the previous stack
            if (curr_stack != prev_stack):
                logger.warning("Current span ID stack is not equal to the previous stack. "
                               "This is likely an error. Report this to the NeMo Agent toolkit team.")

            logger.debug("Popped end step %s, name %s, type %s, parent %s, stack id %s",
                         payload.UUID,
                         payload.name,
                         payload.event_type,
                         parent_step_id,
                         id(curr_stack))

        elif (payload.event_state == IntermediateStepState.CHUNK):

            # Get the current step from the outstanding steps
            open_step = self._outstanding_start_steps.get(payload.UUID, None)

            # Generate a warning if the parent step id is not set to the current step id
            if (open_step is None):
                logger.warning(
                    "Created a chunk for step %s, but no matching start step was found. "
                    "Chunks must be created with the same ID as the start step. "
                    "This may occur if the step was started in a different context.",
                    payload.UUID)
                return

            parent_step_id = open_step.step_parent_id
        else:
            assert False, "Invalid event state"

        active_function = self._context_state.active_function.get()

        intermediate_step = IntermediateStep(parent_id=parent_step_id,
                                             function_ancestry=active_function,
                                             payload=payload)

        self._context_state.event_stream.get().on_next(intermediate_step)

    def subscribe(self,
                  on_next: OnNext[IntermediateStep],
                  on_error: OnError = None,
                  on_complete: OnComplete = None) -> Subscription:
        """
        Subscribes to the NAT Event Stream for intermediate steps
        """

        return self._context_state.event_stream.get().subscribe(on_next, on_error, on_complete)

    @classmethod
    def _cleanup_instance_tracking(cls, ref: weakref.ref) -> None:
        """Cleanup callback for weakref when instance is garbage collected."""
        cls._active_instances.discard(ref)

    @classmethod
    def get_active_instance_count(cls) -> int:
        """Get the number of active IntermediateStepManager instances.

        Returns:
            int: Number of active instances (cleaned up automatically via weakref)
        """
        return len(cls._active_instances)

    def get_outstanding_step_count(self) -> int:
        """Get the number of outstanding (started but not ended) steps.

        Returns:
            int: Number of steps that have been started but not yet ended
        """
        return len(self._outstanding_start_steps)
