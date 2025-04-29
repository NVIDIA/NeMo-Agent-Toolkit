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

import contextvars
import dataclasses
import logging

from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepState
from aiq.data_models.invocation_node import InvocationNode
from aiq.utils.reactive.observable import OnComplete
from aiq.utils.reactive.observable import OnError
from aiq.utils.reactive.observable import OnNext
from aiq.utils.reactive.subscription import Subscription

logger = logging.getLogger(__name__)

_current_open_step_id = contextvars.ContextVar[str | None]("_current_open_step_id", default=None)


@dataclasses.dataclass
class OpenStep:
    step_id: str
    step_name: str
    step_type: str
    step_parent_id: str | None
    context: contextvars.Context
    token: contextvars.Token[str | None]


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

        parent_step_id = _current_open_step_id.get()

        if (payload.event_state == IntermediateStepState.START):

            token = _current_open_step_id.set(payload.UUID)

            self._outstanding_start_steps[payload.UUID] = OpenStep(step_id=payload.UUID,
                                                                   step_name=payload.name,
                                                                   step_type=payload.event_type,
                                                                   step_parent_id=parent_step_id,
                                                                   context=contextvars.copy_context(),
                                                                   token=token)

        elif (payload.event_state == IntermediateStepState.END):

            # Remove the current step from the outstanding steps
            open_step = self._outstanding_start_steps.pop(payload.UUID, None)

            if (open_step is None):
                logger.warning("Step id %s not found in outstanding start steps", payload.UUID)
                return

            # Restore the parent step ID directly instead of using a cross‑context token.
            if parent_step_id == payload.UUID:
                _current_open_step_id.set(open_step.step_parent_id)
            else:
                # Different context (e.g. thread‑pool); safely restore the parent ID **without**
                # trying to use a token that belongs to another Context.
                _current_open_step_id.set(open_step.step_parent_id)
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

            if (parent_step_id != payload.UUID):
                # Manually set the parent step ID. This happens when running on the thread pool
                parent_step_id = open_step.step_parent_id

        function_ancestry = InvocationNode(function_name=self._context_state.active_function.get().function_name,
                                           function_id=self._context_state.active_function.get().function_id,
                                           parent_id=parent_step_id,
                                           parent_name=self._context_state.active_function.get().parent_name)

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
