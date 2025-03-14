# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from enum import Enum

from aiq.builder.context import AIQContext
from aiq.builder.context import AIQContextState
from aiq.builder.function import Function
from aiq.data_models.invocation_node import InvocationNode
from aiq.observability.async_otel_listener import AsyncOtelSpanListener
from aiq.utils.reactive.subject import Subject

logger = logging.getLogger(__name__)


class UserManagerBase:
    pass


class AIQRunnerState(Enum):
    UNINITIALIZED = 0
    INITIALIZED = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4


_T = typing.TypeVar("_T")


class AIQRunner:

    def __init__(self, input_message: typing.Any, entry_fn: Function, context_state: AIQContextState):
        """
        The AIQRunner class is used to run a workflow. It handles converting input and output data types and running the
        workflow with the specified concurrency.

        Parameters
        ----------
        input_message : typing.Any
            The input message to the workflow
        entry_fn : Function
            The entry function to the workflow
        context_state : AIQContextState
            The context state to use
        """

        if (entry_fn is None):
            raise ValueError("entry_fn cannot be None")

        self._entry_fn = entry_fn
        self._context_state = context_state
        self._context = AIQContext(self._context_state)

        self._state = AIQRunnerState.UNINITIALIZED

        self._input_message_token = None

        # Before we start, we need to convert the input message to the workflow input type
        self._input_message = input_message

        self._span_manager = AsyncOtelSpanListener(context_state=context_state)

    @property
    def context(self) -> AIQContext:
        return self._context

    def convert(self, value: typing.Any, to_type: type[_T]) -> _T:
        return self._entry_fn.convert(value, to_type)

    async def __aenter__(self):

        # Set the input message on the context
        self._input_message_token = self._context_state.input_message.set(self._input_message)

        # Create reactive event stream
        self._context_state.event_stream.set(Subject())
        self._context_state.active_function.set(InvocationNode(
            function_name="root",
            function_id="root",
        ))

        if (self._state == AIQRunnerState.UNINITIALIZED):
            self._state = AIQRunnerState.INITIALIZED
        else:
            raise ValueError("Cannot enter the context more than once")

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):

        if (self._input_message_token is None):
            raise ValueError("Cannot exit the context without entering it")

        self._context_state.input_message.reset(self._input_message_token)

        if (self._state not in (AIQRunnerState.COMPLETED, AIQRunnerState.FAILED)):
            raise ValueError("Cannot exit the context without completing the workflow")

    @typing.overload
    async def result(self) -> typing.Any:
        ...

    @typing.overload
    async def result(self, to_type: type[_T]) -> _T:
        ...

    async def result(self, to_type: type | None = None):

        if (self._state != AIQRunnerState.INITIALIZED):
            raise ValueError("Cannot run the workflow without entering the context")

        try:
            self._state = AIQRunnerState.RUNNING

            if (not self._entry_fn.has_single_output):
                raise ValueError("Workflow does not support single output")

            async with self._span_manager.start():
                # Run the workflow
                result = await self._entry_fn.ainvoke(self._input_message, to_type=to_type)

                # Close the intermediate stream
                self._context_state.event_stream.get().on_complete()

            self._state = AIQRunnerState.COMPLETED

            return result
        except Exception as e:
            logger.exception("Error running workflow: %s", e)
            self._context_state.event_stream.get().on_complete()
            self._state = AIQRunnerState.FAILED

            raise

    async def result_stream(self, to_type: type | None = None):

        if (self._state != AIQRunnerState.INITIALIZED):
            raise ValueError("Cannot run the workflow without entering the context")

        try:
            self._state = AIQRunnerState.RUNNING

            if (not self._entry_fn.has_streaming_output):
                raise ValueError("Workflow does not support streaming output")

            # Run the workflow
            async with self._span_manager.start():
                async for m in self._entry_fn.astream(self._input_message, to_type=to_type):
                    yield m

                self._state = AIQRunnerState.COMPLETED

                # Close the intermediate stream
                self._context_state.event_stream.get().on_complete()

        except Exception as e:
            logger.exception("Error running workflow: %s", e)
            self._context_state.event_stream.get().on_complete()
            self._state = AIQRunnerState.FAILED

            raise
