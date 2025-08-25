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

from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.function import Function
from nat.data_models.invocation_node import InvocationNode
from nat.observability.exporter_manager import ExporterManager
from nat.utils.reactive.subject import Subject

logger = logging.getLogger(__name__)


class UserManagerBase:
    pass


class RunnerState(Enum):
    UNINITIALIZED = 0
    INITIALIZED = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4


_T = typing.TypeVar("_T")


class Runner:

    def __init__(self,
                 input_message: typing.Any,
                 entry_fn: Function,
                 context_state: ContextState,
                 exporter_manager: ExporterManager):
        """
        The Runner class is used to run a workflow. It handles converting input and output data types and running the
        workflow with the specified concurrency.

        Parameters
        ----------
        input_message : typing.Any
            The input message to the workflow
        entry_fn : Function
            The entry function to the workflow
        context_state : ContextState
            The context state to use
        exporter_manager : ExporterManager
            The exporter manager to use
        """

        if (entry_fn is None):
            raise ValueError("entry_fn cannot be None")

        self._entry_fn = entry_fn
        self._context_state = context_state
        self._context = Context(self._context_state)

        self._state = RunnerState.UNINITIALIZED

        self._input_message_token = None

        # Before we start, we need to convert the input message to the workflow input type
        self._input_message = input_message

        self._exporter_manager = exporter_manager

    @property
    def context(self) -> Context:
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

        if (self._state == RunnerState.UNINITIALIZED):
            self._state = RunnerState.INITIALIZED
        else:
            raise ValueError("Cannot enter the context more than once")

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):

        if (self._input_message_token is None):
            raise ValueError("Cannot exit the context without entering it")

        self._context_state.input_message.reset(self._input_message_token)

        if (self._state not in (RunnerState.COMPLETED, RunnerState.FAILED)):
            raise ValueError("Cannot exit the context without completing the workflow")

    @typing.overload
    async def result(self) -> typing.Any:
        ...

    @typing.overload
    async def result(self, to_type: type[_T]) -> _T:
        ...

    async def result(self, to_type: type | None = None):

        if (self._state != RunnerState.INITIALIZED):
            raise ValueError("Cannot run the workflow without entering the context")

        try:
            self._state = RunnerState.RUNNING

            if (not self._entry_fn.has_single_output):
                raise ValueError("Workflow does not support single output")

            async with self._exporter_manager.start(context_state=self._context_state):
                # Run the workflow
                result = await self._entry_fn.ainvoke(self._input_message, to_type=to_type)

                # Close the intermediate stream
                event_stream = self._context_state.event_stream.get()
                if event_stream:
                    event_stream.on_complete()

            self._state = RunnerState.COMPLETED

            return result
        except Exception as e:
            logger.exception("Error running workflow: %s", e)
            event_stream = self._context_state.event_stream.get()
            if event_stream:
                event_stream.on_complete()
            self._state = RunnerState.FAILED

            raise

    async def result_stream(self, to_type: type | None = None):

        if (self._state != RunnerState.INITIALIZED):
            raise ValueError("Cannot run the workflow without entering the context")

        try:
            self._state = RunnerState.RUNNING

            if (not self._entry_fn.has_streaming_output):
                raise ValueError("Workflow does not support streaming output")

            # Run the workflow
            async with self._exporter_manager.start(context_state=self._context_state):
                async for m in self._entry_fn.astream(self._input_message, to_type=to_type):
                    yield m

                self._state = RunnerState.COMPLETED

                # Close the intermediate stream
                event_stream = self._context_state.event_stream.get()
                if event_stream:
                    event_stream.on_complete()

        except Exception as e:
            logger.exception("Error running workflow: %s", e)
            event_stream = self._context_state.event_stream.get()
            if event_stream:
                event_stream.on_complete()
            self._state = RunnerState.FAILED

            raise


# Compatibility aliases with previous releases
AIQRunnerState = RunnerState
AIQRunner = Runner
