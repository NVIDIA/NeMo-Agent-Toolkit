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

import typing
import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

from aiq.builder.intermediate_step_manager import IntermediateStepManager
from aiq.builder.user_interaction_manager import AIQUserInteractionManager
from aiq.data_models.interactive import HumanResponse
from aiq.data_models.interactive import InteractionPrompt
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import StreamEventData
from aiq.data_models.invocation_node import InvocationNode
from aiq.utils.reactive.subject import Subject


class Singleton(type):

    def __init__(cls, name, bases, dict):  # pylint: disable=W0622
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance


class ActiveFunctionContextManager:

    def __init__(self):
        self._output: typing.Any | None = None

    @property
    def output(self) -> typing.Any | None:
        return self._output

    def set_output(self, output: typing.Any):
        self._output = output


class AIQContextState(metaclass=Singleton):

    def __init__(self):
        self.input_message: ContextVar[typing.Any] = ContextVar("input_message", default=None)
        self.user_manager: ContextVar[typing.Any] = ContextVar("user_manager", default=None)
        self.event_stream: ContextVar[Subject[IntermediateStep] | None] = ContextVar("event_stream", default=Subject())
        self.active_function: ContextVar[InvocationNode] = ContextVar("active_function",
                                                                      default=InvocationNode(function_id="root",
                                                                                             function_name="root"))
        # Default is a lambda no-op which returns NoneType
        self.user_input_callback: ContextVar[Callable[[InteractionPrompt], Awaitable[HumanResponse | None]]
                                             | None] = ContextVar(
                                                 "user_input_callback",
                                                 default=AIQUserInteractionManager.default_callback_handler)

    @staticmethod
    def get() -> "AIQContextState":
        return AIQContextState()


class AIQContext:

    def __init__(self, context: AIQContextState):
        self._context_state = context

    @property
    def input_message(self):
        """
            Retrieves the input message from the context state.

            The input_message property is used to access the message stored in the
            context state. This property returns the message as it is currently
            maintained in the context.

            Returns:
                str: The input message retrieved from the context state.
        """
        return self._context_state.input_message.get()

    @property
    def user_manager(self):
        """
        Retrieves the user manager instance from the current context state.

        This property provides access to the user manager through the context
        state, allowing interaction with user management functionalities.

        Returns:
            UserManager: The instance of the user manager retrieved from the
                context state.
        """
        return self._context_state.user_manager.get()

    @property
    def user_interaction_manager(self) -> AIQUserInteractionManager:
        """
        Return an instance of AIQUserInteractionManager that uses
        the current context's user_input_callback.
        """
        return AIQUserInteractionManager(self._context_state)

    @property
    def intermediate_step_manager(self) -> IntermediateStepManager:
        """
        Retrieves the intermediate step manager instance from the current context state.

        This property provides access to the intermediate step manager through the context
        state, allowing interaction with intermediate step management functionalities.

        Returns:
            IntermediateStepManager: The instance of the intermediate step manager retrieved
                from the context state.
        """
        return IntermediateStepManager(self._context_state)

    @contextmanager
    def push_active_function(self, function_name: str, input_data: typing.Any | None):
        """
        Set the 'active_function' in context, push an invocation node,
        AND create an OTel child span for that function call.
        """
        parent_function_node = self._context_state.active_function.get()
        current_function_id = str(uuid.uuid4())
        current_function_node = InvocationNode(function_id=current_function_id,
                                               function_name=function_name,
                                               parent_id=parent_function_node.function_id,
                                               parent_name=parent_function_node.function_name)

        # 1) Set the active function in the contextvar
        fn_token = self._context_state.active_function.set(current_function_node)

        # 2) Optionally record function start as an intermediate step
        step_manager = self.intermediate_step_manager
        step_manager.push_intermediate_step(
            IntermediateStepPayload(UUID=current_function_id,
                                    event_type=IntermediateStepType.FUNCTION_START,
                                    name=function_name,
                                    data=StreamEventData(input=input_data)))

        manager = ActiveFunctionContextManager()

        try:
            yield manager  # run the function body
        finally:
            # 3) Record function end

            data = StreamEventData(input=input_data, output=manager.output)

            step_manager.push_intermediate_step(
                IntermediateStepPayload(UUID=current_function_id,
                                        event_type=IntermediateStepType.FUNCTION_END,
                                        name=function_name,
                                        data=data))

            # 4) Unset the function contextvar
            self._context_state.active_function.reset(fn_token)

    @property
    def active_function(self) -> InvocationNode:
        """
        Retrieves the active function from the context state.

        This property is used to access the active function stored in the context
        state. The active function is the function that is currently being executed.
        """
        return self._context_state.active_function.get()

    @staticmethod
    def get() -> "AIQContext":
        """
        Static method to retrieve the current AIQContext instance.

        This method creates and returns an instance of the AIQContext class
        by obtaining the current state from the AIQContextState.

        Returns:
            AIQContext: The created AIQContext instance.
        """
        return AIQContext(AIQContextState.get())
