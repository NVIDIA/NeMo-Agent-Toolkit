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
"""Human-in-the-Loop (HITL) middleware for the NeMo Agent Toolkit."""

from __future__ import annotations

from abc import abstractmethod

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.data_models.interactive import InteractionResponse
from nat.middleware.dynamic.dynamic_function_middleware import DynamicFunctionMiddleware
from nat.middleware.hitl.hitl_middleware_config import HITLMiddlewareConfig
from nat.middleware.middleware import InvocationContext


class HITLMiddleware(DynamicFunctionMiddleware):
    """Human-in-the-Loop middleware.

    Intercepts function calls to present human prompts before and/or after execution.
    Subclasses must implement ``_on_pre_invoke_response`` and ``_on_post_invoke_response``
    to control what happens with the prompt response at each phase.
    """

    def __init__(self, config: HITLMiddlewareConfig, builder: Builder) -> None:
        """Initialize the HITL middleware.

        Args:
            config: HITL middleware configuration.
            builder: Workflow builder used for function discovery.
        """
        super().__init__(config=config, builder=builder)
        self._hitl_config: HITLMiddlewareConfig = config

    @abstractmethod
    async def _on_pre_invoke_response(
        self,
        response: InteractionResponse,
        context: InvocationContext,
    ) -> InvocationContext | None:
        """Handle the pre-invoke prompt response and return an invocation decision.

        Args:
            response: The human prompt response collected before the function is called.
            context: Invocation context containing the function arguments.

        Returns:
            ``None`` to proceed unchanged, an ``InvocationContext`` with updated
            ``modified_args`` or ``modified_kwargs`` to alter inputs, or an
            ``InvocationContext`` with ``action`` set to ``InvocationAction.SKIP``
            to bypass the function call entirely.
        """

    @abstractmethod
    async def _on_post_invoke_response(
        self,
        response: InteractionResponse,
        context: InvocationContext,
    ) -> InvocationContext | None:
        """Handle the post-invoke prompt response and return an invocation decision.

        Args:
            response: The human prompt response collected after the function returns.
            context: Invocation context with ``output`` set to the function result for a single call or chunk.

        Returns:
            ``None`` to return the output as-is, or an ``InvocationContext`` with
            ``output`` updated to replace the function's return value. For streaming
            calls, set ``output`` to ``None`` to suppress the chunk entirely.
        """

    async def pre_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Present a human prompt before the function is called.

        A no-op when ``pre_invoke_prompt`` is not configured.

        Args:
            context: Invocation context (output is None at this phase).

        Returns:
            The result of ``_on_pre_invoke_response``.
        """
        if not self._hitl_config.pre_invoke_prompt:
            return None

        response: InteractionResponse = await Context.get().user_interaction_manager.prompt_user_input(
            self._hitl_config.pre_invoke_prompt)
        return await self._on_pre_invoke_response(response, context)

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Present a human prompt after the function returns.

        A no-op when ``post_invoke_prompt`` is not configured.

        Args:
            context: Invocation context with the function output populated.

        Returns:
            The result of ``_on_post_invoke_response``.
        """
        if not self._hitl_config.post_invoke_prompt:
            return None

        response: InteractionResponse = await Context.get().user_interaction_manager.prompt_user_input(
            self._hitl_config.post_invoke_prompt)
        return await self._on_post_invoke_response(response, context)


__all__ = ["HITLMiddleware"]
