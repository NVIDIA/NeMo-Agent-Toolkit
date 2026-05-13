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
"""Timeout middleware that enforces time limits on intercepted function calls."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from nat.builder.builder import Builder
from nat.middleware.dynamic.dynamic_function_middleware import DynamicFunctionMiddleware
from nat.middleware.middleware import CallNext
from nat.middleware.middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.timeout.timeout_middleware_config import TimeoutMiddlewareConfig

logger = logging.getLogger(__name__)


class TimeoutMiddleware(DynamicFunctionMiddleware):
    """Middleware that enforces configurable time limits on intercepted calls.

    Raises ``TimeoutError`` when execution exceeds the configured duration.
    When used in a middleware chain, the timeout covers everything downstream
    from its position — place it last to time only the target function.
    """

    def __init__(self, config: TimeoutMiddlewareConfig, builder: Builder) -> None:
        super().__init__(config=config, builder=builder)
        self._timeout_config: TimeoutMiddlewareConfig = config

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> Any:
        """Wrap the downstream call with an asyncio timeout.

        Args:
            args: Positional arguments for the function.
            call_next: Callable to invoke next middleware or target function.
            context: Static function metadata.
            kwargs: Keyword arguments for the function.

        Returns:
            The function output if it completes within the timeout.

        Raises:
            TimeoutError: If the downstream call exceeds the configured timeout.
        """
        timeout: float = self._timeout_config.timeout
        try:
            return await asyncio.wait_for(
                super().function_middleware_invoke(*args, call_next=call_next, context=context, **kwargs),
                timeout=timeout,
            )
        except TimeoutError:
            logger.error("Function '%s' exceeded timeout of %ss", context.name, timeout)
            msg: str = f"Execution exceeded the configured timeout of {timeout}s."
            if self._timeout_config.timeout_message:
                msg = f"{msg} {self._timeout_config.timeout_message}"
            raise TimeoutError(msg) from None

    async def function_middleware_stream(
        self,
        *args: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Wrap the downstream stream with an asyncio timeout.

        The timeout covers the total stream duration (time from the first
        chunk request to the final chunk), not individual inter-chunk gaps.

        Args:
            args: Positional arguments for the function.
            call_next: Callable to invoke next middleware or target stream.
            context: Static function metadata.
            kwargs: Keyword arguments for the function.

        Yields:
            Stream chunks from the downstream call.

        Raises:
            TimeoutError: If the full stream exceeds the configured timeout.
        """
        timeout: float = self._timeout_config.timeout
        try:
            async with asyncio.timeout(timeout):
                async for chunk in super().function_middleware_stream(*args,
                                                                      call_next=call_next,
                                                                      context=context,
                                                                      **kwargs):
                    yield chunk
        except TimeoutError:
            logger.error("Streaming function '%s' exceeded timeout of %ss", context.name, timeout)
            msg: str = f"Execution exceeded the configured timeout of {timeout}s."
            if self._timeout_config.timeout_message:
                msg = f"{msg} {self._timeout_config.timeout_message}"
            raise TimeoutError(msg) from None


__all__ = ["TimeoutMiddleware"]
