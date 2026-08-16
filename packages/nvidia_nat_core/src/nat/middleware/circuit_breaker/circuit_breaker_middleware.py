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
"""Circuit breaker middleware that prevents cascading failures by isolating failing functions."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from nat.builder.builder import Builder

from nat.middleware.circuit_breaker.circuit_breaker_middleware_config import CircuitBreakerMiddlewareConfig
from nat.middleware.dynamic.dynamic_function_middleware import DynamicFunctionMiddleware
from nat.middleware.middleware import CallNext
from nat.middleware.middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext

logger = logging.getLogger(__name__)


class CircuitBreakerState(StrEnum):
    """Possible states for the circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerMiddleware(DynamicFunctionMiddleware):
    """Middleware that implements the Circuit Breaker pattern for intercepted calls.

    Monitors execution failures and short-circuits calls when downstream services/functions fail
    repeatedly, transitioning through CLOSED, OPEN, and HALF_OPEN states.
    """

    def __init__(self, config: CircuitBreakerMiddlewareConfig, builder: Builder) -> None:
        super().__init__(config=config, builder=builder)
        self._cb_config: CircuitBreakerMiddlewareConfig = config
        self._state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_state_change: float = time.monotonic()
        self._half_open_probing: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        """Return the current circuit breaker state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Return the current consecutive failure count."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Return the current consecutive success count in HALF_OPEN state."""
        return self._success_count

    @property
    def last_state_change(self) -> float:
        """Return the monotonic timestamp of the last state change."""
        return self._last_state_change

    async def _before_invocation(self, context_name: str) -> bool:
        """Check state and determine whether call should proceed or be short-circuited."""
        async with self._lock:
            now = time.monotonic()
            if self._state == CircuitBreakerState.OPEN:
                if (now - self._last_state_change) >= self._cb_config.cooldown_period:
                    logger.info(
                        "Circuit breaker for '%s' cooldown period (%ss) elapsed. "
                        "Transitioning from OPEN to HALF_OPEN.",
                        context_name,
                        self._cb_config.cooldown_period,
                    )
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._last_state_change = now
                    self._success_count = 0
                    self._half_open_probing = True
                    return True
                return False

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_probing:
                    return False
                self._half_open_probing = True
                return True

            return True

    async def _after_invocation_success(self, context_name: str) -> None:
        """Handle successful execution state update."""
        async with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                self._failure_count = 0
            elif self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                self._half_open_probing = False
                if self._success_count >= self._cb_config.half_open_success_threshold:
                    logger.info(
                        "Circuit breaker for '%s' recovered (%d successes). "
                        "Transitioning from HALF_OPEN to CLOSED.",
                        context_name,
                        self._success_count,
                    )
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._last_state_change = time.monotonic()

    async def _after_invocation_failure(self, context_name: str) -> None:
        """Handle failed execution state update."""
        async with self._lock:
            now = time.monotonic()
            if self._state == CircuitBreakerState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._cb_config.failure_threshold:
                    logger.warning(
                        "Circuit breaker for '%s' tripped! "
                        "Transitioning from CLOSED to OPEN after %d consecutive failures.",
                        context_name,
                        self._failure_count,
                    )
                    self._state = CircuitBreakerState.OPEN
                    self._last_state_change = now
            elif self._state == CircuitBreakerState.HALF_OPEN:
                logger.warning(
                    "Circuit breaker for '%s' probe failed! Retripping from HALF_OPEN to OPEN.",
                    context_name,
                )

                self._state = CircuitBreakerState.OPEN
                self._last_state_change = now
                self._half_open_probing = False
                self._failure_count = self._cb_config.failure_threshold
                self._success_count = 0

    def _get_short_circuit_message(self, context_name: str) -> str:
        """Format the short-circuit message when OPEN or blocked in HALF_OPEN."""
        msg = f"Circuit breaker is OPEN for '{context_name}'. Tool is temporarily unavailable."
        if self._cb_config.circuit_breaker_message:
            msg = f"{msg} {self._cb_config.circuit_breaker_message}"
        return msg

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> Any:
        """Wrap downstream invocation with circuit breaker monitoring and short-circuiting."""
        should_execute = await self._before_invocation(context.name)
        if not should_execute:
            logger.warning("Circuit breaker for '%s' is active. Short-circuiting call.", context.name)
            return self._get_short_circuit_message(context.name)

        try:
            result = await super().function_middleware_invoke(*args, call_next=call_next, context=context, **kwargs)
        except Exception:
            await self._after_invocation_failure(context.name)
            raise
        else:
            await self._after_invocation_success(context.name)
            return result

    async def function_middleware_stream(
        self,
        *args: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Wrap downstream streaming call with circuit breaker monitoring and short-circuiting."""
        should_execute = await self._before_invocation(context.name)
        if not should_execute:
            logger.warning("Circuit breaker for '%s' is active. Short-circuiting stream.", context.name)
            yield self._get_short_circuit_message(context.name)
            return

        try:
            async for chunk in super().function_middleware_stream(*args, call_next=call_next, context=context,
                                                                  **kwargs):
                yield chunk
        except Exception:
            await self._after_invocation_failure(context.name)
            raise
        else:
            await self._after_invocation_success(context.name)


__all__ = ["CircuitBreakerMiddleware", "CircuitBreakerState"]
