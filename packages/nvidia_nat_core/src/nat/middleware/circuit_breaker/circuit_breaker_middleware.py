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
import contextlib
import dataclasses
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


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a call is short-circuited because the circuit breaker is ``OPEN``."""


class CircuitBreakerState(StrEnum):
    """Possible states for the circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclasses.dataclass
class _ToolState:
    """Internal state record tracking failures, successes, and probing status per target."""

    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_state_change: float = dataclasses.field(default_factory=time.monotonic)
    half_open_probing: bool = False


class CircuitBreakerMiddleware(DynamicFunctionMiddleware):
    """Middleware implementing the Circuit Breaker pattern with per-target isolation.

    Monitors execution failures and short-circuits calls when downstream functions fail
    repeatedly, transitioning through ``CLOSED``, ``OPEN``, and ``HALF_OPEN`` states per target.
    """

    def __init__(self, config: CircuitBreakerMiddlewareConfig, builder: Builder) -> None:
        """Initialize CircuitBreakerMiddleware.

        Args:
            config: Circuit breaker configuration parameters.
            builder: Workflow builder instance.
        """
        super().__init__(config=config, builder=builder)
        self._cb_config: CircuitBreakerMiddlewareConfig = config
        self._states: dict[str, _ToolState] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    def _get_tool_state(self, target: str) -> _ToolState:
        """Get or initialize state record for a given target name under lock.

        Args:
            target: Identifier for the target function or tool.

        Returns:
            ``_ToolState``: State record for the target.
        """
        if target not in self._states:
            self._states[target] = _ToolState(last_state_change=time.monotonic())
        return self._states[target]

    async def get_state(self, target: str) -> CircuitBreakerState:
        """Return the current circuit breaker state for a specific target.

        Args:
            target: Identifier for the target function or tool.

        Returns:
            CircuitBreakerState: Current state of the target.
        """
        async with self._lock:
            if target in self._states:
                return self._states[target].state
            return CircuitBreakerState.CLOSED

    async def get_failure_count(self, target: str) -> int:
        """Return the consecutive failure count for a specific target.

        Args:
            target: Identifier for the target function or tool.

        Returns:
            int: Number of consecutive failures.
        """
        async with self._lock:
            if target in self._states:
                return self._states[target].failure_count
            return 0

    async def get_success_count(self, target: str) -> int:
        """Return consecutive successes in ``HALF_OPEN`` state towards recovery for a target.

        Args:
            target: Identifier for the target function or tool.

        Returns:
            int: Number of consecutive probe successes.
        """
        async with self._lock:
            if target in self._states:
                return self._states[target].success_count
            return 0

    async def get_last_state_change(self, target: str) -> float:
        """Return the monotonic timestamp of the last state change for a specific target.

        Args:
            target: Identifier for the target function or tool.

        Returns:
            float: Timestamp in seconds.
        """
        async with self._lock:
            if target in self._states:
                return self._states[target].last_state_change
            return 0.0

    async def is_half_open_probing(self, target: str) -> bool:
        """Return whether a probe execution is actively in-flight for a target in ``HALF_OPEN`` state.

        Args:
            target: Identifier for the target function or tool.

        Returns:
            bool: True if a probe call is in-flight in ``HALF_OPEN`` state.
        """
        async with self._lock:
            if target in self._states:
                return self._states[target].half_open_probing
            return False

    def _get_short_circuit_message(self, target: str) -> str:
        """Format the short-circuit message when ``OPEN`` or busy probing.

        Args:
            target: Identifier for the target function or tool.

        Returns:
            str: Explanatory error message.
        """
        msg = f"Circuit breaker is OPEN for '{target}'. Tool is temporarily unavailable."
        if self._cb_config.circuit_breaker_message:
            msg = f"{msg} {self._cb_config.circuit_breaker_message}"
        return msg

    async def _before_invocation(self, target: str) -> bool:
        """Evaluate circuit breaker state and determine whether invocation can proceed.

        Args:
            target: Identifier for the target function or tool.

        Returns:
            bool: True if execution is admitted as a ``HALF_OPEN`` probe, False if normal ``CLOSED``.

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is ``OPEN`` or busy probing in ``HALF_OPEN``.
        """
        async with self._lock:
            state_record = self._get_tool_state(target)
            now = time.monotonic()

            if state_record.state == CircuitBreakerState.OPEN:
                if (now - state_record.last_state_change) >= self._cb_config.cooldown_period:
                    logger.info(
                        "Circuit breaker for '%s' cooldown period (%ss) elapsed. Transitioning to HALF_OPEN.",
                        target,
                        self._cb_config.cooldown_period,
                    )
                    state_record.state = CircuitBreakerState.HALF_OPEN
                    state_record.last_state_change = now
                    state_record.success_count = 0
                    state_record.half_open_probing = True
                    return True

                logger.warning("Circuit breaker for '%s' is OPEN. Short-circuiting call.", target)
                raise CircuitBreakerOpenError(self._get_short_circuit_message(target))

            if state_record.state == CircuitBreakerState.HALF_OPEN:
                if state_record.half_open_probing:
                    logger.warning(
                        "Circuit breaker for '%s' is HALF_OPEN and probing. Short-circuiting concurrent call.",
                        target,
                    )
                    raise CircuitBreakerOpenError(self._get_short_circuit_message(target))

                state_record.half_open_probing = True
                return True

            return False

    async def _after_invocation_success(self, target: str, *, is_probe: bool = False) -> None:
        """Update circuit breaker state upon a successful execution.

        Args:
            target: Identifier for the target function or tool.
            is_probe: Whether this execution was an admitted probe call in ``HALF_OPEN`` state.
        """
        async with self._lock:
            state_record = self._get_tool_state(target)
            # Only admitted probe calls in HALF_OPEN modify probe success counters and recover to CLOSED
            if is_probe:
                state_record.success_count += 1
                state_record.half_open_probing = False
                if state_record.success_count >= self._cb_config.half_open_success_threshold:
                    logger.info(
                        "Circuit breaker for '%s' recovered (%d successes). Transitioning to CLOSED.",
                        target,
                        state_record.success_count,
                    )
                    state_record.state = CircuitBreakerState.CLOSED
                    state_record.failure_count = 0
                    state_record.success_count = 0
                    state_record.last_state_change = time.monotonic()
            elif state_record.state == CircuitBreakerState.CLOSED:
                state_record.failure_count = 0

    async def _after_invocation_failure(self, target: str, *, is_probe: bool = False) -> None:
        """Update circuit breaker state upon a failed execution.

        Args:
            target: Identifier for the target function or tool.
            is_probe: Whether this execution was an admitted probe call in ``HALF_OPEN`` state.
        """
        async with self._lock:
            state_record = self._get_tool_state(target)
            now = time.monotonic()
            # Only admitted probe calls in HALF_OPEN retrip the breaker back to OPEN
            if is_probe:
                logger.warning("Circuit breaker probe for '%s' failed! Retripping to OPEN.", target)
                state_record.state = CircuitBreakerState.OPEN
                state_record.last_state_change = now
                state_record.half_open_probing = False
                state_record.failure_count = self._cb_config.failure_threshold
                state_record.success_count = 0
            elif state_record.state == CircuitBreakerState.CLOSED:
                state_record.failure_count += 1
                if state_record.failure_count >= self._cb_config.failure_threshold:
                    logger.warning(
                        "Circuit breaker for '%s' tripped! Transitioning to OPEN after %d consecutive failures.",
                        target,
                        state_record.failure_count,
                    )
                    state_record.state = CircuitBreakerState.OPEN
                    state_record.last_state_change = now

    async def _after_invocation_cancellation(self, target: str) -> None:
        """Reset probing flag on cancellation without counting as failure.

        Args:
            target: Identifier for the target function or tool.
        """
        async with self._lock:
            state_record = self._get_tool_state(target)
            state_record.half_open_probing = False

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> Any:
        """Wrap downstream invocation with circuit breaker monitoring and short-circuiting.

        Args:
            args: Positional arguments for the intercepted function.
            call_next: Callable to invoke next middleware or target function.
            context: Static function metadata describing the tool being invoked.
            kwargs: Keyword arguments for the intercepted function.

        Returns:
            Any: The tool result.

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is ``OPEN`` or busy probing.
            Exception: Any exception raised by the downstream callable.
        """
        # Target state is scoped to context.name as FunctionMiddlewareContext exposes no component-level namespace.
        # This is a known limitation if different components expose identically named functions.
        target = context.name
        is_probe = await self._before_invocation(target)

        try:
            if is_probe and self._cb_config.probe_timeout is not None:
                coro = super().function_middleware_invoke(*args, call_next=call_next, context=context, **kwargs)
                result = await asyncio.wait_for(coro, timeout=self._cb_config.probe_timeout)
            else:
                result = await super().function_middleware_invoke(
                    *args,
                    call_next=call_next,
                    context=context,
                    **kwargs,
                )
        except asyncio.CancelledError:
            # Cancellation must not count as a failure
            await self._after_invocation_cancellation(target)
            raise
        except Exception:
            await self._after_invocation_failure(target, is_probe=is_probe)
            raise
        else:
            await self._after_invocation_success(target, is_probe=is_probe)
            return result

    async def function_middleware_stream(
        self,
        *args: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Wrap downstream streaming call with circuit breaker monitoring and short-circuiting.

        Args:
            args: Positional arguments for the intercepted function.
            call_next: Callable to invoke next middleware or target stream.
            context: Static function metadata describing the tool being invoked.
            kwargs: Keyword arguments for the intercepted function.

        Yields:
            Any: Stream chunks from downstream execution.

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is ``OPEN`` or busy probing.
            Exception: Any exception raised by the downstream stream.
        """
        # Target state is scoped to context.name as FunctionMiddlewareContext exposes no component-level namespace.
        # This is a known limitation if different components expose identically named functions.
        target = context.name
        is_probe = await self._before_invocation(target)

        handled = False
        try:
            if is_probe and self._cb_config.probe_timeout is not None:
                # Wrap only each downstream __anext__ in timeout so consumer delays during yield do not trigger timeouts
                async with contextlib.aclosing(super().function_middleware_stream(
                        *args,
                        call_next=call_next,
                        context=context,
                        **kwargs,
                )) as stream:
                    stream_iter = stream.__aiter__()
                    while True:
                        try:
                            async with asyncio.timeout(self._cb_config.probe_timeout):
                                chunk = await stream_iter.__anext__()
                        except StopAsyncIteration:
                            break
                        yield chunk
            else:
                async for chunk in super().function_middleware_stream(
                        *args,
                        call_next=call_next,
                        context=context,
                        **kwargs,
                ):
                    yield chunk
        except asyncio.CancelledError:
            handled = True
            await self._after_invocation_cancellation(target)
            raise
        except Exception:
            handled = True
            await self._after_invocation_failure(target, is_probe=is_probe)
            raise
        else:
            handled = True
            await self._after_invocation_success(target, is_probe=is_probe)
        finally:
            # Finalize abandoned generators where no explicit handler ran during generator finalization (via aclose())
            if is_probe and not handled:
                await self._after_invocation_cancellation(target)


__all__ = ["CircuitBreakerMiddleware", "CircuitBreakerOpenError", "CircuitBreakerState"]
