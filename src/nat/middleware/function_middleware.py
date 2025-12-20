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
"""Function-specific middleware for the NeMo Agent toolkit.

This module provides function-specific middleware implementations that extend
the base Middleware class. FunctionMiddleware is a specialized middleware type
designed specifically for wrapping function calls with dedicated methods
for function-specific preprocessing and postprocessing.

Middleware is configured at registration time and is bound to instances when they
are constructed by the workflow builder.

Middleware executes in the order provided and can optionally be marked as *final*.
A final middleware terminates the chain, preventing subsequent middleware or the
wrapped target from running unless the final middleware explicitly delegates to
the next callable.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import Any

from nat.middleware.middleware import CallNext
from nat.middleware.middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext
from nat.middleware.middleware import Middleware


class FunctionMiddleware(Middleware):
    """Base class for function middleware with pre/post-invoke hooks.

    Middleware intercepts function calls and can:
    - Transform inputs before execution (pre_invoke)
    - Transform outputs after execution (post_invoke)
    - Override function_middleware_invoke for full control

    Lifecycle:
    - Framework checks ``enabled`` property before calling any methods
    - If disabled, middleware is skipped entirely (no methods called)
    - Users do NOT need to check ``enabled`` in their implementations

    Inherited abstract members that must be implemented:
    - enabled: Property that returns whether middleware should run
    - pre_invoke: Transform inputs before function execution
    - post_invoke: Transform outputs after function execution

    Context Flow:
    - FunctionMiddlewareContext (frozen): Static function metadata only
    - InvocationContext: Unified context for both pre and post invoke phases
    - Pre-invoke: output is None, modify modified_args/modified_kwargs
    - Post-invoke: output has the result, modify output to transform

    Example::

        class LoggingMiddleware(FunctionMiddleware):
            def __init__(self, config: LoggingConfig):
                super().__init__()
                self._config = config

            @property
            def enabled(self) -> bool:
                return self._config.enabled

            async def pre_invoke(self, context: InvocationContext) -> InvocationContext | None:
                logger.info(f"Calling {context.function_context.name} with {context.modified_args}")
                logger.info(f"Original args: {context.original_args}")
                return None  # Pass through unchanged

            async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
                logger.info(f"Result: {context.output}")
                return None  # Pass through unchanged
    """

    # ==================== Middleware Delegation ====================
    async def middleware_invoke(self,
                                *args: Any,
                                call_next: CallNext,
                                context: FunctionMiddlewareContext,
                                **kwargs: Any) -> Any:
        """Delegate to function_middleware_invoke for function-specific handling."""
        return await self.function_middleware_invoke(*args, call_next=call_next, context=context, **kwargs)

    async def middleware_stream(self,
                                *args: Any,
                                call_next: CallNextStream,
                                context: FunctionMiddlewareContext,
                                **kwargs: Any) -> AsyncIterator[Any]:
        """Delegate to function_middleware_stream for function-specific handling."""
        async for chunk in self.function_middleware_stream(*args, call_next=call_next, context=context, **kwargs):
            yield chunk

    # ==================== Orchestration ====================

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> Any:
        """Execute middleware hooks around function call.

        Default implementation orchestrates: pre_invoke → call_next → post_invoke

        Override for full control over execution flow (e.g., caching,
        retry logic, conditional execution).

        Note: Framework checks ``enabled`` before calling this method.
        You do NOT need to check ``enabled`` yourself.

        Args:
            args: Positional arguments for the function
            call_next: Callable to invoke next middleware or target function
            context: Static function metadata
            kwargs: Keyword arguments for the function

        Returns:
            The (potentially transformed) function output
        """
        # Build invocation context with frozen originals + mutable current
        # output starts as None (pre-invoke phase)
        ctx = InvocationContext(
            function_context=context,
            original_args=args,
            original_kwargs=dict(kwargs),
            modified_args=args,
            modified_kwargs=dict(kwargs),
            output=None,
        )

        # Pre-invoke transformation (output is None at this phase)
        result = await self.pre_invoke(ctx)
        if result is not None:
            ctx = result

        # Execute function with (potentially modified) args/kwargs
        ctx.output = await call_next(*ctx.modified_args, **ctx.modified_kwargs)

        # Post-invoke transformation (output now has the result)
        result = await self.post_invoke(ctx)
        if result is not None:
            ctx = result

        return ctx.output

    async def function_middleware_stream(
        self,
        *args: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Execute middleware hooks around streaming function call.

        Pre-invoke runs once before streaming starts.
        Post-invoke runs per-chunk as they stream through.

        Override for custom streaming behavior (e.g., buffering,
        aggregation, chunk filtering).

        Note: Framework checks ``enabled`` before calling this method.
        You do NOT need to check ``enabled`` yourself.

        Args:
            args: Positional arguments for the function
            call_next: Callable to invoke next middleware or target stream
            context: Static function metadata
            kwargs: Keyword arguments for the function

        Yields:
            Stream chunks (potentially transformed by post_invoke)
        """
        # Build invocation context with frozen originals + mutable current
        # output starts as None (pre-invoke phase)
        ctx = InvocationContext(
            function_context=context,
            original_args=args,
            original_kwargs=dict(kwargs),
            modified_args=args,
            modified_kwargs=dict(kwargs),
            output=None,
        )

        # Pre-invoke transformation (once before streaming)
        result = await self.pre_invoke(ctx)
        if result is not None:
            ctx = result

        # Stream with per-chunk post-invoke
        async for chunk in call_next(*ctx.modified_args, **ctx.modified_kwargs):
            # Set output for this chunk
            ctx.output = chunk

            # Post-invoke transformation per chunk
            result = await self.post_invoke(ctx)
            if result is not None:
                ctx = result

            yield ctx.output


class FunctionMiddlewareChain:
    """Composes middleware into an execution chain.

    The chain builder checks each middleware's ``enabled`` property.
    Disabled middleware is skipped entirely—no methods are called.

    Execution order:
    - Pre-invoke: first middleware → last middleware → function
    - Post-invoke: function → last middleware → first middleware

    Context:
    - FunctionMiddlewareContext contains only static function metadata
    - Original args/kwargs are captured by the orchestration layer
    - Middleware receives InvocationContext with frozen originals and mutable args/output
    """

    def __init__(self, *, middleware: Sequence[FunctionMiddleware], context: FunctionMiddlewareContext) -> None:
        """Initialize the middleware chain.

        Args:
            middleware: Sequence of middleware to chain (order matters)
            context: Static function metadata
        """
        self._middleware = tuple(middleware)
        self._context = context

    def build_single(self, final_call: CallNext) -> CallNext:
        """Build the middleware chain for single-output invocations.

        Disabled middleware (enabled=False) is skipped entirely.

        Args:
            final_call: The final function to call (the actual function implementation)

        Returns:
            A callable that executes the entire middleware chain
        """
        call = final_call

        for mw in reversed(self._middleware):
            # Framework-enforced: skip disabled middleware
            if not mw.enabled:
                continue

            call_next = call

            async def wrapped(*args: Any,
                              _middleware: FunctionMiddleware = mw,
                              _call_next: CallNext = call_next,
                              _context: FunctionMiddlewareContext = self._context,
                              **kwargs: Any) -> Any:
                return await _middleware.middleware_invoke(*args, call_next=_call_next, context=_context, **kwargs)

            call = wrapped  # type: ignore[assignment]

        return call

    def build_stream(self, final_call: CallNextStream) -> CallNextStream:
        """Build the middleware chain for streaming invocations.

        Disabled middleware (enabled=False) is skipped entirely.

        Args:
            final_call: The final function to call (the actual function implementation)

        Returns:
            A callable that executes the entire middleware chain
        """
        call = final_call

        for mw in reversed(self._middleware):
            if not mw.enabled:
                continue

            call_next = call

            async def wrapped(*args: Any,
                              _middleware: FunctionMiddleware = mw,
                              _call_next: CallNextStream = call_next,
                              _context: FunctionMiddlewareContext = self._context,
                              **kwargs: Any) -> AsyncIterator[Any]:
                stream = _middleware.middleware_stream(*args, call_next=_call_next, context=_context, **kwargs)
                async for chunk in stream:
                    yield chunk

            call = wrapped  # type: ignore[assignment]

        return call


def validate_middleware(middleware: Sequence[Middleware] | None) -> tuple[Middleware, ...]:
    """Validate a sequence of middleware, enforcing ordering guarantees."""

    if not middleware:
        return tuple()

    final_found = False
    for idx, mw in enumerate(middleware):
        if not isinstance(mw, Middleware):
            raise TypeError("All middleware must be instances of Middleware")

        if mw.is_final:
            if final_found:
                raise ValueError("Only one final Middleware may be specified per function")

            if idx != len(middleware) - 1:
                raise ValueError("A final Middleware must be the last middleware in the chain")

            final_found = True

    return tuple(middleware)


__all__ = [
    "FunctionMiddleware",
    "FunctionMiddlewareChain",
    "validate_middleware",
]
