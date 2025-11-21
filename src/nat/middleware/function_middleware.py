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
from nat.middleware.middleware import Middleware


class FunctionMiddleware(Middleware):
    """Specialized middleware for function-specific wrapping.

    This class extends the base Middleware class and provides function-specific
    wrapping methods. Functions that use this middleware type will call
    ``function_middleware_invoke`` and ``function_middleware_stream`` instead of
    the base ``middleware_invoke`` and ``middleware_stream`` methods.
    """

    async def middleware_invoke(self, value: Any, call_next: CallNext, context: FunctionMiddlewareContext) -> Any:
        """Delegate to function_middleware_invoke for function-specific handling."""
        return await self.function_middleware_invoke(value, call_next, context)

    async def middleware_stream(self, value: Any, call_next: CallNextStream,
                                context: FunctionMiddlewareContext) -> AsyncIterator[Any]:
        """Delegate to function_middleware_stream for function-specific handling."""
        async for chunk in self.function_middleware_stream(value, call_next, context):
            yield chunk

    async def function_middleware_invoke(self, value: Any, call_next: CallNext,
                                         context: FunctionMiddlewareContext) -> Any:
        """Function-specific middleware for single-output invocations.

        Args:
            value: The input value to process
            call_next: Callable to invoke the next middleware or function
            context: Metadata about the function being wrapped

        Returns:
            The (potentially modified) output from the function

        The default implementation simply delegates to ``call_next``. Override this
        in subclasses to add function-specific preprocessing and postprocessing.
        """
        return await call_next(value)

    async def function_middleware_stream(self,
                                         value: Any,
                                         call_next: CallNextStream,
                                         context: FunctionMiddlewareContext) -> AsyncIterator[Any]:
        """Function-specific middleware for streaming invocations.

        Args:
            value: The input value to process
            call_next: Callable to invoke the next middleware or function stream
            context: Metadata about the function being wrapped

        Yields:
            Chunks from the stream (potentially modified)

        The default implementation forwards to ``call_next`` untouched. Override this
        in subclasses to add function-specific preprocessing and chunk transformations.
        """
        async for chunk in call_next(value):
            yield chunk


class FunctionMiddlewareChain:
    """Utility that composes middleware-style callables.

    This class builds a chain of middleware that executes in order,
    with each middleware able to preprocess inputs, call the next middleware,
    and postprocess outputs.
    """

    def __init__(self, *, middleware: Sequence[Middleware], context: FunctionMiddlewareContext) -> None:
        self._middleware = tuple(middleware)
        self._context = context

    def build_single(self, final_call: CallNext) -> CallNext:
        """Build the middleware chain for single-output invocations.

        Args:
            final_call: The final function to call (the actual function implementation)

        Returns:
            A callable that executes the entire middleware chain
        """
        call = final_call

        for mw in reversed(self._middleware):
            call_next = call

            async def wrapped(value: Any, *, _middleware: Middleware = mw, _call_next: CallNext = call_next) -> Any:
                return await _middleware.middleware_invoke(value, _call_next, self._context)

            call = wrapped

        return call

    def build_stream(self, final_call: CallNextStream) -> CallNextStream:
        """Build the middleware chain for streaming invocations.

        Args:
            final_call: The final function to call (the actual function implementation)

        Returns:
            A callable that executes the entire middleware chain
        """
        call = final_call

        for mw in reversed(self._middleware):
            call_next = call

            async def wrapped(value: Any,
                              *,
                              _middleware: Middleware = mw,
                              _call_next: CallNextStream = call_next) -> AsyncIterator[Any]:
                async for chunk in _middleware.middleware_stream(value, _call_next, self._context):
                    yield chunk

            call = wrapped

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
