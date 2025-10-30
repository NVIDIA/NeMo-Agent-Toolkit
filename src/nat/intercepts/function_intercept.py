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
"""Middleware-style function intercepts for the NeMo Agent toolkit.

This module provides a middleware pattern for function intercepts, allowing you to
wrap function calls with preprocessing and postprocessing logic. Intercepts work like
middleware in web frameworks - they can modify inputs, call the next intercept in the
chain, process outputs, and continue.

Intercepts are configured at registration time through the ``@register_function``
decorator and are bound to function instances when they are constructed by the
workflow builder.

Intercepts execute in the order provided and can optionally be marked as *final*.
A final intercept terminates the chain, preventing subsequent intercepts or the
wrapped function from running unless the final intercept explicitly delegates to
the next callable.
"""

from __future__ import annotations

import dataclasses
from abc import ABC
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from nat.data_models.function import FunctionBaseConfig

CallNext = Callable[[Any], Awaitable[Any]]
"""Callable signature for calling the next middleware in the chain (single-output)."""

CallNextStream = Callable[[Any], AsyncIterator[Any]]
"""Callable signature for calling the next middleware in the chain (streaming)."""


@dataclasses.dataclass(frozen=True)
class FunctionInterceptContext:
    """Context information supplied to each intercept when invoked."""

    name: str
    """The workflow-scoped name of the function instance."""

    config: FunctionBaseConfig
    """The configuration object used to build the function."""

    description: str | None
    """Human-readable description of the function."""

    input_schema: type[BaseModel]
    """Schema describing the validated input payload."""

    single_output_schema: type[BaseModel] | type[None]
    """Schema describing single outputs or :class:`types.NoneType` when absent."""

    stream_output_schema: type[BaseModel] | type[None]
    """Schema describing streaming outputs or :class:`types.NoneType` when absent."""


class FunctionIntercept(ABC):
    """Base class for middleware-style function intercepts.

    Function intercepts work like middleware in web frameworks:

    1. **Preprocess**: Inspect and optionally modify inputs
    2. **Call Next**: Delegate to the next intercept or the function itself
    3. **Postprocess**: Process, transform, or augment the output
    4. **Continue**: Return or yield the final result

    Example::

        class LoggingIntercept(FunctionIntercept):
            async def intercept_invoke(self, value, call_next, context):
                # 1. Preprocess
                print(f"Input: {value}")

                # 2. Call next middleware/function
                result = await call_next(value)

                # 3. Postprocess
                print(f"Output: {result}")

                # 4. Continue
                return result

    Concrete intercepts can override :meth:`intercept_invoke` and
    :meth:`intercept_stream` to implement custom middleware logic.
    """

    def __init__(self, *, is_final: bool = False) -> None:
        self._is_final = is_final

    @property
    def is_final(self) -> bool:
        """Whether this intercept terminates the chain by default."""

        return self._is_final

    async def intercept_invoke(self, value: Any, call_next: CallNext, context: FunctionInterceptContext) -> Any:
        """Middleware for single-output invocations.

        Args:
            value: The input value to process
            call_next: Callable to invoke the next middleware or function
            context: Metadata about the function being intercepted

        Returns:
            The (potentially modified) output from the function

        The default implementation simply delegates to ``call_next``. Override this
        to add preprocessing, postprocessing, or to short-circuit execution::

            async def intercept_invoke(self, value, call_next, context):
                # Preprocess: modify input
                modified_input = transform(value)

                # Call next: delegate to next middleware/function
                result = await call_next(modified_input)

                # Postprocess: modify output
                modified_result = transform_output(result)

                # Continue: return final result
                return modified_result
        """

        del context  # Unused by the default implementation.
        return await call_next(value)

    async def intercept_stream(self, value: Any, call_next: CallNextStream,
                               context: FunctionInterceptContext) -> AsyncIterator[Any]:
        """Middleware for streaming invocations.

        Args:
            value: The input value to process
            call_next: Callable to invoke the next middleware or function stream
            context: Metadata about the function being intercepted

        Yields:
            Chunks from the stream (potentially modified)

        The default implementation forwards to ``call_next`` untouched. Override this
        to add preprocessing, transform chunks, or perform cleanup::

            async def intercept_stream(self, value, call_next, context):
                # Preprocess: setup or modify input
                modified_input = transform(value)

                # Call next: get stream from next middleware/function
                async for chunk in call_next(modified_input):
                    # Process each chunk
                    modified_chunk = transform_chunk(chunk)
                    yield modified_chunk

                # Postprocess: cleanup after stream ends
                await cleanup()
        """

        del context  # Unused by the default implementation.
        async for chunk in call_next(value):
            yield chunk


class FunctionInterceptChain:
    """Utility that composes middleware-style intercept callables.

    This class builds a chain of middleware intercepts that execute in order,
    with each intercept able to preprocess inputs, call the next middleware,
    and postprocess outputs.
    """

    def __init__(self, *, intercepts: Sequence[FunctionIntercept], context: FunctionInterceptContext) -> None:
        self._intercepts = tuple(intercepts)
        self._context = context

    def build_single(self, final_call: CallNext) -> CallNext:
        """Build the middleware chain for single-output invocations.

        Args:
            final_call: The final function to call (the actual function implementation)

        Returns:
            A callable that executes the entire middleware chain
        """
        call = final_call

        for intercept in reversed(self._intercepts):
            call_next = call

            async def wrapped(value: Any,
                              *,
                              _intercept: FunctionIntercept = intercept,
                              _call_next: CallNext = call_next) -> Any:
                return await _intercept.intercept_invoke(value, _call_next, self._context)

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

        for intercept in reversed(self._intercepts):
            call_next = call

            async def wrapped(value: Any,
                              *,
                              _intercept: FunctionIntercept = intercept,
                              _call_next: CallNextStream = call_next) -> AsyncIterator[Any]:
                async for chunk in _intercept.intercept_stream(value, _call_next, self._context):
                    yield chunk

            call = wrapped

        return call


def validate_intercepts(intercepts: Sequence[FunctionIntercept] | None) -> tuple[FunctionIntercept, ...]:
    """Validate a sequence of intercepts, enforcing ordering guarantees."""

    if not intercepts:
        return tuple()

    final_found = False
    for idx, intercept in enumerate(intercepts):
        if not isinstance(intercept, FunctionIntercept):
            raise TypeError("All intercepts must be instances of FunctionIntercept")

        if intercept.is_final:
            if final_found:
                raise ValueError("Only one final FunctionIntercept may be specified per function")

            if idx != len(intercepts) - 1:
                raise ValueError("A final FunctionIntercept must be the last intercept in the chain")

            final_found = True

    return tuple(intercepts)


__all__ = [
    "CallNext",
    "CallNextStream",
    "FunctionIntercept",
    "FunctionInterceptChain",
    "FunctionInterceptContext",
    "validate_intercepts",
]
