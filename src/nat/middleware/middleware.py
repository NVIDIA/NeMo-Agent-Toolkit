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
"""Base middleware class for the NeMo Agent toolkit.

This module provides the base Middleware class that defines the middleware pattern
for wrapping and modifying function calls. Middleware works like middleware in
web frameworks - they can modify inputs, call the next middleware in the chain,
process outputs, and continue.
"""

from __future__ import annotations

import dataclasses
from abc import ABC
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

#: Type alias for single-output invocation callables.
CallNext = Callable[[Any], Awaitable[Any]]

#: Type alias for streaming invocation callables.
CallNextStream = Callable[[Any], AsyncIterator[Any]]


@dataclasses.dataclass(frozen=True, kw_only=True)
class FunctionMiddlewareContext:
    """Context information about the function being wrapped by middleware.

    Middleware receives this context object which describes the function they
    are wrapping. This allows middleware to make decisions based on the
    function's name, configuration, schema, etc.
    """

    name: str
    """Name of the function being wrapped."""

    config: Any
    """Configuration object for the function."""

    description: str | None
    """Optional description of the function."""

    input_schema: type[BaseModel] | None
    """Schema describing expected inputs or :class:`NoneType` when absent."""

    single_output_schema: type[BaseModel] | type[None]
    """Schema describing single outputs or :class:`types.NoneType` when absent."""

    stream_output_schema: type[BaseModel] | type[None]
    """Schema describing streaming outputs or :class:`types.NoneType` when absent."""


class Middleware(ABC):
    """Base class for middleware-style wrapping.

    Middleware works like middleware in web frameworks:

    1. **Preprocess**: Inspect and optionally modify inputs
    2. **Call Next**: Delegate to the next middleware or the target itself
    3. **Postprocess**: Process, transform, or augment the output
    4. **Continue**: Return or yield the final result

    Example::

        class LoggingMiddleware(Middleware):
            async def middleware_invoke(self, value, call_next, context):
                # 1. Preprocess
                print(f"Input: {value}")

                # 2. Call next middleware/target
                result = await call_next(value)

                # 3. Postprocess
                print(f"Output: {result}")

                # 4. Continue
                return result

    Attributes:
        is_final: If True, this middleware terminates the chain. No subsequent
            middleware or the target will be called unless this middleware
            explicitly delegates to ``call_next``.
    """

    def __init__(self, *, is_final: bool = False) -> None:
        self._is_final = is_final

    @property
    def is_final(self) -> bool:
        """Whether this middleware terminates the chain.

        A final middleware prevents subsequent middleware and the target
        from running unless it explicitly calls ``call_next``.
        """

        return self._is_final

    async def middleware_invoke(self, value: Any, call_next: CallNext, context: FunctionMiddlewareContext) -> Any:
        """Middleware for single-output invocations.

        Args:
            value: The input value to process
            call_next: Callable to invoke the next middleware or target
            context: Metadata about the target being wrapped

        Returns:
            The (potentially modified) output from the target

        The default implementation simply delegates to ``call_next``. Override this
        to add preprocessing, postprocessing, or to short-circuit execution::

            async def middleware_invoke(self, value, call_next, context):
                # Preprocess: modify input
                modified_input = transform(value)

                # Call next: delegate to next middleware/target
                result = await call_next(modified_input)

                # Postprocess: modify output
                modified_result = transform_output(result)

                # Continue: return final result
                return modified_result
        """

        del context  # Unused by the default implementation.
        return await call_next(value)

    async def middleware_stream(self, value: Any, call_next: CallNextStream,
                                context: FunctionMiddlewareContext) -> AsyncIterator[Any]:
        """Middleware for streaming invocations.

        Args:
            value: The input value to process
            call_next: Callable to invoke the next middleware or target stream
            context: Metadata about the target being wrapped

        Yields:
            Chunks from the stream (potentially modified)

        The default implementation forwards to ``call_next`` untouched. Override this
        to add preprocessing, transform chunks, or perform cleanup::

            async def middleware_stream(self, value, call_next, context):
                # Preprocess: setup or modify input
                modified_input = transform(value)

                # Call next: get stream from next middleware/target
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


__all__ = [
    "CallNext",
    "CallNextStream",
    "Middleware",
    "FunctionMiddlewareContext",
]
