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
from abc import abstractmethod
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

#: Type alias for single-output invocation callables.
CallNext = Callable[..., Awaitable[Any]]

#: Type alias for streaming invocation callables.
CallNextStream = Callable[..., AsyncIterator[Any]]


@dataclasses.dataclass(frozen=True, kw_only=True)
class FunctionMiddlewareContext:
    """Static metadata about the function being wrapped by middleware.

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


class InvocationContext(BaseModel):
    """Unified context for pre-invoke and post-invoke phases.

    Used for both phases of middleware execution:
    - Pre-invoke: output is None, modify modified_args/modified_kwargs to transform inputs
    - Post-invoke: output contains the function result, modify output to transform results

    This unified context simplifies the middleware interface by using a single
    context type for both hooks.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Frozen fields - cannot be modified after creation
    function_context: FunctionMiddlewareContext = Field(
        frozen=True, description="Static metadata about the function being invoked (frozen).")
    original_args: tuple[Any, ...] = Field(
        frozen=True, description="The original function input arguments before any middleware processing.")
    original_kwargs: dict[str, Any] = Field(
        frozen=True, description="The original function input keyword arguments before any middleware processing.")

    # Mutable fields - modify these to transform inputs/outputs
    modified_args: tuple[Any, ...] = Field(description="Modified args after middleware processing.")
    modified_kwargs: dict[str, Any] = Field(description="Modified kwargs after middleware processing.")
    output: Any = Field(default=None, description="Function output. None pre-invoke, result post-invoke.")


class Middleware(ABC):
    """Base class for middleware-style wrapping with pre/post-invoke hooks.

    Middleware works like middleware in web frameworks:

    1. **Preprocess**: Inspect and optionally modify inputs (via pre_invoke)
    2. **Call Next**: Delegate to the next middleware or the target itself
    3. **Postprocess**: Process, transform, or augment the output (via post_invoke)
    4. **Continue**: Return or yield the final result

    Example::

        class LoggingMiddleware(FunctionMiddleware):
            @property
            def enabled(self) -> bool:
                return True

            async def pre_invoke(self, context: InvocationContext) -> InvocationContext | None:
                print(f"Current args: {context.modified_args}")
                print(f"Original args: {context.original_args}")
                return None  # Pass through unchanged

            async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
                print(f"Output: {context.output}")
                return None  # Pass through unchanged

    Attributes:
        is_final: If True, this middleware terminates the chain. No subsequent
            middleware or the target will be called unless this middleware
            explicitly delegates to ``call_next``.
    """

    def __init__(self, *, is_final: bool = False) -> None:
        self._is_final = is_final

    # ==================== Abstract Members ====================

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this middleware should execute.
        """
        ...

    @abstractmethod
    async def pre_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Transform inputs before execution.

        Called by specialized middleware invoke methods (e.g., function_middleware_invoke).
        Use to validate, transform, or augment inputs. At this phase, context.output is None.

        Args:
            context: Invocation context (Pydantic model) containing:
                - function_context: Static function metadata (frozen)
                - original_args: What entered the middleware chain (frozen)
                - original_kwargs: What entered the middleware chain (frozen)
                - modified_args: Current args (mutable)
                - modified_kwargs: Current kwargs (mutable)
                - output: None (function not yet called)

        Returns:
            InvocationContext: Return the (modified) context to signal changes
            None: Pass through unchanged (framework uses current context state)

        Note:
            Frozen fields (original_args, original_kwargs) cannot be modified.
            Attempting to modify them raises ValidationError.

        Raises:
            Any exception to abort execution
        """
        ...

    @abstractmethod
    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Transform output after execution.

        Called by specialized middleware invoke methods (e.g., function_middleware_invoke).
        For streaming, called per-chunk. Use to validate, transform, or augment outputs.

        Args:
            context: Invocation context (Pydantic model) containing:
                - function_context: Static function metadata (frozen)
                - original_args: What entered the middleware chain (frozen)
                - original_kwargs: What entered the middleware chain (frozen)
                - modified_args: What the function received (mutable)
                - modified_kwargs: What the function received (mutable)
                - output: Current output value (mutable)

        Returns:
            InvocationContext: Return the (modified) context to signal changes
            None: Pass through unchanged (framework uses current context.output)

        Example::

            async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
                # Wrap the output
                context.output = {"result": context.output, "processed": True}
                return context  # Signal modification

        Raises:
            Any exception to abort and propagate error
        """
        ...

    # ==================== Properties ====================

    @property
    def is_final(self) -> bool:
        """Whether this middleware terminates the chain.

        A final middleware prevents subsequent middleware and the target
        from running unless it explicitly calls ``call_next``.
        """

        return self._is_final

    # ==================== Default Invoke Methods ====================

    async def middleware_invoke(self,
                                value: Any,
                                call_next: CallNext,
                                context: FunctionMiddlewareContext,
                                **kwargs: Any) -> Any:
        """Middleware for single-output invocations.

        Args:
            value: The input value to process
            call_next: Callable to invoke the next middleware or target
            context: Metadata about the target being wrapped
            kwargs: Additional function arguments

        Returns:
            The (potentially modified) output from the target

        The default implementation simply delegates to ``call_next``. Override this
        to add preprocessing, postprocessing, or to short-circuit execution::

            async def middleware_invoke(self, value, call_next, context, **kwargs):
                # Preprocess: modify input
                modified_input = transform(value)

                # Call next: delegate to next middleware/target
                result = await call_next(modified_input, **kwargs)

                # Postprocess: modify output
                modified_result = transform_output(result)

                # Continue: return final result
                return modified_result
        """

        del context  # Unused by the default implementation.
        return await call_next(value, **kwargs)

    async def middleware_stream(self,
                                value: Any,
                                call_next: CallNextStream,
                                context: FunctionMiddlewareContext,
                                **kwargs: Any) -> AsyncIterator[Any]:
        """Middleware for streaming invocations.

        Args:
            value: The input value to process
            call_next: Callable to invoke the next middleware or target stream
            context: Metadata about the target being wrapped
            kwargs: Additional function arguments

        Yields:
            Chunks from the stream (potentially modified)

        The default implementation forwards to ``call_next`` untouched. Override this
        to add preprocessing, transform chunks, or perform cleanup::

            async def middleware_stream(self, value, call_next, context, **kwargs):
                # Preprocess: setup or modify input
                modified_input = transform(value)

                # Call next: get stream from next middleware/target
                async for chunk in call_next(modified_input, **kwargs):
                    # Process each chunk
                    modified_chunk = transform_chunk(chunk)
                    yield modified_chunk

                # Postprocess: cleanup after stream ends
                await cleanup()
        """

        del context  # Unused by the default implementation.
        async for chunk in call_next(value, **kwargs):
            yield chunk


__all__ = [
    "CallNext",
    "CallNextStream",
    "FunctionMiddlewareContext",
    "InvocationContext",
    "Middleware",
]
