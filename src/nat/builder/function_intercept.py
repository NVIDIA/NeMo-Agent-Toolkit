# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for configuring per-function intercept chains.

This module introduces the :class:`FunctionIntercept` ABC alongside utility
structures that allow callers to register interceptors which run before a
function's ``ainvoke``/``astream`` logic.  Intercepts are configured at
registration time through the ``@register_function`` decorator and are bound to
function instances when they are constructed by the workflow builder.

Intercepts execute in the order that they are provided and can optionally be
marked as *final*.  A final intercept terminates the chain, preventing any
subsequent intercepts or the wrapped function from running unless the final
intercept explicitly delegates to the next callable.
"""

from __future__ import annotations

import dataclasses
from abc import ABC
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from typing import Sequence

from pydantic import BaseModel

from nat.data_models.function import FunctionBaseConfig

SingleInvokeCallable = Callable[[Any], Awaitable[Any]]
"""Callable signature used for single-output intercept chaining."""

StreamInvokeCallable = Callable[[Any], AsyncIterator[Any]]
"""Callable signature used for streaming intercept chaining."""


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
    """Base class for intercept implementations.

    Concrete intercepts can override :meth:`intercept_invoke` and
    :meth:`intercept_stream` to perform arbitrary logic before delegating to the
    next callable in the chain.  Intercepts must preserve the function's input
    and output schemas and should only return values that satisfy the wrapped
    function's contracts.
    """

    def __init__(self, *, is_final: bool = False) -> None:
        self._is_final = is_final

    @property
    def is_final(self) -> bool:
        """Whether this intercept terminates the chain by default."""

        return self._is_final

    async def intercept_invoke(self,
                               value: Any,
                               next_call: SingleInvokeCallable,
                               context: FunctionInterceptContext) -> Any:
        """Intercept a single-output invocation.

        The default implementation simply delegates to ``next_call``.  Derived
        classes can override this method to add behaviour before or after the
        call, or bypass ``next_call`` entirely (for example, final intercepts).
        """

        del context  # Unused by the default implementation.
        return await next_call(value)

    async def intercept_stream(self,
                               value: Any,
                               next_call: StreamInvokeCallable,
                               context: FunctionInterceptContext) -> AsyncIterator[Any]:
        """Intercept a streaming invocation.

        The default implementation forwards to ``next_call`` untouched.  Custom
        intercepts can yield additional values, transform the stream, or skip
        delegation entirely.
        """

        del context  # Unused by the default implementation.
        async for chunk in next_call(value):
            yield chunk


class FunctionInterceptChain:
    """Utility that composes intercept callables for a function instance."""

    def __init__(self, *,
                 intercepts: Sequence[FunctionIntercept],
                 context: FunctionInterceptContext) -> None:
        self._intercepts = tuple(intercepts)
        self._context = context

    def build_single(self, final_call: SingleInvokeCallable) -> SingleInvokeCallable:
        call = final_call

        for intercept in reversed(self._intercepts):
            next_call = call

            async def wrapped(value: Any,
                              *,
                              _intercept: FunctionIntercept = intercept,
                              _next_call: SingleInvokeCallable = next_call) -> Any:
                return await _intercept.intercept_invoke(value, _next_call, self._context)

            call = wrapped

        return call

    def build_stream(self, final_call: StreamInvokeCallable) -> StreamInvokeCallable:
        call = final_call

        for intercept in reversed(self._intercepts):
            next_call = call

            async def wrapped(value: Any,
                              *,
                              _intercept: FunctionIntercept = intercept,
                              _next_call: StreamInvokeCallable = next_call) -> AsyncIterator[Any]:
                async for chunk in _intercept.intercept_stream(value, _next_call, self._context):
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
    "FunctionIntercept",
    "FunctionInterceptChain",
    "FunctionInterceptContext",
    "SingleInvokeCallable",
    "StreamInvokeCallable",
    "validate_intercepts",
]
