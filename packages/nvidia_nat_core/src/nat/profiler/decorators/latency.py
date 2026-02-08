# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Latency sensitivity decorator for marking functions with latency requirements.

This module provides the @latency_sensitive decorator that allows marking functions
with latency sensitivity levels (LOW, MEDIUM, HIGH). The sensitivity propagates through
the call stack with priority-based merging, where higher sensitivity takes precedence.

Use cases:
- LLM routing: Direct high-sensitivity requests to low-latency backends
- Execution optimization: Adjust timeouts, batch sizes based on sensitivity
- Observability: Track which parts of workflows have strict latency requirements

Example:
    Basic usage with enum::

        from nat.profiler.decorators.latency import LatencySensitivity, latency_sensitive

        @latency_sensitive(LatencySensitivity.HIGH)
        async def critical_llm_call():
            return await llm.generate()

    Using string form::

        @latency_sensitive("low")
        def background_task():
            pass

    Reading current sensitivity::

        from nat.builder.context import Context

        def my_function():
            sensitivity = Context.get().latency_sensitivity
            if sensitivity == LatencySensitivity.HIGH:
                # Use fast path
                pass
"""

import functools
import inspect
from collections.abc import Callable
from enum import StrEnum
from typing import Any
from typing import TypeVar


class LatencySensitivity(StrEnum):
    """
    Latency sensitivity levels for function execution.

    The sensitivity level indicates how time-critical a function's execution is.
    Higher sensitivity values take precedence when contexts are nested.

    Attributes:
        LOW: Low latency sensitivity (priority=1). Suitable for background tasks
            or non-time-critical operations.
        MEDIUM: Medium latency sensitivity (priority=2). Default level for most
            operations. Suitable for typical user interactions.
        HIGH: High latency sensitivity (priority=3). Suitable for critical user-facing
            operations requiring immediate response.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def priority(self) -> int:
        """
        Return numeric priority for this sensitivity level.

        Higher priority values indicate higher sensitivity. Used for comparing
        sensitivities when nesting contexts.

        Returns:
            int: Priority value (LOW=1, MEDIUM=2, HIGH=3)
        """
        return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}[self.value]

    @classmethod
    def parse(cls, value: "LatencySensitivity | str") -> "LatencySensitivity":
        """
        Parse string or enum to LatencySensitivity.

        Accepts either a LatencySensitivity enum value or a string representation.
        String parsing is case-insensitive.

        Args:
            value: Either a LatencySensitivity enum or string like "high", "MEDIUM", "Low"

        Returns:
            LatencySensitivity: Parsed enum value

        Raises:
            ValueError: If value is not a valid sensitivity level

        Example:
            >>> LatencySensitivity.parse("high")
            <LatencySensitivity.HIGH: 'HIGH'>
            >>> LatencySensitivity.parse(LatencySensitivity.LOW)
            <LatencySensitivity.LOW: 'LOW'>
        """
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            normalized = value.upper()
            if normalized in {"LOW", "MEDIUM", "HIGH"}:
                return cls(normalized)

        raise ValueError(f"Invalid latency sensitivity: {value!r}. "
                         f"Must be 'LOW', 'MEDIUM', 'HIGH', or LatencySensitivity enum.")


# Type variable for preserving function signature
F = TypeVar("F", bound=Callable[..., Any])


def latency_sensitive(sensitivity: LatencySensitivity | str) -> Callable[[F], F]:
    """
    Decorator to mark a function with a latency sensitivity level.

    The sensitivity is pushed onto the context stack for the duration of the
    function execution. The effective sensitivity is the maximum priority across
    all pushed levels.

    Args:
        sensitivity: Latency sensitivity level (enum or string like "high", "LOW")

    Returns:
        Decorated function that pushes sensitivity onto context stack

    Raises:
        ValueError: If sensitivity is not a valid level

    Example:
        >>> from nat.profiler.decorators.latency import latency_sensitive, LatencySensitivity
        >>> from nat.builder.context import Context
        >>>
        >>> @latency_sensitive(LatencySensitivity.HIGH)
        ... def critical_function():
        ...     return Context.get().latency_sensitivity
        >>>
        >>> @latency_sensitive("low")
        ... async def background_task():
        ...     return await do_work()
    """
    # Parse and validate at decoration time
    parsed_sensitivity = LatencySensitivity.parse(sensitivity)

    def decorator(func: F) -> F:
        # Import here to avoid circular dependency
        from nat.builder.context import Context

        if inspect.isasyncgenfunction(func):
            # Async generator function
            @functools.wraps(func)
            async def async_gen_wrapper(*args: Any, **kwargs: Any):
                ctx = Context.get()
                with ctx.push_latency_sensitivity(parsed_sensitivity):
                    async for item in func(*args, **kwargs):
                        yield item

            return async_gen_wrapper  # type: ignore

        elif inspect.iscoroutinefunction(func):
            # Async function
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx = Context.get()
                with ctx.push_latency_sensitivity(parsed_sensitivity):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore

        elif inspect.isgeneratorfunction(func):
            # Generator function
            @functools.wraps(func)
            def generator_wrapper(*args: Any, **kwargs: Any):
                ctx = Context.get()
                with ctx.push_latency_sensitivity(parsed_sensitivity):
                    yield from func(*args, **kwargs)

            return generator_wrapper  # type: ignore

        else:
            # Regular sync function
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx = Context.get()
                with ctx.push_latency_sensitivity(parsed_sensitivity):
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator
