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
with integer latency sensitivity levels. The sensitivity propagates through
the call stack with max-based merging, where higher values take precedence.

Use cases:
- LLM routing: Direct high-sensitivity requests to low-latency backends
- Execution optimization: Adjust timeouts, batch sizes based on sensitivity
- Observability: Track which parts of workflows have strict latency requirements

Example:
    Basic usage with integers::

        from nat.plugins.profiler.decorators.latency import latency_sensitive

        @latency_sensitive(3)
        async def critical_llm_call():
            return await llm.generate()

    Using integer values::

        @latency_sensitive(1)
        def background_task():
            pass

    Reading current sensitivity::

        from nat.builder.context import Context

        def my_function():
            sensitivity = Context.get().latency_sensitivity
            if sensitivity >= 3:
                # Use fast path
                pass
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any
from typing import TypeVar

# Type variable for preserving function signature
F = TypeVar("F", bound=Callable[..., Any])


def latency_sensitive(sensitivity: int) -> Callable[[F], F]:
    """
    Decorator to mark a function with a latency sensitivity level.

    The sensitivity is pushed onto the context stack for the duration of the
    function execution. The effective sensitivity is the maximum value across
    all pushed levels.

    Args:
        sensitivity: Latency sensitivity level as an integer (e.g. 1=low, 2=medium, 3=high)

    Returns:
        Decorated function that pushes sensitivity onto context stack

    Raises:
        TypeError: If sensitivity is not an int

    Example:
        from nat.plugins.profiler.decorators.latency import latency_sensitive
        >>> from nat.builder.context import Context
        >>>
        >>> @latency_sensitive(3)
        ... def critical_function():
        ...     return Context.get().latency_sensitivity
        >>>
        >>> @latency_sensitive(1)
        ... async def background_task():
        ...     return await do_work()
    """
    # Validate at decoration time
    if not isinstance(sensitivity, int):
        raise TypeError(f"sensitivity must be an int, got {type(sensitivity).__name__}")

    def decorator(func: F) -> F:
        # Import here to avoid circular dependency
        from nat.builder.context import Context

        if inspect.isasyncgenfunction(func):
            # Async generator function
            @functools.wraps(func)
            async def async_gen_wrapper(*args: Any, **kwargs: Any):
                ctx = Context.get()
                with ctx.push_latency_sensitivity(sensitivity):
                    async for item in func(*args, **kwargs):
                        yield item

            return async_gen_wrapper  # type: ignore

        elif inspect.iscoroutinefunction(func):
            # Async function
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx = Context.get()
                with ctx.push_latency_sensitivity(sensitivity):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore

        elif inspect.isgeneratorfunction(func):
            # Generator function
            @functools.wraps(func)
            def generator_wrapper(*args: Any, **kwargs: Any):
                ctx = Context.get()
                with ctx.push_latency_sensitivity(sensitivity):
                    yield from func(*args, **kwargs)

            return generator_wrapper  # type: ignore

        else:
            # Regular sync function
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx = Context.get()
                with ctx.push_latency_sensitivity(sensitivity):
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator
