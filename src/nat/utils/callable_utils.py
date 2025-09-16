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

import inspect
from collections.abc import Callable
from typing import Any


async def ainvoke_any(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute any type of callable and return the result.

    Handles synchronous functions, asynchronous functions, generators,
    and async generators uniformly, returning the final result value.

    Args:
        func (Callable[..., Any]): The function to execute (sync/async function, generator, etc.)

    Returns:
        Any: The result of executing the callable
    """
    # Execute the function
    result_value = func(*args, **kwargs)

    # Handle different return types
    if inspect.iscoroutine(result_value):
        # Async function - await the coroutine
        return await result_value

    if inspect.isgenerator(result_value):
        # Sync generator - consume until StopIteration and get return value
        try:
            while True:
                next(result_value)
        except StopIteration as e:
            # Return the generator's return value, or None if not provided
            return e.value

    if inspect.isasyncgen(result_value):
        # Async generator - consume all values and return the last one
        last_value = None
        async for value in result_value:
            last_value = value
        return last_value

    # Direct value from sync function (most common case)
    return result_value


def is_async_callable(func: Callable[..., Any]) -> bool:
    """Check if a function is async (coroutine function or async generator function).

    Args:
        func (Callable[..., Any]): The function to check

    Returns:
        bool: True if the function is async, False otherwise
    """
    return inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)
