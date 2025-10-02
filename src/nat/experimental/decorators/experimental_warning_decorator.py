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

import functools
import inspect
import logging
from collections.abc import AsyncGenerator
from collections.abc import Callable
from collections.abc import Generator
from typing import Any
from typing import TypeVar
from typing import overload

logger = logging.getLogger(__name__)

BASE_WARNING_MESSAGE = ("is experimental and the API may change in future releases. "
                        "Future versions may introduce breaking changes without notice.")

_warning_issued = set()

# Type variables for overloads
F = TypeVar('F', bound=Callable[..., Any])


def issue_experimental_warning(function_name: str,
                               feature_name: str | None = None,
                               metadata: dict[str, Any] | None = None):
    """
    Log a warning message that the function is experimental.

    A warning is emitted only once per function.  When a ``metadata`` dict
    is supplied, it is appended to the log entry to provide extra context
    (e.g., version, author, feature flag).
    """
    if function_name not in _warning_issued:
        if (feature_name):
            warning_message = f"The {feature_name} feature {BASE_WARNING_MESSAGE}"
        else:
            warning_message = f"This function {BASE_WARNING_MESSAGE}"

        warning_message += f" Function: {function_name}"

        if (metadata):
            warning_message += f" | Metadata: {metadata}"

        # Issue warning and save function name to avoid duplicate warnings
        logger.warning(warning_message)

        _warning_issued.add(function_name)


# Overloads for different function types
@overload
def experimental(func: F, *, feature_name: str | None = None, metadata: dict[str, Any] | None = None) -> F:
    """Overload for when a function is passed directly."""
    ...


@overload
def experimental(*, feature_name: str | None = None, metadata: dict[str, Any] | None = None) -> Callable[[F], F]:
    """Overload for decorator factory usage (when called with parentheses)."""
    ...


def experimental(func: Any = None, *, feature_name: str | None = None, metadata: dict[str, Any] | None = None) -> Any:
    """
    Decorator that can wrap any type of function (sync, async, generator,
    async generator) and logs a warning that the function is experimental.

    Args:
        func: The function to be decorated.
        feature_name: Optional name of the feature that is experimental. If provided, the warning will be
        prefixed with "The <feature_name> feature is experimental".
        metadata: Optional dictionary of metadata to log with the warning. This can include information
        like version, author, etc. If provided, the metadata will be
        logged alongside the experimental warning.
    """
    function_name: str = f"{func.__module__}.{func.__qualname__}" if func else "<unknown_function>"

    # If called as @track_function(...) but not immediately passed a function
    if func is None:

        def decorator_wrapper(actual_func):
            return experimental(actual_func, feature_name=feature_name, metadata=metadata)

        return decorator_wrapper

    # --- Validate metadata ---
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict[str, Any].")
        if any(not isinstance(k, str) for k in metadata.keys()):
            raise TypeError("All metadata keys must be strings.")

    # --- Now detect the function type and wrap accordingly ---
    if inspect.isasyncgenfunction(func):
        # ---------------------
        # ASYNC GENERATOR
        # ---------------------

        @functools.wraps(func)
        async def async_gen_wrapper(*args, **kwargs) -> AsyncGenerator[Any, Any]:
            issue_experimental_warning(function_name, feature_name, metadata)
            async for item in func(*args, **kwargs):
                yield item  # yield the original item

        return async_gen_wrapper

    if inspect.iscoroutinefunction(func):
        # ---------------------
        # ASYNC FUNCTION
        # ---------------------
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            issue_experimental_warning(function_name, feature_name, metadata)
            result = await func(*args, **kwargs)
            return result

        return async_wrapper

    if inspect.isgeneratorfunction(func):
        # ---------------------
        # SYNC GENERATOR
        # ---------------------
        @functools.wraps(func)
        def sync_gen_wrapper(*args, **kwargs) -> Generator[Any, Any, Any]:
            issue_experimental_warning(function_name, feature_name, metadata)
            yield from func(*args, **kwargs)  # yield the original item

        return sync_gen_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        issue_experimental_warning(function_name, feature_name, metadata)
        result = func(*args, **kwargs)
        return result

    return sync_wrapper


# Compatibility aliases with previous releases
aiq_experimental = experimental
