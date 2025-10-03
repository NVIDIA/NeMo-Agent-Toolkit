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
"""Deprecation utilities.

This module provides helpers to standardize deprecation signaling across the
codebase:

- ``issue_deprecation_warning``: Builds and emits a single deprecation message
  per function using the standard logging pipeline.
- ``deprecated``: A decorator that wraps sync/async functions and generators to
  log a one-time deprecation message upon first use. It supports optional
  metadata, a planned removal version, a suggested replacement, and an
  optional feature name label.

Messages are emitted via ``logging.getLogger(__name__).warning`` (not
``warnings.warn``) so they appear in normal application logs and respect global
logging configuration. Each unique function logs at most once per process.
"""

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

_warning_issued = set()

# Type variables for overloads
F = TypeVar('F', bound=Callable[..., Any])


def issue_deprecation_warning(function_name: str,
                              removal_version: str | None = None,
                              replacement: str | None = None,
                              reason: str | None = None,
                              feature_name: str | None = None,
                              metadata: dict[str, Any] | None = None) -> None:
    """
    Log a deprecation warning message for the function.

    A warning is emitted only once per function. When a ``metadata`` dict
    is supplied, it is appended to the log entry to provide extra context
    (e.g., version, author, feature flag).

    Args:
        function_name: The name of the deprecated function
        removal_version: The version when the function will be removed
        replacement: What to use instead of this function
        reason: Why the function is being deprecated
        feature_name: Optional name of the feature that is deprecated
        metadata: Optional dictionary of metadata to log with the warning
    """
    if function_name not in _warning_issued:
        # Build the deprecation message
        if feature_name:
            warning_message = f"{feature_name} is deprecated"
        else:
            warning_message = f"Function {function_name} is deprecated"

        if removal_version:
            warning_message += f" and will be removed in version {removal_version}"
        else:
            warning_message += " and will be removed in a future release"

        warning_message += "."

        if reason:
            warning_message += f" Reason: {reason}."

        if replacement:
            warning_message += f" Use '{replacement}' instead."

        if metadata:
            warning_message += f" | Metadata: {metadata}"

        # Issue warning and save function name to avoid duplicate warnings
        logger.warning(warning_message)
        _warning_issued.add(function_name)


# Overloads for different function types
@overload
def deprecated(func: F,
               *,
               removal_version: str | None = None,
               replacement: str | None = None,
               reason: str | None = None,
               feature_name: str | None = None,
               metadata: dict[str, Any] | None = None) -> F:
    """Overload for direct decorator usage (when called without parentheses)."""
    ...


@overload
def deprecated(*,
               removal_version: str | None = None,
               replacement: str | None = None,
               reason: str | None = None,
               feature_name: str | None = None,
               metadata: dict[str, Any] | None = None) -> Callable[[F], F]:
    """Overload for decorator factory usage (when called with parentheses)."""
    ...


def deprecated(func: Any = None,
               *,
               removal_version: str | None = None,
               replacement: str | None = None,
               reason: str | None = None,
               feature_name: str | None = None,
               metadata: dict[str, Any] | None = None) -> Any:
    """
    Decorator that can wrap any type of function (sync, async, generator,
    async generator) and logs a deprecation warning.

    Args:
        func: The function to be decorated.
        removal_version: The version when the function will be removed
        replacement: What to use instead of this function
        reason: Why the function is being deprecated
        feature_name: Optional name of the feature that is deprecated. If provided, the warning will be
        prefixed with "The <feature_name> feature is deprecated".
        metadata: Optional dictionary of metadata to log with the warning. This can include information
        like version, author, etc. If provided, the metadata will be
        logged alongside the deprecation warning.
    """
    function_name: str = f"{func.__module__}.{func.__qualname__}" if func else "<unknown_function>"

    # If called as @deprecated(...) but not immediately passed a function
    if func is None:

        def decorator_wrapper(actual_func):
            return deprecated(actual_func,
                              removal_version=removal_version,
                              replacement=replacement,
                              reason=reason,
                              feature_name=feature_name,
                              metadata=metadata)

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
            issue_deprecation_warning(function_name, removal_version, replacement, reason, feature_name, metadata)
            async for item in func(*args, **kwargs):
                yield item  # yield the original item

        return async_gen_wrapper

    if inspect.iscoroutinefunction(func):
        # ---------------------
        # ASYNC FUNCTION
        # ---------------------
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            issue_deprecation_warning(function_name, removal_version, replacement, reason, feature_name, metadata)
            result = await func(*args, **kwargs)
            return result

        return async_wrapper

    if inspect.isgeneratorfunction(func):
        # ---------------------
        # SYNC GENERATOR
        # ---------------------
        @functools.wraps(func)
        def sync_gen_wrapper(*args, **kwargs) -> Generator[Any, Any, Any]:
            issue_deprecation_warning(function_name, removal_version, replacement, reason, feature_name, metadata)
            yield from func(*args, **kwargs)  # yield the original item

        return sync_gen_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        issue_deprecation_warning(function_name, removal_version, replacement, reason, feature_name, metadata)
        result = func(*args, **kwargs)
        return result

    return sync_wrapper
