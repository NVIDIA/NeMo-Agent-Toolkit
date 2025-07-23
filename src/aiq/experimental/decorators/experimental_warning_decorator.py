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
from typing import Any

logger = logging.getLogger(__name__)

WARNING_MESSAGE = "This function is experimental and the API may change in future releases."

warning_issued = set()


def issue_warning(function_name: str, metadata: dict[str, Any] | None = None):
    """
    Log a warning message that the function is experimental.

    A warning is emitted only once per function.  When a ``metadata`` dict
    is supplied, it is appended to the log entry to provide extra context
    (e.g., version, author, feature flag).
    """
    if function_name not in warning_issued:
        if metadata:
            logger.warning(
                "%s Function: %s | Metadata: %s",
                WARNING_MESSAGE,
                function_name,
                metadata,
            )
        else:
            logger.warning("%s Function: %s", WARNING_MESSAGE, function_name)
        warning_issued.add(function_name)


def aiq_experimental(func: Any = None, *, metadata: dict[str, Any] | None = None):
    """
    Decorator that can wrap any type of function (sync, async, generator,
    async generator) and logs a warning that the function is experimental.

    Args:
        func: The function to be decorated.
        metadata: Optional dictionary of metadata to log with the warning. This can include information
        like version, author, etc. If provided, the metadata will be
        logged alongside the experimental warning.
    """
    function_name: str = f"{func.__module__}.{func.__qualname__}" if func else "<unknown_function>"

    # If called as @track_function(...) but not immediately passed a function
    if func is None:

        def decorator_wrapper(actual_func):
            return aiq_experimental(actual_func, metadata=metadata)

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
        async def async_gen_wrapper(*args, **kwargs):
            issue_warning(function_name, metadata)
            async for item in func(*args, **kwargs):
                yield item  # yield the original item

        return async_gen_wrapper

    if inspect.iscoroutinefunction(func):
        # ---------------------
        # ASYNC FUNCTION
        # ---------------------
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            issue_warning(function_name, metadata)
            result = await func(*args, **kwargs)
            return result

        return async_wrapper

    if inspect.isgeneratorfunction(func):
        # ---------------------
        # SYNC GENERATOR
        # ---------------------
        @functools.wraps(func)
        def sync_gen_wrapper(*args, **kwargs):
            issue_warning(function_name, metadata)
            for item in func(*args, **kwargs):
                yield item  # yield the original item

        return sync_gen_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        issue_warning(function_name, metadata)
        result = func(*args, **kwargs)
        return result

    return sync_wrapper
