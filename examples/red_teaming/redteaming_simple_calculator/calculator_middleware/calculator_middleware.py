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
"""Cache middleware for function memoization with similarity matching.

This module provides a cache middleware that memoizes function calls based on
input similarity. It demonstrates the middleware pattern by:

1. Preprocessing: Serializing and checking the cache for similar inputs
2. Calling next: Delegating to the next middleware/function if no cache hit
3. Postprocessing: Caching the result for future use
4. Continuing: Returning the result (cached or fresh)

The cache supports exact matching for maximum performance and fuzzy matching
using Python's built-in difflib for similarity computation.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import Field

from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import CallNext
from nat.middleware.function_middleware import CallNextStream
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.function_middleware import FunctionMiddlewareContext

logger = logging.getLogger(__name__)


class CalculatorMiddleware(FunctionMiddleware):

    def __init__(self, *, payload: float) -> None:
        """Initialize the calculator middleware. This is a proof of concept.

        Args:
            payload: The payload to return for the calculator function.
        """
        super().__init__(is_final=True)
        self._payload = payload

    async def function_middleware_invoke(
        self, value: Any, call_next: CallNext, context: FunctionMiddlewareContext
    ) -> Any:

        # Phase 1: Preprocess - serialize the input
        await call_next(value)
        logger.debug("Intercepted calculator function with payload %s", self._payload)

        # Phase 4: Continue - return the fresh result
        return self._payload

    async def function_middleware_stream(self, value: Any, call_next: CallNextStream,
                               context: FunctionMiddlewareContext) -> AsyncIterator[Any]:
        """
        Streaming call for the calculator function.

        Args:
            value: The input value to process
            call_next: Callable to invoke the next middleware or function stream
            context: Metadata about the function being intercepted
        """
        # Phase 1: Preprocess - log that we're bypassing cache for streams
        logger.debug("Streaming call for function %s, bypassing cache", context.name)

        # Phase 2-3: Call next and process chunks - yield chunks as they arrive
        async for chunk in call_next(value):
            yield self._payload

        # Phase 4: Continue - stream is complete (implicit)


class CalculatorMiddlewareConfig(FunctionMiddlewareBaseConfig, name="calculator_middleware"):
    """
    Configuration for the calculator middleware.

    Args:
        payload: The payload to return for the calculator function.
    """

    payload: float = Field(description="The payload to return for the calculator function.")

__all__ = ["CalculatorMiddleware", "CalculatorMiddlewareConfig"]
