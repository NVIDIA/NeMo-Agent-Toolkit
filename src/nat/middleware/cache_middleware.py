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

import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from typing import Literal

from pydantic import Field

from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import CallNext
from nat.middleware.function_middleware import CallNextStream
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.function_middleware import FunctionMiddlewareContext

logger = logging.getLogger(__name__)


class CacheMiddleware(FunctionMiddleware):
    """Cache middleware that memoizes function outputs based on input similarity.

    This middleware demonstrates the four-phase middleware pattern:

    1. **Preprocess**: Serialize input and check cache for similar entries
    2. **Call Next**: Delegate to next middleware/function if cache miss
    3. **Postprocess**: Store the result in cache for future use
    4. **Continue**: Return the result (from cache or fresh)

    The cache serializes function inputs to strings and performs similarity
    matching against previously seen inputs. If a similar input is found above
    the configured threshold, it returns the cached output without calling the
    next middleware or function.

    Args:
        enabled_mode: Either "always" to always cache, or "eval" to only
            cache when Context.is_evaluating is True.
        similarity_threshold: Float between 0 and 1. If 1.0, performs
            exact string matching. Otherwise uses difflib for similarity
            computation.
    """

    def __init__(self, *, enabled_mode: str, similarity_threshold: float) -> None:
        """Initialize the cache middleware.

        Args:
            enabled_mode: Either "always" or "eval". If "eval", only caches
                when Context.is_evaluating is True.
            similarity_threshold: Similarity threshold between 0 and 1.
                If 1.0, performs exact matching. Otherwise uses fuzzy matching.
        """
        super().__init__(is_final=True)
        self._enabled_mode = enabled_mode
        self._similarity_threshold = similarity_threshold
        self._cache: dict[str, Any] = {}

    def _should_cache(self) -> bool:
        """Check if caching should be enabled based on the current context."""
        if self._enabled_mode == "always":
            return True

        # Get the current context and check if we're in evaluation mode
        try:
            context_state = ContextState.get()
            context = Context(context_state)
            return context.is_evaluating
        except Exception:
            logger.warning("Failed to get context for cache decision", exc_info=True)
            return False

    def _serialize_input(self, value: Any) -> str | None:
        """Serialize the input value to a string for caching.

        Args:
            value: The input value to serialize.

        Returns:
            String representation of the input, or None if serialization
            fails.
        """
        try:
            # Try JSON serialization first for best results
            return json.dumps(value, sort_keys=True, default=str)
        except Exception:
            logger.debug("Failed to serialize input for caching", exc_info=True)
            return None

    def _find_similar_key(self, input_str: str) -> str | None:
        """Find a cached key that is similar to the input string.

        Args:
            input_str: The serialized input string to match.

        Returns:
            The most similar cached key if above threshold, None otherwise.
        """
        if self._similarity_threshold == 1.0:
            # Exact matching - fast path
            return input_str if input_str in self._cache else None

        # Fuzzy matching using difflib
        import difflib

        best_match = None
        best_ratio = 0.0

        for cached_key in self._cache:
            # Use SequenceMatcher for similarity computation
            matcher = difflib.SequenceMatcher(None, input_str, cached_key)
            ratio = matcher.ratio()

            if ratio >= self._similarity_threshold and ratio > best_ratio:
                best_ratio = ratio
                best_match = cached_key

        return best_match

    async def function_middleware_invoke(self, value: Any, call_next: CallNext,
                                         context: FunctionMiddlewareContext) -> Any:
        """Cache middleware for single-output invocations.

        Implements the four-phase middleware pattern:

        1. **Preprocess**: Check if caching is enabled and serialize input
        2. **Call Next**: Delegate to next middleware/function if cache miss
        3. **Postprocess**: Store the result in cache
        4. **Continue**: Return the result (cached or fresh)

        Args:
            value: The input value to process
            call_next: Callable to invoke the next middleware or function
            context: Metadata about the function being wrapped

        Returns:
            The cached output if found, otherwise the fresh output
        """
        # Phase 1: Preprocess - check if caching should be enabled
        if not self._should_cache():
            return await call_next(value)

        # Phase 1: Preprocess - serialize the input
        input_str = self._serialize_input(value)
        if input_str is None:
            # Can't serialize, pass through to next middleware/function
            logger.debug("Could not serialize input for function %s, bypassing cache", context.name)
            return await call_next(value)

        # Phase 1: Preprocess - look for a similar cached input
        similar_key = self._find_similar_key(input_str)
        if similar_key is not None:
            # Cache hit - short-circuit and return cached output
            logger.debug("Cache hit for function %s with similarity %.2f",
                         context.name,
                         1.0 if similar_key == input_str else self._similarity_threshold)
            # Phase 4: Continue - return cached result
            return self._cache[similar_key]

        # Phase 2: Call next - no cache hit, call next middleware/function
        logger.debug("Cache miss for function %s", context.name)
        result = await call_next(value)

        # Phase 3: Postprocess - cache the result for future use
        self._cache[input_str] = result
        logger.debug("Cached result for function %s", context.name)

        # Phase 4: Continue - return the fresh result
        return result

    async def function_middleware_stream(self,
                                         value: Any,
                                         call_next: CallNextStream,
                                         context: FunctionMiddlewareContext) -> AsyncIterator[Any]:
        """Cache middleware for streaming invocations - bypasses caching.

        Streaming results are not cached as they would need to be buffered
        entirely in memory, which would defeat the purpose of streaming.

        This method demonstrates the middleware pattern for streams:

        1. **Preprocess**: Log that we're bypassing cache
        2. **Call Next**: Get stream from next middleware/function
        3. **Process Chunks**: Yield each chunk as it arrives
        4. **Continue**: Complete the stream

        Args:
            value: The input value to process
            call_next: Callable to invoke the next middleware or function stream
            context: Metadata about the function being wrapped

        Yields:
            Chunks from the stream (unmodified)
        """
        # Phase 1: Preprocess - log that we're bypassing cache for streams
        logger.debug("Streaming call for function %s, bypassing cache", context.name)

        # Phase 2-3: Call next and process chunks - yield chunks as they arrive
        async for chunk in call_next(value):
            yield chunk

        # Phase 4: Continue - stream is complete (implicit)


class CacheMiddlewareConfig(FunctionMiddlewareBaseConfig, name="cache"):
    """Configuration for cache middleware.

    The cache middleware memoizes function outputs based on input similarity,
    with support for both exact and fuzzy matching.

    Args:
        enabled_mode: Controls when caching is active:
            - "always": Cache is always enabled
            - "eval": Cache only active when Context.is_evaluating is True
        similarity_threshold: Float between 0 and 1 for input matching:
            - 1.0: Exact string matching (fastest)
            - < 1.0: Fuzzy matching using difflib similarity
    """

    enabled_mode: Literal["always", "eval"] = Field(
        default="eval", description="When caching is enabled: 'always' or 'eval' (only during evaluation)")
    similarity_threshold: float = Field(default=1.0,
                                        ge=0.0,
                                        le=1.0,
                                        description="Similarity threshold between 0 and 1. Use 1.0 for exact matching")


__all__ = ["CacheMiddleware", "CacheMiddlewareConfig"]
