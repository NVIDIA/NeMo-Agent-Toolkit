# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache intercept for function memoization with similarity matching.

This module provides a cache intercept that can memoize function calls based on
input similarity. It supports exact matching for maximum performance and fuzzy
matching using Python's built-in difflib for similarity computation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.intercepts.function_intercept import FunctionIntercept
from nat.intercepts.function_intercept import FunctionInterceptContext
from nat.intercepts.function_intercept import SingleInvokeCallable
from nat.intercepts.function_intercept import StreamInvokeCallable

logger = logging.getLogger(__name__)


class CacheIntercept(FunctionIntercept):
    """Cache function outputs based on input similarity.

    This intercept serializes function inputs to strings and performs
    similarity matching against previously seen inputs. If a similar input
    is found above the configured threshold, it returns the cached output
    instead of calling the function.

    Args:
        enabled_mode: Either "always" to always cache, or "eval" to only
            cache when Context.is_evaluating is True.
        similarity_threshold: Float between 0 and 1. If 1.0, performs
            exact string matching. Otherwise uses difflib for similarity
            computation.
    """

    def __init__(self, *, enabled_mode: str = "eval", similarity_threshold: float = 1.0) -> None:
        """Initialize the cache intercept.

        Args:
            enabled_mode: Either "always" or "eval". If "eval", only caches
                when Context.is_evaluating is True.
            similarity_threshold: Similarity threshold between 0 and 1.
                If 1.0, performs exact matching. Otherwise uses fuzzy matching.

        Raises:
            ValueError: If enabled_mode is not "always" or "eval", or if
                similarity_threshold is not between 0 and 1.
        """
        super().__init__(is_final=True)

        if enabled_mode not in ("always", "eval"):
            raise ValueError(f"enabled_mode must be 'always' or 'eval', "
                             f"got '{enabled_mode}'")

        if not 0 <= similarity_threshold <= 1:
            raise ValueError(f"similarity_threshold must be between 0 and 1, "
                             f"got {similarity_threshold}")

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

    async def intercept_invoke(self, value: Any, next_call: SingleInvokeCallable,
                               context: FunctionInterceptContext) -> Any:
        """Intercept single-output invocations with caching logic.

        This method:
        1. Checks if caching should be enabled
        2. Serializes the input
        3. Looks for similar cached inputs
        4. Returns cached output if found, otherwise delegates to function
        5. Caches the output for future use
        """
        # Check if we should cache
        if not self._should_cache():
            return await next_call(value)

        # Try to serialize the input
        input_str = self._serialize_input(value)
        if input_str is None:
            # Can't serialize, pass through to function
            logger.debug("Could not serialize input for function %s, bypassing cache", context.name)
            return await next_call(value)

        # Look for a similar cached input
        similar_key = self._find_similar_key(input_str)
        if similar_key is not None:
            # Found a match, return the cached output
            logger.debug("Cache hit for function %s with similarity %.2f",
                         context.name,
                         1.0 if similar_key == input_str else self._similarity_threshold)
            return self._cache[similar_key]

        # No match found, call the function
        logger.debug("Cache miss for function %s", context.name)
        result = await next_call(value)

        # Cache the result
        self._cache[input_str] = result
        logger.debug("Cached result for function %s", context.name)

        return result

    async def intercept_stream(self, value: Any, next_call: StreamInvokeCallable,
                               context: FunctionInterceptContext) -> AsyncIterator[Any]:
        """Intercept streaming invocations - always delegates to function.

        Streaming results are not cached as they would need to be buffered
        entirely in memory, defeating the purpose of streaming.
        """
        logger.debug("Streaming call for function %s, bypassing cache", context.name)
        async for chunk in next_call(value):
            yield chunk


__all__ = ["CacheIntercept"]
