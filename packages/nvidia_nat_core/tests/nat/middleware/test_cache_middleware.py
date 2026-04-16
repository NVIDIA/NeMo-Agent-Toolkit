# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the CacheMiddleware middleware functionality."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.data_models.runtime_enum import RuntimeTypeEnum
from nat.middleware.cache.cache_middleware import CacheMiddleware
from nat.middleware.middleware import FunctionMiddlewareContext


class _TestInput(BaseModel):
    """Test input model."""
    value: str
    number: int


class _TestOutput(BaseModel):
    """Test output model."""
    result: str


@pytest.fixture
def middleware_context():
    """Create a test FunctionMiddlewareContext (static metadata only)."""
    return FunctionMiddlewareContext(name="test_function",
                                     config=MagicMock(),
                                     description="Test function",
                                     input_schema=_TestInput,
                                     single_output_schema=_TestOutput,
                                     stream_output_schema=None)


class TestCacheMiddlewareInitialization:
    """Test CacheMiddleware initialization and configuration."""

    def test_default_initialization(self):
        """Test default initialization with required parameters."""
        middleware = CacheMiddleware(enabled_mode="eval", similarity_threshold=1.0)
        # Check internal attributes
        assert hasattr(middleware, '_enabled_mode')
        assert hasattr(middleware, '_similarity_threshold')
        assert middleware.is_final is True

    def test_custom_initialization(self):
        """Test custom initialization."""
        # Use 0.9 (above the enforced minimum) to exercise non-default fuzzy mode.
        middleware = CacheMiddleware(enabled_mode="always", similarity_threshold=0.9)
        # Check attributes are set
        assert hasattr(middleware, '_enabled_mode')
        assert hasattr(middleware, '_similarity_threshold')


class TestCacheMiddlewareCaching:
    """Test caching behavior."""

    async def test_exact_match_caching(self, middleware_context):
        """Test exact match caching with similarity_threshold=1.0."""
        middleware = CacheMiddleware(enabled_mode="always", similarity_threshold=1.0)

        # Mock the next call
        call_count = 0

        async def mock_next_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            _val = args[0]
            return _TestOutput(result=f"Result for {_val['value']}")

        # First call - should call the function
        input1 = {"value": "test", "number": 42}
        result1 = await middleware.function_middleware_invoke(input1,
                                                              call_next=mock_next_call,
                                                              context=middleware_context)
        assert call_count == 1
        assert result1.result == "Result for test"

        # Second call with same input - should use cache
        result2 = await middleware.function_middleware_invoke(input1,
                                                              call_next=mock_next_call,
                                                              context=middleware_context)
        assert call_count == 1  # No additional call
        assert result2.result == "Result for test"

        # Third call with different input - should call function
        input2 = {"value": "test", "number": 43}  # Different number
        result3 = await middleware.function_middleware_invoke(input2,
                                                              call_next=mock_next_call,
                                                              context=middleware_context)
        assert call_count == 2
        assert result3.result == "Result for test"

    async def test_fuzzy_match_caching(self, middleware_context):
        """Test fuzzy matching with similarity_threshold < 1.0.

        Uses 0.9 (above the enforced minimum) — 0.8 is no longer a valid
        threshold after the cache-poisoning hardening.
        """
        middleware = CacheMiddleware(enabled_mode="always", similarity_threshold=0.9)

        call_count = 0

        async def mock_next_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _TestOutput(result=f"Result {call_count}")

        # First call
        input1 = {"value": "hello world", "number": 42}
        result1 = await middleware.function_middleware_invoke(input1,
                                                              call_next=mock_next_call,
                                                              context=middleware_context)
        assert call_count == 1
        assert result1.result == "Result 1"

        # Second call with similar input - should use cache
        input2 = {"value": "hello world!", "number": 42}
        result2 = await middleware.function_middleware_invoke(input2,
                                                              call_next=mock_next_call,
                                                              context=middleware_context)
        assert call_count == 1  # No additional call due to similarity
        assert result2.result == "Result 1"

        # Third call with very different input - should call function
        input3 = {"value": "goodbye universe", "number": 99}
        result3 = await middleware.function_middleware_invoke(input3,
                                                              call_next=mock_next_call,
                                                              context=middleware_context)
        assert call_count == 2
        assert result3.result == "Result 2"

    async def test_eval_mode_caching(self, middleware_context):
        """Test caching only works in eval mode when configured."""
        middleware = CacheMiddleware(enabled_mode="eval", similarity_threshold=1.0)

        call_count = 0

        async def mock_next_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _TestOutput(result=f"Result {call_count}")

        # Mock ContextState to control is_evaluating
        mock_ctx_cls = 'nat.middleware.cache.cache_middleware.ContextState'
        with patch(mock_ctx_cls) as mock_context_state:
            mock_state = MagicMock()
            mock_context_state.get.return_value = mock_state

            # First, test when NOT evaluating
            mock_state.runtime_type.get.return_value = RuntimeTypeEnum.RUN_OR_SERVE

            input1 = {"value": "test", "number": 42}
            await middleware.function_middleware_invoke(input1, call_next=mock_next_call, context=middleware_context)
            assert call_count == 1

            # Same input again - should NOT use cache
            await middleware.function_middleware_invoke(input1, call_next=mock_next_call, context=middleware_context)
            assert call_count == 2  # Called again

            # Now test when evaluating
            mock_state.runtime_type.get.return_value = RuntimeTypeEnum.EVALUATE

            # Same input - should call function (no cache before)
            await middleware.function_middleware_invoke(input1, call_next=mock_next_call, context=middleware_context)
            assert call_count == 3

            # Same input again - should use cache now
            await middleware.function_middleware_invoke(input1, call_next=mock_next_call, context=middleware_context)
            assert call_count == 3  # No additional call

    async def test_serialization_failure(self, middleware_context):
        """Test behavior when input serialization fails."""
        middleware = CacheMiddleware(enabled_mode="always", similarity_threshold=1.0)

        call_count = 0

        async def mock_next_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _TestOutput(result="Result")

        # Create an object that can't be serialized
        class UnserializableObject:

            def __init__(self):
                self.circular_ref = self

        # Mock json.dumps to raise an exception
        with patch('json.dumps', side_effect=Exception("Cannot serialize")):
            input_obj = UnserializableObject()
            await middleware.function_middleware_invoke(input_obj, call_next=mock_next_call, context=middleware_context)
            assert call_count == 1

            # Try again - should call function again (no caching)
            await middleware.function_middleware_invoke(input_obj, call_next=mock_next_call, context=middleware_context)
            assert call_count == 2


class TestCacheMiddlewareStreaming:
    """Test streaming behavior."""

    async def test_streaming_bypass(self, middleware_context):
        """Test that streaming always bypasses cache."""
        middleware = CacheMiddleware(enabled_mode="always", similarity_threshold=1.0)

        call_count = 0

        async def mock_stream_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            for i in range(3):
                yield f"Chunk {i}"

        # First streaming call
        input1 = {"value": "test", "number": 42}
        chunks1 = []
        async for chunk in middleware.function_middleware_stream(input1,
                                                                 call_next=mock_stream_call,
                                                                 context=middleware_context):
            chunks1.append(chunk)
        assert call_count == 1
        assert chunks1 == ["Chunk 0", "Chunk 1", "Chunk 2"]

        # Second streaming call with same input - should call again
        chunks2 = []
        async for chunk in middleware.function_middleware_stream(input1,
                                                                 call_next=mock_stream_call,
                                                                 context=middleware_context):
            chunks2.append(chunk)
        assert call_count == 2  # Function called again
        assert chunks2 == ["Chunk 0", "Chunk 1", "Chunk 2"]


class TestCacheMiddlewareEdgeCases:
    """Test edge cases and error handling."""

    async def test_context_retrieval_failure(self, middleware_context):
        """Test behavior when context retrieval fails in eval mode."""
        middleware = CacheMiddleware(enabled_mode="eval", similarity_threshold=1.0)

        call_count = 0

        async def mock_next_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _TestOutput(result="Result")

        # Mock ContextState.get to raise an exception
        mock_ctx_cls = 'nat.middleware.cache.cache_middleware.ContextState.get'
        with patch(mock_ctx_cls, side_effect=Exception("Context error")):
            input1 = {"value": "test", "number": 42}
            await middleware.function_middleware_invoke(input1, call_next=mock_next_call, context=middleware_context)
            assert call_count == 1  # Should fall back to calling function

    def test_similarity_computation_for_different_thresholds(self):
        """Test similarity computation for different thresholds."""
        # This is more of a unit test for the similarity logic.
        # Uses 0.9 (above the enforced minimum) to exercise fuzzy matching
        # without enabling cache-poisoning-prone low thresholds.
        middleware = CacheMiddleware(enabled_mode="always", similarity_threshold=0.9)

        # Directly test internal methods
        # Add a cached entry
        test_key = "hello world"
        middleware._cache[test_key] = "cached_result"  # noqa

        # Test various similarity levels
        # Exact match
        assert middleware._find_similar_key(test_key) == test_key  # noqa
        # Very similar (one char shorter, ~0.95 ratio)
        assert middleware._find_similar_key("hello worl") == test_key  # noqa
        # Too different - use a completely different string
        assert middleware._find_similar_key("xyz123abc") is None  # noqa

    async def test_multiple_similar_entries(self, middleware_context):
        """Test behavior with multiple similar cached entries.

        Uses 0.85 (the enforced minimum) instead of the original 0.7 —
        below 0.85 is now rejected as a cache-poisoning risk.
        """
        middleware = CacheMiddleware(enabled_mode="always", similarity_threshold=0.85)

        # Pre-populate cache with similar entries
        key1 = middleware._serialize_input(  # noqa
            {
                "value": "test input 1", "number": 42
            })
        key2 = middleware._serialize_input(  # noqa
            {
                "value": "test input 2", "number": 42
            })
        middleware._cache[key1] = _TestOutput(result="Result 1")  # noqa
        middleware._cache[key2] = _TestOutput(result="Result 2")  # noqa

        async def mock_next_call(*args, **kwargs):
            return _TestOutput(result="New Result")

        # Query with something similar to all
        input_str = {"value": "test input X", "number": 42}
        await middleware.function_middleware_invoke(input_str, call_next=mock_next_call, context=middleware_context)
        # The exact behavior depends on which cached key is most similar


class TestSimilarityThresholdFloor:
    """The constructor must reject similarity thresholds below the safe floor.

    Below ~0.85, crafting an input whose difflib ratio exceeds the threshold
    against a legitimate cached key is trivial (small edits, common prefixes,
    shared structural tokens). Accepting those values silently produces a
    cache where one caller can hijack another caller's response.
    """

    @pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.7, 0.84])
    def test_below_floor_is_rejected(self, threshold):
        with pytest.raises(ValueError, match="outside the safe range"):
            CacheMiddleware(enabled_mode="always", similarity_threshold=threshold)

    @pytest.mark.parametrize("threshold", [0.85, 0.9, 0.95, 1.0])
    def test_at_or_above_floor_is_allowed(self, threshold):
        mw = CacheMiddleware(enabled_mode="always", similarity_threshold=threshold)
        assert mw._similarity_threshold == threshold  # noqa: SLF001

    def test_threshold_above_one_is_rejected(self):
        with pytest.raises(ValueError, match="outside the safe range"):
            CacheMiddleware(enabled_mode="always", similarity_threshold=1.5)

    def test_threshold_non_numeric_is_rejected(self):
        with pytest.raises(ValueError, match="must be a number"):
            CacheMiddleware(enabled_mode="always", similarity_threshold="high")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_bool", [True, False])
    def test_threshold_bool_is_rejected(self, bad_bool):
        """`isinstance(True, int)` is True in Python — reject bools explicitly
        so a config with the wrong key type doesn't silently become 1.0 or 0.0."""
        with pytest.raises(ValueError, match="got bool"):
            CacheMiddleware(enabled_mode="always", similarity_threshold=bad_bool)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_threshold_non_finite_is_rejected(self, bad_value):
        """NaN, +inf, -inf must be rejected before the range comparison."""
        with pytest.raises(ValueError, match="must be finite"):
            CacheMiddleware(enabled_mode="always", similarity_threshold=bad_value)


class TestMaxEntriesLruEviction:
    """The cache must bound its size to prevent memory-exhaustion DoS.

    The previous implementation used an unbounded dict; sustained unique
    inputs would grow the cache without limit, eventually crashing the
    process. LRU eviction ensures the cache stays within max_entries.
    """

    async def test_default_max_entries_is_positive(self):
        mw = CacheMiddleware(enabled_mode="always", similarity_threshold=1.0)
        assert mw._max_entries > 0  # noqa: SLF001

    def test_zero_max_entries_is_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            CacheMiddleware(enabled_mode="always", similarity_threshold=1.0, max_entries=0)

    def test_negative_max_entries_is_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            CacheMiddleware(enabled_mode="always", similarity_threshold=1.0, max_entries=-5)

    @pytest.mark.parametrize("bad_bool", [True, False])
    def test_bool_max_entries_is_rejected(self, bad_bool):
        """Same bool-as-int foot-gun protection as similarity_threshold."""
        with pytest.raises(ValueError, match="positive integer"):
            CacheMiddleware(
                enabled_mode="always",
                similarity_threshold=1.0,
                max_entries=bad_bool,  # type: ignore[arg-type]
            )

    async def test_cache_evicts_oldest_when_exceeding_max_entries(self, middleware_context):
        """Insert more unique entries than max_entries; verify size stays bounded."""
        mw = CacheMiddleware(
            enabled_mode="always",
            similarity_threshold=1.0,  # exact match keeps the test deterministic
            max_entries=3,
        )

        call_count = 0

        async def mock_next_call(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            return _TestOutput(result=f"result_{call_count}")

        for i in range(10):
            await mw.function_middleware_invoke(
                {"value": f"unique_input_{i}"},
                call_next=mock_next_call,
                context=middleware_context,
            )

        assert len(mw._cache) == 3  # noqa: SLF001
        # The MOST recent three inserts should be what's left.
        latest_keys = list(mw._cache.keys())  # noqa: SLF001
        for i in range(7, 10):
            assert any(f"unique_input_{i}" in k for k in latest_keys)

    async def test_cache_hit_promotes_entry_to_most_recently_used(self, middleware_context):
        """A cache hit should move the entry to MRU so later evictions spare it."""
        mw = CacheMiddleware(
            enabled_mode="always",
            similarity_threshold=1.0,
            max_entries=3,
        )

        async def mock_next_call(*_args, **_kwargs):
            return _TestOutput(result="r")

        # Fill the cache with A, B, C (A is oldest)
        for key in ("A", "B", "C"):
            await mw.function_middleware_invoke(
                {"value": key}, call_next=mock_next_call, context=middleware_context)

        # Hit A again — should promote A to the MRU end
        await mw.function_middleware_invoke(
            {"value": "A"}, call_next=mock_next_call, context=middleware_context)

        # Now insert D — B (now oldest) should be evicted, not A.
        await mw.function_middleware_invoke(
            {"value": "D"}, call_next=mock_next_call, context=middleware_context)

        keys = "".join(list(mw._cache.keys()))  # noqa: SLF001
        assert '"value": "A"' in keys
        assert '"value": "D"' in keys
        assert '"value": "B"' not in keys
