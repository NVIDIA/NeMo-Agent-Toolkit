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
"""Tests for the CacheIntercept middleware functionality."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.builder.context import Context  # noqa: F401
from nat.builder.context import ContextState  # noqa: F401
from nat.intercepts.cache_intercept import CacheIntercept
from nat.intercepts.function_intercept import FunctionInterceptContext


class TestInput(BaseModel):
    """Test input model."""
    value: str
    number: int


class TestOutput(BaseModel):
    """Test output model."""
    result: str


@pytest.fixture
def intercept_context():
    """Create a test FunctionInterceptContext."""
    return FunctionInterceptContext(name="test_function",
                                    config=MagicMock(),
                                    description="Test function",
                                    input_schema=TestInput,
                                    single_output_schema=TestOutput,
                                    stream_output_schema=None)


class TestCacheInterceptInitialization:
    """Test CacheIntercept initialization and configuration."""

    def test_default_initialization(self):
        """Test default initialization."""
        intercept = CacheIntercept()
        # Check internal attributes
        assert hasattr(intercept, '_enabled_mode')
        assert hasattr(intercept, '_similarity_threshold')
        assert intercept.is_final is True

    def test_custom_initialization(self):
        """Test custom initialization."""
        intercept = CacheIntercept(enabled_mode="always", similarity_threshold=0.8)
        # Check attributes are set
        assert hasattr(intercept, '_enabled_mode')
        assert hasattr(intercept, '_similarity_threshold')

    def test_invalid_enabled_mode(self):
        """Test invalid enabled_mode raises ValueError."""
        with pytest.raises(ValueError, match="enabled_mode must be 'always' or 'eval'"):
            CacheIntercept(enabled_mode="invalid")

    def test_invalid_similarity_threshold(self):
        """Test invalid similarity_threshold raises ValueError."""
        msg = "similarity_threshold must be between 0 and 1"
        with pytest.raises(ValueError, match=msg):
            CacheIntercept(similarity_threshold=1.5)

        with pytest.raises(ValueError, match=msg):
            CacheIntercept(similarity_threshold=-0.1)


class TestCacheInterceptCaching:
    """Test caching behavior."""

    @pytest.mark.asyncio
    async def test_exact_match_caching(self, intercept_context):
        """Test exact match caching with similarity_threshold=1.0."""
        intercept = CacheIntercept(enabled_mode="always", similarity_threshold=1.0)

        # Mock the next call
        call_count = 0

        async def mock_next_call(_val):
            nonlocal call_count
            call_count += 1
            return TestOutput(result=f"Result for {_val['value']}")

        # First call - should call the function
        input1 = {"value": "test", "number": 42}
        result1 = await intercept.intercept_invoke(input1, mock_next_call, intercept_context)
        assert call_count == 1
        assert result1.result == "Result for test"

        # Second call with same input - should use cache
        result2 = await intercept.intercept_invoke(input1, mock_next_call, intercept_context)
        assert call_count == 1  # No additional call
        assert result2.result == "Result for test"

        # Third call with different input - should call function
        input2 = {"value": "test", "number": 43}  # Different number
        result3 = await intercept.intercept_invoke(input2, mock_next_call, intercept_context)
        assert call_count == 2
        assert result3.result == "Result for test"

    @pytest.mark.asyncio
    async def test_fuzzy_match_caching(self, intercept_context):
        """Test fuzzy matching with similarity_threshold < 1.0."""
        intercept = CacheIntercept(enabled_mode="always", similarity_threshold=0.8)

        call_count = 0

        async def mock_next_call(_val):
            nonlocal call_count
            call_count += 1
            return TestOutput(result=f"Result {call_count}")

        # First call
        input1 = {"value": "hello world", "number": 42}
        result1 = await intercept.intercept_invoke(input1, mock_next_call, intercept_context)
        assert call_count == 1
        assert result1.result == "Result 1"

        # Second call with similar input - should use cache
        input2 = {"value": "hello world!", "number": 42}
        result2 = await intercept.intercept_invoke(input2, mock_next_call, intercept_context)
        assert call_count == 1  # No additional call due to similarity
        assert result2.result == "Result 1"

        # Third call with very different input - should call function
        input3 = {"value": "goodbye universe", "number": 99}
        result3 = await intercept.intercept_invoke(input3, mock_next_call, intercept_context)
        assert call_count == 2
        assert result3.result == "Result 2"

    @pytest.mark.asyncio
    async def test_eval_mode_caching(self, intercept_context):
        """Test caching only works in eval mode when configured."""
        intercept = CacheIntercept(enabled_mode="eval", similarity_threshold=1.0)

        call_count = 0

        async def mock_next_call(_val):
            nonlocal call_count
            call_count += 1
            return TestOutput(result=f"Result {call_count}")

        # Mock ContextState to control is_evaluating
        mock_ctx_cls = 'nat.intercepts.cache_intercept.ContextState'
        with patch(mock_ctx_cls) as mock_context_state:
            mock_state = MagicMock()
            mock_context_state.get.return_value = mock_state

            # First, test when NOT evaluating
            mock_state.is_evaluating.get.return_value = False

            input1 = {"value": "test", "number": 42}
            await intercept.intercept_invoke(input1, mock_next_call, intercept_context)
            assert call_count == 1

            # Same input again - should NOT use cache
            await intercept.intercept_invoke(input1, mock_next_call, intercept_context)
            assert call_count == 2  # Called again

            # Now test when evaluating
            mock_state.is_evaluating.get.return_value = True

            # Same input - should call function (no cache before)
            await intercept.intercept_invoke(input1, mock_next_call, intercept_context)
            assert call_count == 3

            # Same input again - should use cache now
            await intercept.intercept_invoke(input1, mock_next_call, intercept_context)
            assert call_count == 3  # No additional call

    @pytest.mark.asyncio
    async def test_serialization_failure(self, intercept_context):
        """Test behavior when input serialization fails."""
        intercept = CacheIntercept(enabled_mode="always", similarity_threshold=1.0)

        call_count = 0

        async def mock_next_call(_val):
            nonlocal call_count
            call_count += 1
            return TestOutput(result="Result")

        # Create an object that can't be serialized
        class UnserializableObject:

            def __init__(self):
                self.circular_ref = self

        # Mock json.dumps to raise an exception
        with patch('json.dumps', side_effect=Exception("Cannot serialize")):
            input_obj = UnserializableObject()
            await intercept.intercept_invoke(input_obj, mock_next_call, intercept_context)
            assert call_count == 1

            # Try again - should call function again (no caching)
            await intercept.intercept_invoke(input_obj, mock_next_call, intercept_context)
            assert call_count == 2


class TestCacheInterceptStreaming:
    """Test streaming behavior."""

    @pytest.mark.asyncio
    async def test_streaming_bypass(self, intercept_context):
        """Test that streaming always bypasses cache."""
        intercept = CacheIntercept(enabled_mode="always", similarity_threshold=1.0)

        call_count = 0

        async def mock_stream_call(_val):
            nonlocal call_count
            call_count += 1
            for i in range(3):
                yield f"Chunk {i}"

        # First streaming call
        input1 = {"value": "test", "number": 42}
        chunks1 = []
        async for chunk in intercept.intercept_stream(input1, mock_stream_call, intercept_context):
            chunks1.append(chunk)
        assert call_count == 1
        assert chunks1 == ["Chunk 0", "Chunk 1", "Chunk 2"]

        # Second streaming call with same input - should call again
        chunks2 = []
        async for chunk in intercept.intercept_stream(input1, mock_stream_call, intercept_context):
            chunks2.append(chunk)
        assert call_count == 2  # Function called again
        assert chunks2 == ["Chunk 0", "Chunk 1", "Chunk 2"]


class TestCacheInterceptEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_context_retrieval_failure(self, intercept_context):
        """Test behavior when context retrieval fails in eval mode."""
        intercept = CacheIntercept(enabled_mode="eval", similarity_threshold=1.0)

        call_count = 0

        async def mock_next_call(_val):
            nonlocal call_count
            call_count += 1
            return TestOutput(result="Result")

        # Mock ContextState.get to raise an exception
        mock_ctx_cls = 'nat.intercepts.cache_intercept.ContextState.get'
        with patch(mock_ctx_cls, side_effect=Exception("Context error")):
            input1 = {"value": "test", "number": 42}
            await intercept.intercept_invoke(input1, mock_next_call, intercept_context)
            assert call_count == 1  # Should fall back to calling function

    def test_similarity_computation_for_different_thresholds(self):
        """Test similarity computation for different thresholds."""
        # This is more of a unit test for the similarity logic
        intercept = CacheIntercept(enabled_mode="always", similarity_threshold=0.5)

        # Directly test internal methods
        # Add a cached entry
        test_key = "hello world"
        intercept._cache[test_key] = "cached_result"  # noqa

        # Test various similarity levels
        # Exact match
        assert intercept._find_similar_key(test_key) == test_key  # noqa
        # Very similar
        assert intercept._find_similar_key("hello worl") == test_key  # noqa
        # Too different - use a completely different string
        assert intercept._find_similar_key("xyz123abc") is None  # noqa

    @pytest.mark.asyncio
    async def test_multiple_similar_entries(self, intercept_context):
        """Test behavior with multiple similar cached entries."""
        intercept = CacheIntercept(enabled_mode="always", similarity_threshold=0.7)

        # Pre-populate cache with similar entries
        key1 = intercept._serialize_input(  # noqa
            {
                "value": "test input 1", "number": 42
            })
        key2 = intercept._serialize_input(  # noqa
            {
                "value": "test input 2", "number": 42
            })
        intercept._cache[key1] = TestOutput(result="Result 1")  # noqa
        intercept._cache[key2] = TestOutput(result="Result 2")  # noqa

        async def mock_next_call(_val):
            return TestOutput(result="New Result")

        # Query with something similar to all
        input_str = {"value": "test input X", "number": 42}
        await intercept.intercept_invoke(input_str, mock_next_call, intercept_context)
        # The exact behavior depends on which cached key is most similar
