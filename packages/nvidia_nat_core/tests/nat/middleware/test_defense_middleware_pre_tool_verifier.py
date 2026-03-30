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
"""Tests for PreToolVerifierMiddleware, including chunked analysis of long inputs."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.builder.function import FunctionGroup
from nat.middleware.defense.defense_middleware_pre_tool_verifier import PreToolVerifierMiddleware
from nat.middleware.defense.defense_middleware_pre_tool_verifier import PreToolVerifierMiddlewareConfig
from nat.middleware.middleware import FunctionMiddlewareContext

_MAX_CONTENT_LENGTH = 32000


class _TestInput(BaseModel):
    """Test input model."""
    query: str


class _TestOutput(BaseModel):
    """Test output model."""
    result: str


@pytest.fixture(name="mock_builder")
def fixture_mock_builder():
    """Create a mock builder."""
    return MagicMock()


@pytest.fixture(name="middleware_context")
def fixture_middleware_context():
    """Create a test FunctionMiddlewareContext."""
    return FunctionMiddlewareContext(name=f"my_tool{FunctionGroup.SEPARATOR}search",
                                    config=MagicMock(),
                                    description="Search function",
                                    input_schema=_TestInput,
                                    single_output_schema=_TestOutput,
                                    stream_output_schema=type(None))


def _make_llm_response(violation: bool,
                       confidence: float = 0.9,
                       reason: str = "test reason",
                       violation_types: list[str] | None = None,
                       sanitized: str | None = None) -> MagicMock:
    """Build a mock LLM response with the given verification result."""
    vt = violation_types or (["prompt_injection"] if violation else [])
    sanitized_str = f'"{sanitized}"' if sanitized is not None else "null"
    content = (f'{{"violation_detected": {str(violation).lower()}, "confidence": {confidence}, '
               f'"reason": "{reason}", "violation_types": {vt}, "sanitized_input": {sanitized_str}}}')
    mock_response = MagicMock()
    mock_response.content = content
    return mock_response


class TestAnalyzeContentChunking:
    """Tests for the content chunking behavior in _analyze_content."""

    async def test_short_content_single_llm_call(self, mock_builder, middleware_context):
        """Content within limit is analyzed with a single LLM call."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(False, confidence=0.1))
        middleware._llm = mock_llm

        short_content = "a" * (_MAX_CONTENT_LENGTH - 1)
        result = await middleware._analyze_content(short_content, function_name=middleware_context.name)

        assert mock_llm.ainvoke.call_count == 1
        assert not result.violation_detected
        assert not result.should_refuse

    async def test_long_content_split_into_multiple_chunks(self, mock_builder, middleware_context):
        """Content exceeding limit is split and each chunk analyzed separately."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(False, confidence=0.1))
        middleware._llm = mock_llm

        # 2.5x the limit => 3 chunks
        long_content = "a" * int(_MAX_CONTENT_LENGTH * 2.5)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert mock_llm.ainvoke.call_count == 3
        assert not result.violation_detected

    async def test_malicious_payload_in_middle_chunk_detected(self, mock_builder, middleware_context):
        """A violation hidden in the middle of long content is detected (was vulnerable before fix)."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # chunk 0: clean, chunk 1: malicious (middle), chunk 2: clean
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(True, confidence=0.95, reason="prompt injection detected"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 3)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert mock_llm.ainvoke.call_count == 3
        assert result.violation_detected
        assert result.should_refuse
        assert result.confidence == 0.95
        assert "prompt injection detected" in result.reason

    async def test_violation_in_last_chunk_detected(self, mock_builder, middleware_context):
        """A violation in the last chunk is detected."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(True, confidence=0.85, reason="jailbreak in last chunk"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert result.violation_detected
        assert result.should_refuse
        assert "jailbreak in last chunk" in result.reason

    async def test_no_violation_in_any_chunk_returns_clean(self, mock_builder, middleware_context):
        """When all chunks are clean, result is clean."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.2, reason="clean"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert not result.violation_detected
        assert not result.should_refuse

    async def test_chunked_max_confidence_taken(self, mock_builder, middleware_context):
        """Aggregated confidence is the maximum across all chunks."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(True, confidence=0.75, reason="low confidence violation"),
            _make_llm_response(True, confidence=0.95, reason="high confidence violation"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert result.confidence == 0.95

    async def test_chunked_violation_types_deduplicated(self, mock_builder, middleware_context):
        """Violation types from all chunks are merged without duplicates."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(True, confidence=0.8, violation_types=["prompt_injection", "jailbreak"]),
            _make_llm_response(True, confidence=0.8, violation_types=["jailbreak", "social_engineering"]),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert set(result.violation_types) == {"prompt_injection", "jailbreak", "social_engineering"}
        assert len(result.violation_types) == 3

    async def test_chunked_sanitized_input_reconstructed(self, mock_builder, middleware_context):
        """Sanitized input is reconstructed by concatenating sanitized/original chunks."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="redirection", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        clean_chunk = "a" * _MAX_CONTENT_LENGTH
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(True, confidence=0.9, reason="violation", sanitized="sanitized_part"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert result.violation_detected
        # First chunk is clean => original; second chunk is sanitized
        assert result.sanitized_input == clean_chunk + "sanitized_part"

    async def test_chunked_sanitized_input_none_when_chunk_missing_sanitization(self, mock_builder, middleware_context):
        """sanitized_input is None when a violating chunk provides no sanitized version."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="redirection", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(True, confidence=0.9, reason="violation", sanitized=None),
            _make_llm_response(False, confidence=0.1, reason="clean"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert result.violation_detected
        assert result.sanitized_input is None

    async def test_chunked_reasons_combined(self, mock_builder, middleware_context):
        """Reasons from all violating chunks are combined with semicolons."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(True, confidence=0.8, reason="reason A"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(True, confidence=0.9, reason="reason B"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 3)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert "reason A" in result.reason
        assert "reason B" in result.reason

    async def test_chunked_error_in_one_chunk_propagates(self, mock_builder, middleware_context):
        """An error in any chunk sets error=True on the aggregated result."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance", fail_closed=False)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # First chunk raises an exception, second succeeds
        mock_llm.ainvoke = AsyncMock(side_effect=[
            Exception("LLM failure"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        with patch('nat.middleware.defense.defense_middleware_pre_tool_verifier.logger'):
            result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert result.error


class TestPreToolVerifierInvoke:
    """Tests for function_middleware_invoke behavior."""

    async def test_clean_input_passes_through(self, mock_builder, middleware_context):
        """Clean input is passed to the tool unchanged."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(False, confidence=0.1))
        middleware._llm = mock_llm

        call_next_input = None

        async def mock_next(value):
            nonlocal call_next_input
            call_next_input = value
            return "result"

        result = await middleware.function_middleware_invoke("safe input",
                                                             call_next=mock_next,
                                                             context=middleware_context)

        assert result == "result"
        assert call_next_input == "safe input"

    async def test_refusal_action_blocks_violating_input(self, mock_builder, middleware_context):
        """Violating input raises ValueError when action is 'refusal'."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(True, confidence=0.9))
        middleware._llm = mock_llm

        async def mock_next(value):
            return "should not reach"

        with pytest.raises(ValueError, match="Input blocked by security policy"):
            await middleware.function_middleware_invoke("injected input",
                                                        call_next=mock_next,
                                                        context=middleware_context)

    async def test_redirection_action_sanitizes_input(self, mock_builder, middleware_context):
        """Violating input is replaced with sanitized version when action is 'redirection'."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="redirection", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=_make_llm_response(True, confidence=0.9, sanitized="sanitized query"))
        middleware._llm = mock_llm

        call_next_input = None

        async def mock_next(value):
            nonlocal call_next_input
            call_next_input = value
            return "result"

        await middleware.function_middleware_invoke("injected input", call_next=mock_next, context=middleware_context)

        assert call_next_input == "sanitized query"

    async def test_partial_compliance_logs_but_allows_input(self, mock_builder, middleware_context):
        """Violating input is logged but allowed through when action is 'partial_compliance'."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(True, confidence=0.9))
        middleware._llm = mock_llm

        call_next_input = None

        async def mock_next(value):
            nonlocal call_next_input
            call_next_input = value
            return "result"

        with patch('nat.middleware.defense.defense_middleware_pre_tool_verifier.logger') as mock_logger:
            result = await middleware.function_middleware_invoke("injected input",
                                                                 call_next=mock_next,
                                                                 context=middleware_context)

            mock_logger.warning.assert_called()

        assert result == "result"
        assert call_next_input == "injected input"

    async def test_skips_non_targeted_function(self, mock_builder, middleware_context):
        """Defense is skipped for functions not matching target_function_or_group."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm",
                                                 target_function_or_group="other_tool",
                                                 action="refusal")
        middleware = PreToolVerifierMiddleware(config, mock_builder)
        mock_llm = AsyncMock()
        middleware._llm = mock_llm

        async def mock_next(value):
            return "result"

        result = await middleware.function_middleware_invoke("any input",
                                                             call_next=mock_next,
                                                             context=middleware_context)

        assert result == "result"
        assert not mock_llm.ainvoke.called

    async def test_below_threshold_does_not_trigger_refusal(self, mock_builder, middleware_context):
        """A violation below the confidence threshold does not block the input."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.9)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # violation_detected=True but confidence (0.5) is below threshold (0.9)
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(True, confidence=0.5))
        middleware._llm = mock_llm

        async def mock_next(value):
            return "result"

        result = await middleware.function_middleware_invoke("input", call_next=mock_next, context=middleware_context)

        assert result == "result"


class TestPreToolVerifierStreaming:
    """Tests for function_middleware_stream behavior."""

    async def test_streaming_clean_input_passes_through(self, mock_builder, middleware_context):
        """Clean input allows streaming chunks to pass through unchanged."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(False, confidence=0.1))
        middleware._llm = mock_llm

        async def mock_stream(value):
            yield "chunk1"
            yield "chunk2"

        chunks = []
        async for chunk in middleware.function_middleware_stream("safe input",
                                                                  call_next=mock_stream,
                                                                  context=middleware_context):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]

    async def test_streaming_refusal_blocks_violating_input(self, mock_builder, middleware_context):
        """Violating input raises ValueError before streaming begins."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(True, confidence=0.9))
        middleware._llm = mock_llm

        async def mock_stream(value):
            yield "should not reach"

        with pytest.raises(ValueError, match="Input blocked by security policy"):
            async for _ in middleware.function_middleware_stream("injected input",
                                                                  call_next=mock_stream,
                                                                  context=middleware_context):
                pass

    async def test_streaming_skips_non_targeted_function(self, mock_builder, middleware_context):
        """Streaming skips defense for functions not matching target_function_or_group."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm",
                                                 target_function_or_group="other_tool",
                                                 action="refusal")
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        async def mock_stream(value):
            yield "chunk1"
            yield "chunk2"

        chunks = []
        async for chunk in middleware.function_middleware_stream("input",
                                                                  call_next=mock_stream,
                                                                  context=middleware_context):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]
