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

import json
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
_STRIDE = _MAX_CONTENT_LENGTH // 2  # 50% overlap — injections ≤ _STRIDE chars are guaranteed full coverage
_MAX_CHUNKS = 16


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
    vt = violation_types if violation_types is not None else (["prompt_injection"] if violation else [])
    content = json.dumps({
        "violation_detected": violation,
        "confidence": confidence,
        "reason": reason,
        "violation_types": vt,
        "sanitized_input": sanitized,
    })
    mock_response = MagicMock()
    mock_response.content = content
    return mock_response


class TestAnalyzeContentChunking:
    """Tests for the sliding-window analysis behavior in _analyze_content.

    With _MAX_CONTENT_LENGTH=32000 and _STRIDE=16000 (50% overlap):
      - 64000 chars  → range(0, 64000, 16000) → 4 windows
      - 80000 chars  → range(0, 80000, 16000) → 5 windows
      - 96000 chars  → range(0, 96000, 16000) → 6 windows

    The loop exits early as soon as a window returns should_refuse=True, so call counts
    may be lower than the total window count when a violation is found mid-scan.
    Inputs requiring more than _MAX_CHUNKS windows bypass LLM calls entirely.
    """

    async def test_short_content_single_llm_call(self, mock_builder, middleware_context):
        """Content within limit is analyzed with a single LLM call (no windowing)."""
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

    async def test_long_content_uses_sliding_windows(self, mock_builder, middleware_context):
        """Content exceeding limit is analyzed using overlapping sliding windows."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance")
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(False, confidence=0.1))
        middleware._llm = mock_llm

        # 2.5x the limit → 5 overlapping windows
        long_content = "a" * int(_MAX_CONTENT_LENGTH * 2.5)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert mock_llm.ainvoke.call_count == 5
        assert not result.violation_detected

    async def test_malicious_payload_in_middle_window_detected(self, mock_builder, middleware_context):
        """A violation in any window of long content is detected; early exit stops remaining windows."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # 96000 chars → 6 windows; window 2 carries the violation.
        # Early exit fires after window 2 (should_refuse=True), so only 3 calls are made.
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(True, confidence=0.95, reason="prompt injection detected"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
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

    async def test_violation_in_last_window_detected(self, mock_builder, middleware_context):
        """A violation in the last sliding window is detected."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # 64000 chars → 4 windows; last window carries the violation
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(True, confidence=0.85, reason="jailbreak in last window"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert result.violation_detected
        assert result.should_refuse
        assert "jailbreak in last window" in result.reason

    async def test_no_violation_in_any_window_returns_clean(self, mock_builder, middleware_context):
        """When all sliding windows are clean, the result is clean."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # 64000 chars → 4 windows, all clean
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.2, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert not result.violation_detected
        assert not result.should_refuse

    async def test_windowed_max_confidence_taken(self, mock_builder, middleware_context):
        """Aggregated confidence is the maximum across all windows."""
        # threshold=0.99 prevents early exit so all windows are scanned and max confidence is correct
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance", threshold=0.99)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # 64000 chars → 4 windows; windows 0 and 1 have violations at different confidences
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(True, confidence=0.75, reason="low confidence violation"),
            _make_llm_response(True, confidence=0.95, reason="high confidence violation"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert result.confidence == 0.95

    async def test_windowed_violation_types_deduplicated(self, mock_builder, middleware_context):
        """Violation types from all windows are merged without duplicates."""
        # threshold=0.99 prevents early exit so all windows are scanned and types from both are merged
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance", threshold=0.99)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # 64000 chars → 4 windows; windows 0 and 1 report overlapping type sets
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(True, confidence=0.8, violation_types=["prompt_injection", "jailbreak"]),
            _make_llm_response(True, confidence=0.8, violation_types=["jailbreak", "social_engineering"]),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert set(result.violation_types) == {"prompt_injection", "jailbreak", "social_engineering"}
        assert len(result.violation_types) == 3

    async def test_windowed_sanitized_input_always_none(self, mock_builder, middleware_context):
        """sanitized_input is always None for multi-window content.

        Overlapping windows make it impossible to reconstruct a sanitized version of the
        original input, so we always return None regardless of what individual windows report.
        """
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="redirection", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # 64000 chars → 4 windows; window 1 reports a violation with a sanitized version
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(True, confidence=0.9, reason="violation", sanitized="sanitized_part"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert result.violation_detected
        assert result.sanitized_input is None

    async def test_windowed_reasons_combined(self, mock_builder, middleware_context):
        """Reasons from all violating windows are combined with semicolons."""
        # threshold=0.99 prevents early exit so all windows are scanned and both reasons are collected
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance", threshold=0.99)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # 96000 chars → 6 windows; windows 0 and 4 carry violations
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(True, confidence=0.8, reason="reason A"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(True, confidence=0.9, reason="reason B"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 3)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert "reason A" in result.reason
        assert "reason B" in result.reason

    async def test_malicious_payload_split_at_old_boundary_detected(self, mock_builder, middleware_context):
        """A directive split at the old disjoint-chunk boundary is caught by the overlapping window.

        With stride=_STRIDE, window 1 starts at _STRIDE and ends at _STRIDE+_MAX_CONTENT_LENGTH,
        so it spans the position _MAX_CONTENT_LENGTH that was previously a hard boundary.
        Any injection straddling that boundary is fully visible in window 1.
        Early exit fires after window 1 (should_refuse=True), so only 2 calls are made.
        """
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # 64000 chars → 4 windows:
        #   window 0: [0 : 32000]            - clean (only left side of old boundary)
        #   window 1: [16000 : 48000]         - VIOLATION (spans old boundary at 32000) → early exit
        #   window 2: [32000 : 64000]         - never reached
        #   window 3: [48000 : 64000] (short) - never reached
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(True, confidence=0.9, reason="injection spanning old boundary"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert mock_llm.ainvoke.call_count == 2

        # Verify window 1 was passed content starting at _STRIDE (spans the old boundary)
        window1_messages = mock_llm.ainvoke.call_args_list[1][0][0]
        window1_user_content = window1_messages[1]["content"]
        expected_window1 = "a" * _MAX_CONTENT_LENGTH  # content[_STRIDE : _STRIDE + _MAX_CONTENT_LENGTH]
        assert expected_window1 in window1_user_content

        assert result.violation_detected
        assert result.should_refuse
        assert result.confidence == 0.9
        assert result.sanitized_input is None

    async def test_early_exit_stops_after_first_refusing_window(self, mock_builder, middleware_context):
        """Scanning stops immediately after the first window that returns should_refuse=True."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # 64000 chars → 4 windows; window 0 carries the violation → only 1 call should be made
        mock_llm.ainvoke = AsyncMock(side_effect=[
            _make_llm_response(True, confidence=0.95, reason="early violation"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
        ])
        middleware._llm = mock_llm

        long_content = "a" * (_MAX_CONTENT_LENGTH * 2)
        result = await middleware._analyze_content(long_content, function_name=middleware_context.name)

        assert mock_llm.ainvoke.call_count == 1
        assert result.violation_detected
        assert result.should_refuse

    async def test_over_cap_randomly_samples_max_chunks_windows(self, mock_builder, middleware_context):
        """Input requiring more than _MAX_CHUNKS windows is analyzed by sampling exactly _MAX_CHUNKS windows."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="refusal", threshold=0.7)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(False, confidence=0.1))
        middleware._llm = mock_llm

        # (_MAX_CHUNKS * _STRIDE) + 1 chars → _MAX_CHUNKS + 1 windows, exceeding the cap
        over_cap_content = "a" * (_MAX_CHUNKS * _STRIDE + 1)
        result = await middleware._analyze_content(over_cap_content, function_name=middleware_context.name)

        # All sampled windows are clean → exactly _MAX_CHUNKS calls, no early exit
        assert mock_llm.ainvoke.call_count == _MAX_CHUNKS
        assert not result.violation_detected
        assert not result.should_refuse

    async def test_windowed_error_in_one_window_propagates(self, mock_builder, middleware_context):
        """An error in any window sets error=True on the aggregated result."""
        config = PreToolVerifierMiddlewareConfig(llm_name="test_llm", action="partial_compliance", fail_closed=False)
        middleware = PreToolVerifierMiddleware(config, mock_builder)

        mock_llm = AsyncMock()
        # 64000 chars → 4 windows; window 0 fails, rest succeed
        mock_llm.ainvoke = AsyncMock(side_effect=[
            Exception("LLM failure"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
            _make_llm_response(False, confidence=0.1, reason="clean"),
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
