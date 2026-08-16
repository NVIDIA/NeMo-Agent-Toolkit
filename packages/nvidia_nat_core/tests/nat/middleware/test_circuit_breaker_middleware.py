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
"""Tests for CircuitBreakerMiddleware."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from nat.middleware.circuit_breaker.circuit_breaker_middleware import CircuitBreakerMiddleware
from nat.middleware.circuit_breaker.circuit_breaker_middleware import CircuitBreakerOpenError
from nat.middleware.circuit_breaker.circuit_breaker_middleware import CircuitBreakerState
from nat.middleware.circuit_breaker.circuit_breaker_middleware_config import CircuitBreakerMiddlewareConfig
from nat.middleware.middleware import FunctionMiddlewareContext


@pytest.fixture(name="mock_builder")
def fixture_mock_builder():
    builder: Mock = Mock()
    builder._functions = {}
    builder.get_llm = AsyncMock()
    builder.get_embedder = AsyncMock()
    builder.get_retriever = AsyncMock()
    builder.get_memory_client = AsyncMock()
    builder.get_object_store_client = AsyncMock()
    builder.get_auth_provider = AsyncMock()
    builder.get_function = AsyncMock()
    builder.get_function_config = Mock()
    return builder


@pytest.fixture(name="function_context")
def fixture_function_context():
    return FunctionMiddlewareContext(
        name="test_tool",
        config=Mock(),
        description="A test tool",
        input_schema=None,
        single_output_schema=type(None),
        stream_output_schema=type(None),
    )


def _make_context(name: str) -> FunctionMiddlewareContext:
    return FunctionMiddlewareContext(
        name=name,
        config=Mock(),
        description=f"Test tool {name}",
        input_schema=None,
        single_output_schema=type(None),
        stream_output_schema=type(None),
    )


def _make_middleware(
    mock_builder: Mock,
    *,
    failure_threshold: int = 3,
    cooldown_period: float = 0.2,
    half_open_success_threshold: int = 1,
    probe_timeout: float | None = None,
    circuit_breaker_message: str | None = None,
) -> CircuitBreakerMiddleware:
    kwargs: dict[str, Any] = {
        "failure_threshold": failure_threshold,
        "cooldown_period": cooldown_period,
        "half_open_success_threshold": half_open_success_threshold,
    }
    if probe_timeout is not None:
        kwargs["probe_timeout"] = probe_timeout
    if circuit_breaker_message is not None:
        kwargs["circuit_breaker_message"] = circuit_breaker_message
    config: CircuitBreakerMiddlewareConfig = CircuitBreakerMiddlewareConfig(**kwargs)
    return CircuitBreakerMiddleware(config=config, builder=mock_builder)


class TestCircuitBreakerMiddlewareConfig:
    """Test validation and defaults for CircuitBreakerMiddlewareConfig."""

    def test_valid_config(self):
        config = CircuitBreakerMiddlewareConfig(
            failure_threshold=5,
            cooldown_period=30.0,
            half_open_success_threshold=2,
            probe_timeout=5.0,
            circuit_breaker_message="Custom msg",
        )
        assert config.failure_threshold == 5
        assert config.cooldown_period == 30.0
        assert config.half_open_success_threshold == 2
        assert config.probe_timeout == 5.0
        assert config.circuit_breaker_message == "Custom msg"

    def test_invalid_failure_threshold(self):
        with pytest.raises(ValidationError):
            CircuitBreakerMiddlewareConfig(failure_threshold=0)

    def test_invalid_cooldown_period(self):
        with pytest.raises(ValidationError):
            CircuitBreakerMiddlewareConfig(cooldown_period=-1.0)

    def test_invalid_half_open_success_threshold(self):
        with pytest.raises(ValidationError):
            CircuitBreakerMiddlewareConfig(half_open_success_threshold=0)

    def test_invalid_probe_timeout(self):
        with pytest.raises(ValidationError):
            CircuitBreakerMiddlewareConfig(probe_timeout=-0.5)


class TestCircuitBreakerMiddlewareInvoke:
    """Test invocation behavior and state transitions for CircuitBreakerMiddleware."""

    async def test_closed_state_success(self, mock_builder, function_context):
        middleware = _make_middleware(mock_builder, failure_threshold=3)
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED

        async def success_fn(*args, **kwargs):
            return "ok"

        call_next = AsyncMock(side_effect=success_fn)
        res = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert res == "ok"
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED
        assert middleware.get_failure_count("test_tool") == 0
        call_next.assert_called_once()

    async def test_closed_to_open_transition(self, mock_builder, function_context):
        middleware = _make_middleware(mock_builder, failure_threshold=3)

        async def failing_fn(*args, **kwargs):
            raise RuntimeError("Backend service down")

        call_next = AsyncMock(side_effect=failing_fn)

        with pytest.raises(RuntimeError, match="Backend service down"):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED
        assert middleware.get_failure_count("test_tool") == 1

        with pytest.raises(RuntimeError, match="Backend service down"):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED
        assert middleware.get_failure_count("test_tool") == 2

        with pytest.raises(RuntimeError, match="Backend service down"):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN
        assert middleware.get_failure_count("test_tool") == 3
        assert call_next.call_count == 3

    async def test_short_circuit_raises_circuit_breaker_open_error(self, mock_builder, function_context):
        middleware = _make_middleware(
            mock_builder,
            failure_threshold=1,
            cooldown_period=10.0,
            circuit_breaker_message="Please retry later.",
        )

        call_next = AsyncMock(side_effect=RuntimeError("Failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        call_next.reset_mock()
        with pytest.raises(CircuitBreakerOpenError,
                           match="Circuit breaker is OPEN for 'test_tool'.*Please retry later."):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        call_next.assert_not_called()

    async def test_default_open_error_message(self, mock_builder, function_context):
        middleware = _make_middleware(
            mock_builder,
            failure_threshold=1,
            cooldown_period=10.0,
            circuit_breaker_message=None,
        )

        call_next = AsyncMock(side_effect=RuntimeError("Failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert str(exc_info.value) == "Circuit breaker is OPEN for 'test_tool'. Tool is temporarily unavailable."

    async def test_open_to_half_open_recovery_single_probe(self, mock_builder, function_context):
        middleware = _make_middleware(
            mock_builder,
            failure_threshold=1,
            cooldown_period=0.1,
            half_open_success_threshold=1,
        )

        call_next = AsyncMock(side_effect=RuntimeError("Failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)

        call_next.side_effect = None
        call_next.return_value = "recovered_result"

        res = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert res == "recovered_result"
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED
        assert middleware.get_failure_count("test_tool") == 0

    async def test_half_open_multiple_success_threshold(self, mock_builder, function_context):
        middleware = _make_middleware(
            mock_builder,
            failure_threshold=1,
            cooldown_period=0.1,
            half_open_success_threshold=2,
        )

        call_next = AsyncMock(side_effect=RuntimeError("Failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)

        call_next.side_effect = None
        call_next.return_value = "probe_1_ok"

        # Probe #1 succeeds, but breaker requires 2 consecutive successes to close
        res1 = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert res1 == "probe_1_ok"
        assert middleware.get_state("test_tool") == CircuitBreakerState.HALF_OPEN
        assert middleware.get_success_count("test_tool") == 1

        # Probe #2 succeeds and meets threshold, closing breaker
        call_next.return_value = "probe_2_ok"
        res2 = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert res2 == "probe_2_ok"
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED
        assert middleware.get_success_count("test_tool") == 0
        assert middleware.get_failure_count("test_tool") == 0

    async def test_half_open_failed_probe_retrips(self, mock_builder, function_context):
        middleware = _make_middleware(mock_builder, failure_threshold=1, cooldown_period=0.1)

        call_next = AsyncMock(side_effect=RuntimeError("Failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)

        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)

        # On failed probe, failure count is reset to threshold rather than incremented
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN
        assert middleware.get_failure_count("test_tool") == 1

    async def test_half_open_probe_timeout(self, mock_builder, function_context):
        middleware = _make_middleware(
            mock_builder,
            failure_threshold=1,
            cooldown_period=0.1,
            probe_timeout=0.05,
        )

        call_next = AsyncMock(side_effect=RuntimeError("Initial failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)

        async def slow_probe(*args, **kwargs):
            await asyncio.sleep(0.3)
            return "too_late"

        call_next.side_effect = slow_probe

        with pytest.raises(TimeoutError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)

        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN
        assert middleware._get_tool_state("test_tool").half_open_probing is False

    async def test_probe_timeout_none_waits_for_completion(self, mock_builder, function_context):
        middleware = _make_middleware(
            mock_builder,
            failure_threshold=1,
            cooldown_period=0.1,
            probe_timeout=None,
        )

        call_next = AsyncMock(side_effect=RuntimeError("Initial failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)

        async def slow_successful_probe(*args, **kwargs):
            await asyncio.sleep(0.1)
            return "slow_success"

        call_next.side_effect = slow_successful_probe
        res = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert res == "slow_success"
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED

    async def test_half_open_cancellation_does_not_trip_or_increment_failures(self, mock_builder, function_context):
        middleware = _make_middleware(mock_builder, failure_threshold=2, cooldown_period=0.1)

        call_next = AsyncMock(side_effect=RuntimeError("Initial failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)

        probe_started = asyncio.Event()

        async def hung_probe(*args, **kwargs):
            probe_started.set()
            await asyncio.sleep(10.0)

        call_next.side_effect = hung_probe

        task = asyncio.create_task(
            middleware.function_middleware_invoke("input", call_next=call_next, context=function_context))
        await probe_started.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert middleware.get_state("test_tool") == CircuitBreakerState.HALF_OPEN
        assert middleware._get_tool_state("test_tool").half_open_probing is False

        call_next.side_effect = None
        call_next.return_value = "recovered_after_cancellation"
        res = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert res == "recovered_after_cancellation"
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED

    async def test_half_open_concurrency_safety(self, mock_builder, function_context):
        middleware = _make_middleware(mock_builder, failure_threshold=1, cooldown_period=0.1)

        call_next = AsyncMock(side_effect=RuntimeError("Initial failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)

        probe_started = asyncio.Event()
        probe_release = asyncio.Event()

        async def slow_probe(*args, **kwargs):
            probe_started.set()
            await probe_release.wait()
            return "probe_success"

        call_next.side_effect = slow_probe

        task = asyncio.create_task(
            middleware.function_middleware_invoke("input", call_next=call_next, context=function_context))
        await probe_started.wait()

        with pytest.raises(CircuitBreakerOpenError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)

        probe_release.set()
        res_probe = await task
        assert res_probe == "probe_success"
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED

    async def test_tool_target_isolation(self, mock_builder):
        middleware = _make_middleware(mock_builder, failure_threshold=2)
        ctx_a = _make_context("tool_a")
        ctx_b = _make_context("tool_b")

        call_next_a = AsyncMock(side_effect=RuntimeError("tool_a failed"))
        call_next_b = AsyncMock(return_value="tool_b ok")

        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next_a, context=ctx_a)
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next_a, context=ctx_a)

        assert middleware.get_state("tool_a") == CircuitBreakerState.OPEN
        assert middleware.get_state("tool_b") == CircuitBreakerState.CLOSED

        res_b = await middleware.function_middleware_invoke("input", call_next=call_next_b, context=ctx_b)
        assert res_b == "tool_b ok"
        assert middleware.get_state("tool_b") == CircuitBreakerState.CLOSED


class TestCircuitBreakerMiddlewareStream:
    """Test streaming behavior for CircuitBreakerMiddleware."""

    async def test_streaming_success(self, mock_builder, function_context):
        middleware = _make_middleware(mock_builder, failure_threshold=2)

        async def stream_fn(*args, **kwargs):
            yield "chunk1"
            yield "chunk2"

        call_next = Mock(side_effect=stream_fn)

        chunks = []
        async for chunk in middleware.function_middleware_stream("input", call_next=call_next,
                                                                 context=function_context):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED

    async def test_streaming_failure_trips_breaker(self, mock_builder, function_context):
        middleware = _make_middleware(mock_builder, failure_threshold=1)

        async def failing_stream(*args, **kwargs):
            yield "chunk1"
            raise RuntimeError("Stream broken")

        call_next = Mock(side_effect=failing_stream)

        with pytest.raises(RuntimeError, match="Stream broken"):
            async for _ in middleware.function_middleware_stream("input", call_next=call_next,
                                                                 context=function_context):
                pass

        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        with pytest.raises(CircuitBreakerOpenError):
            async for _ in middleware.function_middleware_stream("input", call_next=call_next,
                                                                 context=function_context):
                pass

    async def test_streaming_probe_timeout(self, mock_builder, function_context):
        middleware = _make_middleware(
            mock_builder,
            failure_threshold=1,
            cooldown_period=0.1,
            probe_timeout=0.05,
        )

        call_next = Mock(side_effect=RuntimeError("Initial failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)

        async def slow_stream(*args, **kwargs):
            yield "chunk1"
            await asyncio.sleep(0.3)
            yield "chunk2"

        call_next.side_effect = slow_stream

        with pytest.raises(TimeoutError):
            async for _ in middleware.function_middleware_stream("input", call_next=call_next,
                                                                 context=function_context):
                pass

        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN
        assert middleware._get_tool_state("test_tool").half_open_probing is False

    async def test_streaming_probe_timeout_does_not_trigger_on_consumer_delay(self, mock_builder, function_context):
        """Consumer delays between yielded chunks must not trigger downstream probe timeouts."""
        middleware = _make_middleware(
            mock_builder,
            failure_threshold=1,
            cooldown_period=0.1,
            probe_timeout=0.1,
        )

        call_next = Mock(side_effect=RuntimeError("Initial failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)

        async def fast_stream(*args, **kwargs):
            yield "chunk1"
            yield "chunk2"

        call_next.side_effect = fast_stream

        chunks = []
        async for chunk in middleware.function_middleware_stream("input", call_next=call_next,
                                                                 context=function_context):
            chunks.append(chunk)
            # Consumer delay exceeds probe_timeout, but does not trip timeout because __anext__ was fast
            await asyncio.sleep(0.15)

        assert chunks == ["chunk1", "chunk2"]
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED

    async def test_streaming_early_break_resets_probing_and_admits_next_call(self, mock_builder, function_context):
        """Early break from streaming probe cleans up probing state and admits subsequent calls."""
        middleware = _make_middleware(mock_builder, failure_threshold=1, cooldown_period=0.1)

        call_next = Mock(side_effect=RuntimeError("Initial failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.get_state("test_tool") == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)

        async def infinite_stream(*args, **kwargs):
            count = 0
            while True:
                yield f"chunk_{count}"
                count += 1
                await asyncio.sleep(0.01)

        call_next.side_effect = infinite_stream

        async for chunk in middleware.function_middleware_stream("input", call_next=call_next,
                                                                 context=function_context):
            assert chunk == "chunk_0"
            break

        assert middleware._get_tool_state("test_tool").half_open_probing is False

        # Verify next probe call is admitted without being short-circuited
        call_next = AsyncMock(return_value="probe_success")
        res = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert res == "probe_success"
        assert middleware.get_state("test_tool") == CircuitBreakerState.CLOSED
