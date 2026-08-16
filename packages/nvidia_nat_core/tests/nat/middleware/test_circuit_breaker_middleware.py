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

from nat.middleware.circuit_breaker.circuit_breaker_middleware import CircuitBreakerMiddleware
from nat.middleware.circuit_breaker.circuit_breaker_middleware import CircuitBreakerState
from nat.middleware.circuit_breaker.circuit_breaker_middleware_config import CircuitBreakerMiddlewareConfig
from nat.middleware.middleware import FunctionMiddlewareContext

# ==================== Fixtures ====================


@pytest.fixture(name="mock_builder")
def fixture_mock_builder():
    """Create a mock builder for middleware instantiation."""
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
    """Create a test FunctionMiddlewareContext."""
    return FunctionMiddlewareContext(
        name="test_tool",
        config=Mock(),
        description="A test tool",
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
    circuit_breaker_message: str | None = None,
) -> CircuitBreakerMiddleware:
    """Create a CircuitBreakerMiddleware with test configurations."""
    kwargs: dict[str, Any] = {
        "failure_threshold": failure_threshold,
        "cooldown_period": cooldown_period,
        "half_open_success_threshold": half_open_success_threshold,
    }
    if circuit_breaker_message is not None:
        kwargs["circuit_breaker_message"] = circuit_breaker_message
    config: CircuitBreakerMiddlewareConfig = CircuitBreakerMiddlewareConfig(**kwargs)
    return CircuitBreakerMiddleware(config=config, builder=mock_builder)


# ==================== Single Invocation Tests ====================


class TestCircuitBreakerMiddlewareInvoke:
    """Tests for function_middleware_invoke circuit breaker state machine."""

    async def test_closed_state_success(self, mock_builder, function_context):
        """Normal successful calls pass through in CLOSED state."""
        middleware = _make_middleware(mock_builder, failure_threshold=3)
        assert middleware.state == CircuitBreakerState.CLOSED

        async def success_fn(*args, **kwargs):
            return "ok"

        call_next = AsyncMock(side_effect=success_fn)

        res = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert res == "ok"
        assert middleware.state == CircuitBreakerState.CLOSED
        assert middleware.failure_count == 0
        call_next.assert_called_once()

    async def test_closed_to_open_transition(self, mock_builder, function_context):
        """Failure threshold consecutive exceptions trip circuit breaker to OPEN."""
        middleware = _make_middleware(mock_builder, failure_threshold=3)

        async def failing_fn(*args, **kwargs):
            raise RuntimeError("Backend service down")

        call_next = AsyncMock(side_effect=failing_fn)

        # 1st failure
        with pytest.raises(RuntimeError, match="Backend service down"):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.state == CircuitBreakerState.CLOSED
        assert middleware.failure_count == 1

        # 2nd failure
        with pytest.raises(RuntimeError, match="Backend service down"):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.state == CircuitBreakerState.CLOSED
        assert middleware.failure_count == 2

        # 3rd failure -> Trips to OPEN
        with pytest.raises(RuntimeError, match="Backend service down"):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.state == CircuitBreakerState.OPEN
        assert middleware.failure_count == 3
        assert call_next.call_count == 3

    async def test_short_circuit_when_open(self, mock_builder, function_context):
        """Immediate short-circuiting in OPEN state without calling downstream target."""
        middleware = _make_middleware(
            mock_builder,
            failure_threshold=1,
            cooldown_period=10.0,
            circuit_breaker_message="Please try again later.",
        )

        call_next = AsyncMock(side_effect=RuntimeError("Failure"))

        # Trip to OPEN
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.state == CircuitBreakerState.OPEN

        call_next.reset_mock()

        # Subsequent call should short-circuit immediately
        res = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert "Circuit breaker is OPEN for 'test_tool'" in res
        assert "Please try again later." in res
        call_next.assert_not_called()

    async def test_open_to_half_open_recovery(self, mock_builder, function_context):
        """Cooldown period elapsing transitions to HALF_OPEN and successful probe recovers to CLOSED."""
        middleware = _make_middleware(mock_builder,
                                      failure_threshold=1,
                                      cooldown_period=0.1,
                                      half_open_success_threshold=1)

        call_next = AsyncMock(side_effect=RuntimeError("Failure"))

        # Trip to OPEN
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.state == CircuitBreakerState.OPEN

        # Wait for cooldown
        await asyncio.sleep(0.15)

        # Update call_next to succeed
        call_next.side_effect = None
        call_next.return_value = "recovered_result"

        # Probe call in HALF_OPEN
        res = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert res == "recovered_result"
        assert middleware.state == CircuitBreakerState.CLOSED
        assert middleware.failure_count == 0

    async def test_half_open_failed_probe_retrips(self, mock_builder, function_context):
        """Failed probe in HALF_OPEN trips back to OPEN and resets cooldown."""
        middleware = _make_middleware(mock_builder, failure_threshold=1, cooldown_period=0.1)

        call_next = AsyncMock(side_effect=RuntimeError("Failure"))

        # Trip to OPEN
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.state == CircuitBreakerState.OPEN

        # Wait for cooldown
        await asyncio.sleep(0.15)

        # Probe call fails
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)

        # Re-tripped to OPEN
        assert middleware.state == CircuitBreakerState.OPEN

    async def test_half_open_concurrency_safety(self, mock_builder, function_context):
        """Only one probe call is allowed during HALF_OPEN; concurrent calls short-circuit."""
        middleware = _make_middleware(mock_builder, failure_threshold=1, cooldown_period=0.1)

        call_next = AsyncMock(side_effect=RuntimeError("Initial failure"))
        with pytest.raises(RuntimeError):
            await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)
        assert middleware.state == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)

        probe_started = asyncio.Event()
        probe_release = asyncio.Event()

        async def slow_probe(*args, **kwargs):
            probe_started.set()
            await probe_release.wait()
            return "probe_success"

        call_next.side_effect = slow_probe

        # Start probe call in background
        task = asyncio.create_task(
            middleware.function_middleware_invoke("input", call_next=call_next, context=function_context))
        await probe_started.wait()

        # Concurrent call while probe is in flight should short-circuit
        res_concurrent = await middleware.function_middleware_invoke("input",
                                                                     call_next=call_next,
                                                                     context=function_context)
        assert "Circuit breaker is OPEN for 'test_tool'" in res_concurrent

        # Finish probe
        probe_release.set()
        res_probe = await task
        assert res_probe == "probe_success"
        assert middleware.state == CircuitBreakerState.CLOSED


# ==================== Streaming Invocation Tests ====================


class TestCircuitBreakerMiddlewareStream:
    """Tests for function_middleware_stream circuit breaker state machine."""

    async def test_streaming_success(self, mock_builder, function_context):
        """Successful stream passes through in CLOSED state."""
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
        assert middleware.state == CircuitBreakerState.CLOSED

    async def test_streaming_failure_trips_breaker(self, mock_builder, function_context):
        """Exception during streaming increments failure count and trips to OPEN."""
        middleware = _make_middleware(mock_builder, failure_threshold=1)

        async def failing_stream(*args, **kwargs):
            yield "chunk1"
            raise RuntimeError("Stream broken")

        call_next = Mock(side_effect=failing_stream)

        with pytest.raises(RuntimeError, match="Stream broken"):
            async for _ in middleware.function_middleware_stream("input", call_next=call_next,
                                                                 context=function_context):
                pass

        assert middleware.state == CircuitBreakerState.OPEN

        # Subsequent stream should short circuit
        chunks_open = []
        async for chunk in middleware.function_middleware_stream("input", call_next=call_next,
                                                                 context=function_context):
            chunks_open.append(chunk)

        assert len(chunks_open) == 1
        assert "Circuit breaker is OPEN" in chunks_open[0]
