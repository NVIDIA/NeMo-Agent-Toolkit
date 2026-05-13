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
"""Tests for TimeoutMiddleware."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.timeout.timeout_middleware import TimeoutMiddleware
from nat.middleware.timeout.timeout_middleware_config import TimeoutMiddlewareConfig

# ==================== Fixtures ====================


@pytest.fixture(name="mock_builder")
def fixture_mock_builder():
    """Create a mock builder with all required methods."""
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
        name="test_function",
        config=Mock(),
        description="A test function",
        input_schema=None,
        single_output_schema=type(None),
        stream_output_schema=type(None),
    )


def _make_middleware(
    mock_builder: Mock,
    *,
    timeout: float,
    timeout_message: str | None = None,
) -> TimeoutMiddleware:
    """Create a TimeoutMiddleware with the given timeout."""
    kwargs: dict[str, Any] = {"timeout": timeout}
    if timeout_message is not None:
        kwargs["timeout_message"] = timeout_message
    config: TimeoutMiddlewareConfig = TimeoutMiddlewareConfig(**kwargs)
    return TimeoutMiddleware(config=config, builder=mock_builder)


# ==================== Single Invocation Tests ====================


class TestTimeoutMiddlewareInvoke:
    """Tests for function_middleware_invoke timeout enforcement."""

    async def test_completes_within_timeout(self, mock_builder, function_context):
        """Function that completes within the timeout returns normally."""
        middleware: TimeoutMiddleware = _make_middleware(mock_builder, timeout=5.0)

        async def fast_function(*args, **kwargs):
            return "result"

        call_next: AsyncMock = AsyncMock(side_effect=fast_function)

        result = await middleware.function_middleware_invoke(
            "input",
            call_next=call_next,
            context=function_context,
        )

        assert result == "result"
        call_next.assert_called_once()

    async def test_exceeds_timeout_raises(self, mock_builder, function_context):
        """Function that exceeds the timeout raises TimeoutError with the configured message."""
        middleware: TimeoutMiddleware = _make_middleware(mock_builder, timeout=0.05)

        async def slow_function(*args, **kwargs):
            await asyncio.sleep(10)
            return "never"

        call_next: AsyncMock = AsyncMock(side_effect=slow_function)

        with pytest.raises(TimeoutError, match=r"Execution exceeded the configured timeout of 0\.05s"):
            await middleware.function_middleware_invoke(
                "input",
                call_next=call_next,
                context=function_context,
            )

    async def test_propagates_function_exception(self, mock_builder, function_context):
        """Non-timeout exceptions from the function propagate unchanged."""
        middleware: TimeoutMiddleware = _make_middleware(mock_builder, timeout=5.0)

        call_next: AsyncMock = AsyncMock(side_effect=ValueError("bad input"))

        with pytest.raises(ValueError, match="bad input"):
            await middleware.function_middleware_invoke(
                "input",
                call_next=call_next,
                context=function_context,
            )

    async def test_custom_timeout_message(self, mock_builder, function_context):
        """Custom timeout_message is used in the TimeoutError."""
        middleware: TimeoutMiddleware = _make_middleware(
            mock_builder,
            timeout=0.01,
            timeout_message="LLM call timed out, try a smaller prompt",
        )

        async def slow_function(*args, **kwargs):
            await asyncio.sleep(10)

        call_next: AsyncMock = AsyncMock(side_effect=slow_function)

        with pytest.raises(TimeoutError, match="LLM call timed out, try a smaller prompt"):
            await middleware.function_middleware_invoke(
                "input",
                call_next=call_next,
                context=function_context,
            )


# ==================== Streaming Tests ====================


class TestTimeoutMiddlewareStream:
    """Tests for function_middleware_stream timeout enforcement."""

    async def test_stream_completes_within_timeout(self, mock_builder, function_context):
        """Stream that completes within the timeout yields all chunks."""
        middleware: TimeoutMiddleware = _make_middleware(mock_builder, timeout=5.0)

        async def fast_stream(*args, **kwargs):
            for i in range(3):
                yield f"chunk_{i}"

        collected: list[str] = []
        async for chunk in middleware.function_middleware_stream(
                "input",
                call_next=fast_stream,
                context=function_context,
        ):
            collected.append(chunk)

        assert collected == ["chunk_0", "chunk_1", "chunk_2"]

    async def test_stream_exceeds_timeout_raises(self, mock_builder, function_context):
        """Stream that exceeds the timeout raises TimeoutError with the configured message."""
        middleware: TimeoutMiddleware = _make_middleware(mock_builder, timeout=0.05)

        async def slow_stream(*args, **kwargs):
            yield "chunk_0"
            await asyncio.sleep(10)
            yield "chunk_1"

        with pytest.raises(TimeoutError, match=r"Execution exceeded the configured timeout of 0\.05s"):
            async for _ in middleware.function_middleware_stream(
                    "input",
                    call_next=slow_stream,
                    context=function_context,
            ):
                pass

    async def test_stream_propagates_function_exception(self, mock_builder, function_context):
        """Non-timeout exceptions from the stream propagate unchanged."""
        middleware: TimeoutMiddleware = _make_middleware(mock_builder, timeout=5.0)

        async def error_stream(*args, **kwargs):
            yield "chunk_0"
            raise RuntimeError("stream failed")

        with pytest.raises(RuntimeError, match="stream failed"):
            async for _ in middleware.function_middleware_stream(
                    "input",
                    call_next=error_stream,
                    context=function_context,
            ):
                pass

    async def test_stream_custom_timeout_message(self, mock_builder, function_context):
        """Custom timeout_message is used in the streaming TimeoutError."""
        middleware: TimeoutMiddleware = _make_middleware(
            mock_builder,
            timeout=0.01,
            timeout_message="Stream took too long",
        )

        async def slow_stream(*args, **kwargs):
            await asyncio.sleep(10)
            yield "never"

        with pytest.raises(TimeoutError, match="Stream took too long"):
            async for _ in middleware.function_middleware_stream(
                    "input",
                    call_next=slow_stream,
                    context=function_context,
            ):
                pass
