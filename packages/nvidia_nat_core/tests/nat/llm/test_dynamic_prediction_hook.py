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
"""Tests for dynamic prediction lookup with _DynamoTransport."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import httpx
import pytest

from nat.builder.context import Context
from nat.llm.dynamo_llm import DynamoPrefixContext
from nat.llm.dynamo_llm import LLMHeaderPrefix
from nat.llm.dynamo_llm import _DynamoTransport
from nat.llm.prediction_context import get_call_tracker
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup


@pytest.fixture(name="sample_trie_lookup")
def fixture_sample_trie_lookup() -> PredictionTrieLookup:
    """Create a sample trie lookup for testing."""
    prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=3.0, p50=3.0, p90=4.0, p95=5.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )

    agent_node = PredictionTrieNode(
        name="react_agent",
        predictions_by_call_index={
            1: prediction, 2: prediction
        },
        predictions_any_index=prediction,
    )

    workflow_node = PredictionTrieNode(
        name="my_workflow",
        children={"react_agent": agent_node},
        predictions_any_index=prediction,
    )

    root = PredictionTrieNode(
        name="root",
        children={"my_workflow": workflow_node},
        predictions_any_index=prediction,
    )

    return PredictionTrieLookup(root)


class TestDynamicPredictionTransport:
    """Tests for _DynamoTransport with dynamic prediction lookup."""

    async def test_transport_injects_prediction_headers(self, sample_trie_lookup):
        """Test that transport overrides headers based on context predictions."""
        # Create mock base transport
        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # Create transport with prediction lookup
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,  # Should be overridden
            osl="MEDIUM",  # Should be overridden
            iat="LOW",  # Should be overridden
            prediction_lookup=sample_trie_lookup,
        )

        ctx = Context.get()
        state = ctx._context_state

        # Reset state
        state._function_path_stack.set(None)

        # Set prefix ID
        DynamoPrefixContext.set("test-prediction")

        with ctx.push_active_function("my_workflow", input_data=None):
            with ctx.push_active_function("react_agent", input_data=None):
                # Simulate LLM call tracker increment
                tracker = get_call_tracker()
                tracker.increment(ctx.active_function.function_id)

                request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
                await transport.handle_async_request(request)

                # Get the modified request
                modified_request = mock_transport.handle_async_request.call_args[0][0]

                # Prediction values should override static config:
                # - remaining_calls.mean=3.0 -> x-prefix-total-requests="3"
                # - output_tokens.p90=200.0 -> x-prefix-osl="LOW" (< 256)
                # - interarrival_ms.mean=500.0 -> x-prefix-iat="HIGH" (>= 500)
                prefix = f"{LLMHeaderPrefix.DYNAMO.value}"
                assert modified_request.headers[f"{prefix}-total-requests"] == "3"
                assert modified_request.headers[f"{prefix}-osl"] == "LOW"
                assert modified_request.headers[f"{prefix}-iat"] == "HIGH"

        DynamoPrefixContext.clear()

    async def test_transport_uses_root_fallback(self, sample_trie_lookup):
        """Test that transport falls back to root prediction for unknown paths."""
        # Create mock base transport
        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl="MEDIUM",
            iat="LOW",
            prediction_lookup=sample_trie_lookup,
        )

        ctx = Context.get()
        state = ctx._context_state

        # Reset state
        state._function_path_stack.set(None)

        DynamoPrefixContext.set("test-fallback")

        with ctx.push_active_function("unknown_workflow", input_data=None):
            tracker = get_call_tracker()
            tracker.increment(ctx.active_function.function_id)

            request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
            await transport.handle_async_request(request)

            # Get the modified request
            modified_request = mock_transport.handle_async_request.call_args[0][0]

            # Should fall back to root aggregated predictions
            prefix = f"{LLMHeaderPrefix.DYNAMO.value}"
            assert f"{prefix}-total-requests" in modified_request.headers
            # Root prediction has remaining_calls.mean=3.0
            assert modified_request.headers[f"{prefix}-total-requests"] == "3"

        DynamoPrefixContext.clear()

    async def test_transport_handles_empty_context(self, sample_trie_lookup):
        """Test that transport handles missing context gracefully."""
        # Create mock base transport
        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl="MEDIUM",
            iat="LOW",
            prediction_lookup=sample_trie_lookup,
        )

        ctx = Context.get()
        state = ctx._context_state

        # Reset state to empty
        state._function_path_stack.set(None)
        state._active_function.set(None)

        DynamoPrefixContext.set("test-empty-context")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})

        # Should not raise an exception
        await transport.handle_async_request(request)

        # Get the modified request
        modified_request = mock_transport.handle_async_request.call_args[0][0]

        # Should still inject headers (falls back to root or static config)
        prefix = f"{LLMHeaderPrefix.DYNAMO.value}"
        assert f"{prefix}-total-requests" in modified_request.headers

        DynamoPrefixContext.clear()

    async def test_transport_no_prediction_found(self):
        """Test that transport handles case where no prediction is found."""
        # Create empty trie with no predictions
        empty_root = PredictionTrieNode(name="root")
        empty_trie = PredictionTrieLookup(empty_root)

        # Create mock base transport
        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl="MEDIUM",
            iat="LOW",
            prediction_lookup=empty_trie,
        )

        ctx = Context.get()
        state = ctx._context_state

        # Reset state
        state._function_path_stack.set(None)

        DynamoPrefixContext.set("test-no-prediction")

        with ctx.push_active_function("some_function", input_data=None):
            request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
            await transport.handle_async_request(request)

            # Get the modified request
            modified_request = mock_transport.handle_async_request.call_args[0][0]

            # Should fall back to static config values when no prediction found
            prefix = f"{LLMHeaderPrefix.DYNAMO.value}"
            assert modified_request.headers[f"{prefix}-total-requests"] == "10"  # static value
            assert modified_request.headers[f"{prefix}-osl"] == "MEDIUM"  # static value
            assert modified_request.headers[f"{prefix}-iat"] == "LOW"  # static value

        DynamoPrefixContext.clear()

    async def test_prediction_overrides_both_headers_and_agent_hints(self, sample_trie_lookup):
        """Test that predictions override both HTTP headers AND nvext.agent_hints."""
        # Create mock base transport
        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,  # Should be overridden
            osl="MEDIUM",  # Should be overridden
            iat="LOW",  # Should be overridden
            prediction_lookup=sample_trie_lookup,
        )

        ctx = Context.get()
        state = ctx._context_state

        # Reset state
        state._function_path_stack.set(None)

        DynamoPrefixContext.set("test-both-mechanisms")

        with ctx.push_active_function("my_workflow", input_data=None):
            with ctx.push_active_function("react_agent", input_data=None):
                # Simulate LLM call tracker increment
                tracker = get_call_tracker()
                tracker.increment(ctx.active_function.function_id)

                request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
                await transport.handle_async_request(request)

                # Get the modified request
                modified_request = mock_transport.handle_async_request.call_args[0][0]

                # Verify headers
                prefix = f"{LLMHeaderPrefix.DYNAMO.value}"
                assert modified_request.headers[f"{prefix}-total-requests"] == "3"
                assert modified_request.headers[f"{prefix}-osl"] == "LOW"
                assert modified_request.headers[f"{prefix}-iat"] == "HIGH"

                # Verify agent_hints
                body = json.loads(modified_request.content.decode("utf-8"))
                agent_hints = body["nvext"]["agent_hints"]
                assert agent_hints["total_requests"] == 3
                assert agent_hints["osl"] == "LOW"
                assert agent_hints["iat"] == "HIGH"

        DynamoPrefixContext.clear()
