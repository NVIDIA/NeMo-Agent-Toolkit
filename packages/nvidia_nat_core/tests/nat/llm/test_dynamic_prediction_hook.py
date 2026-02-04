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

import pytest

from nat.builder.context import Context
from nat.llm.dynamo_llm import _create_dynamic_prediction_hook
from nat.llm.dynamo_llm import create_httpx_client_with_dynamo_hooks
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


class MockRequest:
    """Mock httpx.Request for testing."""

    def __init__(self):
        self.headers = {}


async def test_dynamic_hook_injects_headers(sample_trie_lookup):
    """Test that dynamic hook overrides x-prefix-* headers based on context predictions."""
    ctx = Context.get()
    state = ctx._context_state

    # Reset state
    state._function_path_stack.set(None)

    hook = _create_dynamic_prediction_hook(sample_trie_lookup)

    with ctx.push_active_function("my_workflow", input_data=None):
        with ctx.push_active_function("react_agent", input_data=None):
            # Simulate LLM call tracker increment (normally done by step manager)
            tracker = get_call_tracker()
            tracker.increment(ctx.active_function.function_id)

            request = MockRequest()
            await hook(request)

            # Prediction values are converted to x-prefix-* headers:
            # - remaining_calls.mean=3.0 -> x-prefix-total-requests="3"
            # - output_tokens.p90=200.0 -> x-prefix-osl="LOW" (< 256)
            # - interarrival_ms.mean=500.0 -> x-prefix-iat="HIGH" (>= 500)
            assert "x-prefix-total-requests" in request.headers
            assert request.headers["x-prefix-total-requests"] == "3"
            assert request.headers["x-prefix-osl"] == "LOW"
            assert request.headers["x-prefix-iat"] == "HIGH"


async def test_dynamic_hook_uses_root_fallback(sample_trie_lookup):
    """Test that hook falls back to root prediction for unknown paths."""
    ctx = Context.get()
    state = ctx._context_state

    # Reset state
    state._function_path_stack.set(None)

    hook = _create_dynamic_prediction_hook(sample_trie_lookup)

    with ctx.push_active_function("unknown_workflow", input_data=None):
        tracker = get_call_tracker()
        tracker.increment(ctx.active_function.function_id)

        request = MockRequest()
        await hook(request)

        # Should fall back to root aggregated predictions
        assert "x-prefix-total-requests" in request.headers


async def test_dynamic_hook_handles_empty_context(sample_trie_lookup):
    """Test that hook handles missing context gracefully."""
    ctx = Context.get()
    state = ctx._context_state

    # Reset state to empty
    state._function_path_stack.set(None)
    state._active_function.set(None)

    hook = _create_dynamic_prediction_hook(sample_trie_lookup)

    request = MockRequest()
    # Should not raise an exception
    await hook(request)

    # Should still inject headers from root fallback
    assert "x-prefix-total-requests" in request.headers


async def test_dynamic_hook_no_prediction_found():
    """Test that hook handles case where no prediction is found."""
    # Create empty trie with no predictions
    empty_root = PredictionTrieNode(name="root")
    empty_trie = PredictionTrieLookup(empty_root)

    ctx = Context.get()
    state = ctx._context_state

    # Reset state
    state._function_path_stack.set(None)

    hook = _create_dynamic_prediction_hook(empty_trie)

    with ctx.push_active_function("some_function", input_data=None):
        request = MockRequest()
        await hook(request)

        # Headers should not be overridden when no prediction found
        # (the static Dynamo hook would set them, but this hook runs after)
        assert "x-prefix-total-requests" not in request.headers


async def test_client_includes_prediction_hook_when_lookup_provided(sample_trie_lookup):
    """Test that client includes prediction hook when trie_lookup is provided."""
    client = create_httpx_client_with_dynamo_hooks(
        prefix_template="test-{uuid}",
        total_requests=10,
        osl="MEDIUM",
        iat="LOW",
        prediction_lookup=sample_trie_lookup,
    )

    # Should have 2 hooks: dynamo prefix + prediction
    assert len(client.event_hooks["request"]) == 2

    await client.aclose()


async def test_client_works_without_prediction_lookup():
    """Test that client works when prediction_lookup is None."""
    client = create_httpx_client_with_dynamo_hooks(
        prefix_template="test-{uuid}",
        total_requests=10,
        osl="MEDIUM",
        iat="LOW",
        prediction_lookup=None,
    )

    # Should have 1 hook: dynamo prefix only
    assert len(client.event_hooks["request"]) == 1

    await client.aclose()
