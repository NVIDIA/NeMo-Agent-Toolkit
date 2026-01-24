# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.builder.context import Context
from nat.llm.dynamo_llm import _create_dynamic_prediction_hook
from nat.llm.prediction_context import get_call_tracker
from nat.profiler.prediction_trie import PredictionTrieLookup
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode


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
    """Test that dynamic hook injects prediction headers based on context."""
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

            assert "x-nat-remaining-llm-calls" in request.headers
            assert request.headers["x-nat-remaining-llm-calls"] == "3"
            assert request.headers["x-nat-interarrival-ms"] == "500"
            assert request.headers["x-nat-expected-output-tokens"] == "200"


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
        assert "x-nat-remaining-llm-calls" in request.headers


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
    assert "x-nat-remaining-llm-calls" in request.headers


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

        # Headers should not be injected when no prediction found
        assert "x-nat-remaining-llm-calls" not in request.headers
