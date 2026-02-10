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
"""End-to-end test for runtime prediction trie integration.

This test validates that all pieces work together:
1. function_path_stack gets updated when push_active_function is called
2. IntermediateStepManager increments call tracker on LLM_START
3. _DynamoTransport reads context and looks up predictions
4. Correct headers are injected based on call index
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import httpx

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.llm.dynamo_llm import DynamoPrefixContext
from nat.llm.dynamo_llm import LLMHeaderPrefix
from nat.llm.dynamo_llm import _DynamoTransport
from nat.profiler.prediction_trie import load_prediction_trie
from nat.profiler.prediction_trie import save_prediction_trie
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup


def create_test_trie() -> PredictionTrieNode:
    """Create a test trie with known predictions."""
    # Agent at call 1: 2 remaining, 500ms interarrival, 150 tokens
    call_1_prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=2.0, p50=2.0, p90=3.0, p95=4.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )

    # Agent at call 2: 1 remaining, 300ms interarrival, 100 tokens
    call_2_prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=1.0, p50=1.0, p90=2.0, p95=2.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=300.0, p50=280.0, p90=400.0, p95=450.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=100.0, p50=90.0, p90=150.0, p95=180.0),
    )

    # Agent at call 3: 0 remaining
    call_3_prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=0.0, p50=0.0, p90=0.0, p95=0.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=0.0, p50=0.0, p90=0.0, p95=0.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=80.0, p50=75.0, p90=120.0, p95=140.0),
    )

    # Aggregated for fallback
    aggregated = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=30, mean=1.0, p50=1.0, p90=2.0, p95=3.0),
        interarrival_ms=PredictionMetrics(sample_count=30, mean=400.0, p50=380.0, p90=550.0, p95=600.0),
        output_tokens=PredictionMetrics(sample_count=30, mean=110.0, p50=100.0, p90=160.0, p95=190.0),
    )

    agent_node = PredictionTrieNode(
        name="react_agent",
        predictions_by_call_index={
            1: call_1_prediction, 2: call_2_prediction, 3: call_3_prediction
        },
        predictions_any_index=aggregated,
    )

    workflow_node = PredictionTrieNode(
        name="my_workflow",
        children={"react_agent": agent_node},
        predictions_any_index=aggregated,
    )

    return PredictionTrieNode(
        name="root",
        children={"my_workflow": workflow_node},
        predictions_any_index=aggregated,
    )


async def test_e2e_prediction_headers_injected_correctly():
    """Test complete flow: context tracking -> step manager -> transport -> headers."""
    # Create and save trie
    trie = create_test_trie()

    with tempfile.TemporaryDirectory() as tmpdir:
        trie_path = Path(tmpdir) / "prediction_trie.json"
        save_prediction_trie(trie, trie_path, workflow_name="test")

        # Load trie
        loaded_trie = load_prediction_trie(trie_path)
        lookup = PredictionTrieLookup(loaded_trie)

        # Create mock base transport
        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # Create transport with prediction lookup (default use_raw_values=True)
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=lookup,
            disable_headers=False,
        )

        ctx = Context.get()
        state = ctx._context_state
        step_manager = ctx.intermediate_step_manager

        # Reset state
        state._function_path_stack.set(None)

        DynamoPrefixContext.set("e2e-test")

        with ctx.push_active_function("my_workflow", input_data=None):
            with ctx.push_active_function("react_agent", input_data=None):
                # Simulate first LLM call
                step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID="llm-1",
                        event_type=IntermediateStepType.LLM_START,
                        name="test-model",
                    ))

                request1 = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
                await transport.handle_async_request(request1)

                modified_request1 = mock_transport.handle_async_request.call_args[0][0]
                prefix = f"{LLMHeaderPrefix.DYNAMO}"

                # Call 1 raw predictions: remaining_calls.mean=2.0, output_tokens.p90=200, interarrival_ms.mean=500
                assert modified_request1.headers[f"{prefix}-total-requests"] == "2"
                assert modified_request1.headers[f"{prefix}-osl"] == "200"
                assert modified_request1.headers[f"{prefix}-iat"] == "500"

                # Simulate second LLM call
                step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID="llm-2",
                        event_type=IntermediateStepType.LLM_START,
                        name="test-model",
                    ))

                request2 = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
                await transport.handle_async_request(request2)

                modified_request2 = mock_transport.handle_async_request.call_args[0][0]

                # Call 2 raw predictions: remaining_calls.mean=1.0, output_tokens.p90=150, interarrival_ms.mean=300
                assert modified_request2.headers[f"{prefix}-total-requests"] == "1"
                assert modified_request2.headers[f"{prefix}-osl"] == "150"
                assert modified_request2.headers[f"{prefix}-iat"] == "300"

                # Simulate third LLM call
                step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID="llm-3",
                        event_type=IntermediateStepType.LLM_START,
                        name="test-model",
                    ))

                request3 = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
                await transport.handle_async_request(request3)

                modified_request3 = mock_transport.handle_async_request.call_args[0][0]

                # Call 3 raw predictions: remaining_calls.mean=0.0, output_tokens.p90=120, interarrival_ms.mean=0
                assert modified_request3.headers[f"{prefix}-total-requests"] == "0"
                assert modified_request3.headers[f"{prefix}-osl"] == "120"

        DynamoPrefixContext.clear()


async def test_e2e_fallback_to_root():
    """Test that unknown paths fall back to root predictions."""
    trie = create_test_trie()
    lookup = PredictionTrieLookup(trie)

    # Create mock base transport
    mock_response = httpx.Response(200, json={"result": "ok"})
    mock_transport = MagicMock()
    mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

    transport = _DynamoTransport(
        transport=mock_transport,
        total_requests=10,
        osl=512,
        iat=250,
        prediction_lookup=lookup,
        disable_headers=False,
    )

    ctx = Context.get()
    state = ctx._context_state
    step_manager = ctx.intermediate_step_manager

    # Reset state
    state._function_path_stack.set(None)

    DynamoPrefixContext.set("e2e-fallback")

    with ctx.push_active_function("unknown_workflow", input_data=None):
        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID="llm-unknown",
                event_type=IntermediateStepType.LLM_START,
                name="test-model",
            ))

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        prefix = f"{LLMHeaderPrefix.DYNAMO}"

        # Should fall back to root aggregated predictions (raw values)
        # remaining_calls.mean=1.0, output_tokens.p90=160, interarrival_ms.mean=400
        assert f"{prefix}-total-requests" in modified_request.headers
        assert modified_request.headers[f"{prefix}-total-requests"] == "1"
        assert modified_request.headers[f"{prefix}-osl"] == "160"
        assert modified_request.headers[f"{prefix}-iat"] == "400"

    DynamoPrefixContext.clear()


async def test_e2e_multiple_calls_in_same_context():
    """Test that call tracking increments correctly for multiple LLM calls in the same function context."""
    trie = create_test_trie()
    lookup = PredictionTrieLookup(trie)

    # Create mock base transport
    mock_response = httpx.Response(200, json={"result": "ok"})
    mock_transport = MagicMock()
    mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

    transport = _DynamoTransport(
        transport=mock_transport,
        total_requests=10,
        osl=512,
        iat=250,
        prediction_lookup=lookup,
        disable_headers=False,
    )

    ctx = Context.get()
    state = ctx._context_state
    step_manager = ctx.intermediate_step_manager

    # Reset state
    state._function_path_stack.set(None)

    DynamoPrefixContext.set("e2e-multiple-calls")

    with ctx.push_active_function("my_workflow", input_data=None):
        with ctx.push_active_function("react_agent", input_data=None):
            # First LLM call in this context
            step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID="llm-1",
                    event_type=IntermediateStepType.LLM_START,
                    name="test-model",
                ))

            request1 = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
            await transport.handle_async_request(request1)

            modified_request1 = mock_transport.handle_async_request.call_args[0][0]
            prefix = f"{LLMHeaderPrefix.DYNAMO}"

            # First call should use call_index=1 predictions
            assert modified_request1.headers[f"{prefix}-total-requests"] == "2"

            # Second LLM call in the SAME context
            step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID="llm-2",
                    event_type=IntermediateStepType.LLM_START,
                    name="test-model",
                ))

            request2 = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
            await transport.handle_async_request(request2)

            modified_request2 = mock_transport.handle_async_request.call_args[0][0]

            # Second call should use call_index=2 predictions
            assert modified_request2.headers[f"{prefix}-total-requests"] == "1"

            # Third LLM call in the SAME context
            step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID="llm-3",
                    event_type=IntermediateStepType.LLM_START,
                    name="test-model",
                ))

            request3 = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
            await transport.handle_async_request(request3)

            modified_request3 = mock_transport.handle_async_request.call_args[0][0]

            # Third call should use call_index=3 predictions
            assert modified_request3.headers[f"{prefix}-total-requests"] == "0"

    DynamoPrefixContext.clear()
