# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nat.llm.dynamo_llm import create_httpx_client_with_prediction_headers
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics


async def test_prediction_headers_injected():
    """Test that prediction headers are injected into requests."""
    prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=3.0, p50=3.0, p90=4.0, p95=5.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )

    # Create a mock request to capture headers
    captured_headers = {}

    async def capture_hook(request):
        captured_headers.update(dict(request.headers))

    client = create_httpx_client_with_prediction_headers(
        prediction=prediction,
        prefix_template="test-{uuid}",
        total_requests=10,
        osl="MEDIUM",
        iat="LOW",
    )

    # Add our capture hook
    client.event_hooks["request"].append(capture_hook)

    # Make a test request (will fail, but headers will be captured)
    try:
        await client.post("http://localhost:1/test", json={})
    except Exception:
        pass

    # Prediction hook overrides x-prefix-* headers with prediction-derived values
    # remaining_calls.mean=3.0 → x-prefix-total-requests="3"
    assert "x-prefix-total-requests" in captured_headers
    assert captured_headers["x-prefix-total-requests"] == "3"
    # output_tokens.p90=200.0 (< 256) → x-prefix-osl="LOW"
    assert "x-prefix-osl" in captured_headers
    assert captured_headers["x-prefix-osl"] == "LOW"
    # interarrival_ms.mean=500.0 (>= 500) → x-prefix-iat="HIGH"
    assert "x-prefix-iat" in captured_headers
    assert captured_headers["x-prefix-iat"] == "HIGH"

    await client.aclose()
