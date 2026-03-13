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
"""Tests for ATIF-native runtime evaluators."""

from __future__ import annotations

import pytest

from nat.data_models.atif import ATIFAgentConfig
from nat.data_models.atif import ATIFStep
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.atif import Metrics
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.profiler.runtime_evaluator.atif_evaluate import AverageLLMLatencyAtifEvaluator
from nat.plugins.profiler.runtime_evaluator.atif_evaluate import AverageNumberOfLLMCallsAtifEvaluator
from nat.plugins.profiler.runtime_evaluator.atif_evaluate import AverageTokensPerLLMEndAtifEvaluator
from nat.plugins.profiler.runtime_evaluator.atif_evaluate import AverageWorkflowRuntimeAtifEvaluator
from nat.plugins.profiler.runtime_evaluator.atif_evaluate import _iso_to_epoch


def _make_sample(
    item_id: str | int,
    steps: list[ATIFStep],
) -> AtifEvalSample:
    trajectory = ATIFTrajectory(
        session_id="test-session",
        agent=ATIFAgentConfig(name="test-agent", version="0.0.0"),
        steps=steps,
    )
    return AtifEvalSample(item_id=item_id, trajectory=trajectory, metadata={})


# --- _iso_to_epoch conversion (type conversion is critical path) ---


@pytest.mark.parametrize(
    "ts,expected",
    [
        ("2024-01-01T12:00:00", True),
        ("2024-01-01T12:00:00Z", True),
        ("2024-01-01T12:00:00+00:00", True),
        (None, False),
        ("", False),
        ("not-a-date", False),
    ],
)
def test_iso_to_epoch_conversion(ts, expected):
    """Verify ISO timestamp parsing returns epoch float or None for invalid input."""
    result = _iso_to_epoch(ts)
    if expected:
        assert result is not None
        assert isinstance(result, float)
    else:
        assert result is None


# --- evaluate_atif_item: core latency computation ---


async def test_evaluate_atif_item_single_valid_latency():
    """Agent step with metrics and span_event_timestamp yields correct latency."""
    steps = [
        ATIFStep(
            step_id=1,
            source="agent",
            timestamp="2024-01-01T12:00:05",
            metrics=Metrics(prompt_tokens=10, completion_tokens=20),
            extra={"span_event_timestamp": "2024-01-01T12:00:00"},
        ),
    ]
    sample = _make_sample("item-1", steps)
    evaluator = AverageLLMLatencyAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.id == "item-1"
    assert result.score == pytest.approx(5.0, abs=1e-4)
    assert result.reasoning["num_llm_calls"] == 1
    assert result.reasoning["latencies"] == pytest.approx([5.0], abs=1e-4)


async def test_evaluate_atif_item_multiple_latencies_averaged():
    """Multiple agent steps with valid timestamps yield correct average."""
    steps = [
        ATIFStep(
            step_id=1,
            source="agent",
            timestamp="2024-01-01T12:00:02",
            metrics=Metrics(prompt_tokens=1),
            extra={"span_event_timestamp": "2024-01-01T12:00:00"},
        ),
        ATIFStep(
            step_id=2,
            source="agent",
            timestamp="2024-01-01T12:00:08",
            metrics=Metrics(prompt_tokens=1),
            extra={"span_event_timestamp": "2024-01-01T12:00:04"},
        ),
    ]
    sample = _make_sample("item-2", steps)
    evaluator = AverageLLMLatencyAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.score == pytest.approx(3.0, abs=1e-4)  # (2 + 4) / 2
    assert result.reasoning["num_llm_calls"] == 2
    assert result.reasoning["latencies"] == pytest.approx([2.0, 4.0], abs=1e-4)


# --- evaluate_atif_item: edge cases (avoid false negatives) ---


async def test_evaluate_atif_item_empty_trajectory():
    """Empty trajectory returns 0.0 without crashing."""
    sample = _make_sample("empty", [])
    evaluator = AverageLLMLatencyAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.score == 0.0
    assert result.reasoning["num_llm_calls"] == 0
    assert result.reasoning["latencies"] == []


async def test_evaluate_atif_item_no_agent_steps():
    """User/system steps only yield 0.0."""
    steps = [
        ATIFStep(step_id=1, source="user", message="hello"),
        ATIFStep(step_id=2, source="system", message="ok"),
    ]
    sample = _make_sample("no-agent", steps)
    evaluator = AverageLLMLatencyAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.score == 0.0
    assert result.reasoning["num_llm_calls"] == 0


async def test_evaluate_atif_item_agent_with_metrics_no_span_timestamp():
    """Agent steps with metrics but no span_event_timestamp: skip, 0.0."""
    steps = [
        ATIFStep(
            step_id=1,
            source="agent",
            timestamp="2024-01-01T12:00:05",
            metrics=Metrics(prompt_tokens=10),
            extra=None,
        ),
    ]
    sample = _make_sample("no-span", steps)
    evaluator = AverageLLMLatencyAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.score == 0.0
    assert result.reasoning["num_llm_calls"] == 0


async def test_evaluate_atif_item_timestamp_none_skips_step():
    """Agent step with timestamp=None is skipped."""
    steps = [
        ATIFStep(
            step_id=1,
            source="agent",
            timestamp=None,
            metrics=Metrics(prompt_tokens=1),
            extra={"span_event_timestamp": "2024-01-01T12:00:00"},
        ),
    ]
    sample = _make_sample("ts-none", steps)
    evaluator = AverageLLMLatencyAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.score == 0.0
    assert result.reasoning["num_llm_calls"] == 0


async def test_evaluate_atif_item_invalid_span_timestamp_skips_step():
    """span_event_timestamp as non-string (e.g. dict) is skipped."""
    steps = [
        ATIFStep(
            step_id=1,
            source="agent",
            timestamp="2024-01-01T12:00:05",
            metrics=Metrics(prompt_tokens=1),
            extra={"span_event_timestamp": {
                "invalid": "dict"
            }},
        ),
    ]
    sample = _make_sample("bad-span", steps)
    evaluator = AverageLLMLatencyAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.score == 0.0


async def test_evaluate_atif_item_mixed_valid_and_invalid_steps():
    """One valid and one invalid step: only valid contributes to average."""
    steps = [
        ATIFStep(
            step_id=1,
            source="agent",
            timestamp="2024-01-01T12:00:05",
            metrics=Metrics(prompt_tokens=1),
            extra={"span_event_timestamp": "2024-01-01T12:00:00"},
        ),
        ATIFStep(
            step_id=2,
            source="agent",
            timestamp="2024-01-01T12:00:10",
            metrics=Metrics(prompt_tokens=1),
            extra=None,
        ),
    ]
    sample = _make_sample("mixed", steps)
    evaluator = AverageLLMLatencyAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.score == pytest.approx(5.0, abs=1e-4)
    assert result.reasoning["num_llm_calls"] == 1


# --- evaluate_atif_fn: batch orchestration ---


async def test_evaluate_atif_fn_batch_aggregation():
    """evaluate_atif_fn aggregates multiple samples and computes average_score."""
    sample1 = _make_sample(
        "a",
        [
            ATIFStep(
                step_id=1,
                source="agent",
                timestamp="2024-01-01T12:00:02",
                metrics=Metrics(prompt_tokens=1),
                extra={"span_event_timestamp": "2024-01-01T12:00:00"},
            ),
        ],
    )
    sample2 = _make_sample(
        "b",
        [
            ATIFStep(
                step_id=1,
                source="agent",
                timestamp="2024-01-01T12:00:06",
                metrics=Metrics(prompt_tokens=1),
                extra={"span_event_timestamp": "2024-01-01T12:00:00"},
            ),
        ],
    )
    evaluator = AverageLLMLatencyAtifEvaluator()

    output = await evaluator.evaluate_atif_fn([sample1, sample2])

    assert output.average_score == pytest.approx(4.0, abs=1e-2)  # (2 + 6) / 2
    assert len(output.eval_output_items) == 2
    assert output.eval_output_items[0].id == "a"
    assert output.eval_output_items[1].id == "b"


# --- AverageWorkflowRuntimeAtifEvaluator ---


async def test_workflow_runtime_atif_valid_timestamps():
    """Multiple steps with timestamps yield correct runtime."""
    steps = [
        ATIFStep(step_id=1, source="user", message="hi", timestamp="2024-01-01T12:00:00"),
        ATIFStep(step_id=2, source="agent", message="hello", timestamp="2024-01-01T12:00:05"),
        ATIFStep(step_id=3, source="user", message="bye", timestamp="2024-01-01T12:00:10"),
    ]
    sample = _make_sample("wf-1", steps)
    evaluator = AverageWorkflowRuntimeAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.id == "wf-1"
    assert result.score == pytest.approx(10.0, abs=1e-4)
    assert result.reasoning["steps"] == 3


async def test_workflow_runtime_atif_steps_without_timestamp_skipped():
    """Steps with timestamp=None are skipped; valid timestamps still compute runtime."""
    steps = [
        ATIFStep(step_id=1, source="user", message="hi", timestamp=None),
        ATIFStep(step_id=2, source="agent", message="ok", timestamp="2024-01-01T12:00:00"),
        ATIFStep(step_id=3, source="user", message="bye", timestamp="2024-01-01T12:00:03"),
    ]
    sample = _make_sample("partial", steps)
    evaluator = AverageWorkflowRuntimeAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.score == pytest.approx(3.0, abs=1e-4)
    assert result.reasoning["steps"] == 2


# --- AverageNumberOfLLMCallsAtifEvaluator ---


async def test_num_llm_calls_atif_counts_agent_steps_with_metrics():
    """Agent steps with metrics are counted as LLM calls."""
    steps = [
        ATIFStep(step_id=1, source="user", message="hi"),
        ATIFStep(step_id=2, source="agent", message="ok", metrics=Metrics(prompt_tokens=10)),
        ATIFStep(step_id=3, source="agent", message="ok", metrics=Metrics(prompt_tokens=5)),
    ]
    sample = _make_sample("calls-1", steps)
    evaluator = AverageNumberOfLLMCallsAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.score == 2.0
    assert result.reasoning["num_llm_calls"] == 2


# --- AverageTokensPerLLMEndAtifEvaluator ---


async def test_tokens_per_llm_end_atif_averages_from_metrics():
    """Average tokens computed from prompt_tokens + completion_tokens per agent step."""
    steps = [
        ATIFStep(
            step_id=1,
            source="agent",
            message="ok",
            metrics=Metrics(prompt_tokens=100, completion_tokens=50),
        ),
        ATIFStep(
            step_id=2,
            source="agent",
            message="ok",
            metrics=Metrics(prompt_tokens=200, completion_tokens=100),
        ),
    ]
    sample = _make_sample("tokens-1", steps)
    evaluator = AverageTokensPerLLMEndAtifEvaluator()

    result = await evaluator.evaluate_atif_item(sample)

    assert result.score == pytest.approx(225.0, abs=1e-2)  # (150 + 300) / 2
    assert result.reasoning["num_llm_calls"] == 2


# --- Registration ---


async def test_register_avg_llm_latency_exposes_evaluate_atif_fn():
    """Registration wires evaluate_atif_fn so harness can dispatch ATIF lane."""
    from unittest.mock import MagicMock

    from nat.plugins.profiler.runtime_evaluator.register import AverageLLMLatencyConfig
    from nat.plugins.profiler.runtime_evaluator.register import register_avg_llm_latency_evaluator

    builder = MagicMock()
    builder.get_max_concurrency = MagicMock(return_value=4)
    config = AverageLLMLatencyConfig()

    async with register_avg_llm_latency_evaluator(config=config, builder=builder) as evaluator_info:
        assert hasattr(evaluator_info, "evaluate_fn")
        assert hasattr(evaluator_info, "evaluate_atif_fn")
        assert callable(getattr(evaluator_info, "evaluate_atif_fn", None))
