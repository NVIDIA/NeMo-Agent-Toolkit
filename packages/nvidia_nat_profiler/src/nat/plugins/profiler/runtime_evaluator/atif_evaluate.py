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
"""ATIF-native runtime evaluators for the profiler package."""

from __future__ import annotations

import math
from datetime import datetime

from nat.atif import Step
from nat.plugins.eval.data_models.evaluator_io import EvalOutputItem
from nat.plugins.eval.evaluator.atif_base_evaluator import AtifBaseEvaluator
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample


def _iso_to_epoch(ts: str | None) -> float | None:
    """Convert ISO 8601 timestamp to epoch seconds, or None if invalid."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _step_llm_latency(step: Step) -> float | None:
    """Return the LLM start-to-end latency (seconds) for one ATIF agent step.

    Prefers the converter-written ``step.extra["invocation"]`` timing
    (``start_timestamp`` / ``end_timestamp`` in epoch seconds), which is what
    the ATOF-to-ATIF converter actually emits. Falls back to the legacy
    ``step.extra["span_event_timestamp"]`` (ISO start) paired with
    ``step.timestamp`` (ISO end) for producers that emit that shape instead.

    Returns ``None`` when neither timing source is present, or when the
    invocation timestamps are not finite, so the caller can skip the step
    rather than record a spurious zero.
    """
    extra = step.extra or {}
    invocation = extra.get("invocation")
    if isinstance(invocation, dict):
        start = invocation.get("start_timestamp")
        end = invocation.get("end_timestamp")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            try:
                latency = float(end) - float(start)
            except OverflowError:
                latency = None
            if latency is not None and math.isfinite(latency):
                return max(0.0, latency)
    start_raw = extra.get("span_event_timestamp")
    start_ts = _iso_to_epoch(start_raw) if isinstance(start_raw, str) else None
    end_ts = _iso_to_epoch(step.timestamp)
    if start_ts is not None and end_ts is not None:
        return max(0.0, end_ts - start_ts)
    return None


class AverageLLMLatencyAtifEvaluator(AtifBaseEvaluator):
    """
    ATIF-native mean latency between LLM start and end for agent steps with metrics.

    Per step, prefers the ``step.extra["invocation"]`` start/end timestamps
    written by the ATOF-to-ATIF converter, and falls back to
    ``step.extra["span_event_timestamp"]`` (start) paired with ``step.timestamp``
    (end). A step is skipped only when neither timing source is available
    (see NEP-008 for ATIF profiling metadata).
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency)

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        latencies: list[float] = []
        for step in sample.trajectory.steps:
            if step.source != "agent" or not step.metrics:
                continue
            latency = _step_llm_latency(step)
            if latency is not None:
                latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        reasoning: dict = {
            "num_llm_calls": len(latencies),
            "latencies": latencies,
        }
        return EvalOutputItem(
            id=sample.item_id,
            score=round(avg_latency, 4),
            reasoning=reasoning,
        )


class AverageWorkflowRuntimeAtifEvaluator(AtifBaseEvaluator):
    """
    ATIF-native workflow runtime per item: max(step.timestamp) - min(step.timestamp) across all steps.
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency)

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        timestamps: list[float] = []
        for step in sample.trajectory.steps:
            ts = _iso_to_epoch(step.timestamp)
            if ts is not None:
                timestamps.append(ts)

        runtime = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0
        reasoning: dict = {"steps": len(timestamps)}
        return EvalOutputItem(
            id=sample.item_id,
            score=round(max(0.0, runtime), 4),
            reasoning=reasoning,
        )


class AverageNumberOfLLMCallsAtifEvaluator(AtifBaseEvaluator):
    """
    ATIF-native count of LLM calls per item: agent steps with metrics.
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency)

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        num_calls = sum(1 for step in sample.trajectory.steps if step.source == "agent" and step.metrics is not None)
        return EvalOutputItem(
            id=sample.item_id,
            score=float(num_calls),
            reasoning={"num_llm_calls": num_calls},
        )


class AverageTokensPerLLMEndAtifEvaluator(AtifBaseEvaluator):
    """
    ATIF-native average total tokens per LLM call: (prompt_tokens + completion_tokens) from step.metrics.
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency)

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        totals: list[int] = []
        for step in sample.trajectory.steps:
            if step.source != "agent" or not step.metrics:
                continue
            prompt = step.metrics.prompt_tokens or 0
            completion = step.metrics.completion_tokens or 0
            totals.append(prompt + completion)

        avg_tokens = (sum(totals) / len(totals)) if totals else 0.0
        reasoning: dict = {"num_llm_calls": len(totals), "totals": totals}
        return EvalOutputItem(
            id=sample.item_id,
            score=round(avg_tokens, 2),
            reasoning=reasoning,
        )
