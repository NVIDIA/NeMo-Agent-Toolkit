# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from nat.data_models.intermediate_step import IntermediateStepType
from nat.eval.evaluator.base_evaluator import BaseEvaluator
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor


@dataclass
class _CallTiming:
    start_ts: float | None = None
    end_ts: float | None = None

    @property
    def latency(self) -> float | None:
        if self.start_ts is None or self.end_ts is None:
            return None
        return max(0.0, self.end_ts - self.start_ts)


class AverageLLMLatencyEvaluator(BaseEvaluator):
    """
    Mean difference between connected LLM_START and LLM_END events (same UUID).
    The score is the average latency in seconds for the item. Reasoning contains per-call latencies.
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating Avg LLM Latency")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:  # noqa: D401
        calls: dict[str, _CallTiming] = defaultdict(_CallTiming)

        for step in (IntermediatePropertyAdaptor.from_intermediate_step(s) for s in item.trajectory):
            if step.event_type == IntermediateStepType.LLM_START:
                calls[step.UUID].start_ts = step.event_timestamp
            elif step.event_type == IntermediateStepType.LLM_END:
                calls[step.UUID].end_ts = step.event_timestamp

        latencies = [ct.latency for ct in calls.values() if ct.latency is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        reasoning = {
            "num_llm_calls": len(latencies),
            "latencies": latencies,
        }
        return EvalOutputItem(id=item.id, score=round(avg_latency, 4), reasoning=reasoning)


class AverageWorkflowRuntimeEvaluator(BaseEvaluator):
    """
    Average workflow runtime per item: max(event_timestamp) - min(event_timestamp) across the trajectory.
    The score is the runtime in seconds for the item.
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating Avg Workflow Runtime")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:  # noqa: D401
        if not item.trajectory:
            return EvalOutputItem(id=item.id, score=0.0, reasoning={"note": "no steps"})

        timestamps = [s.event_timestamp for s in item.trajectory]
        runtime = max(timestamps) - min(timestamps)
        return EvalOutputItem(id=item.id, score=round(max(0.0, runtime), 4), reasoning={"steps": len(timestamps)})


class AverageNumberOfLLMCallsEvaluator(BaseEvaluator):
    """
    Average number of LLM calls per item. The score is the count for the item.
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating Avg # LLM Calls")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:  # noqa: D401
        num_calls = sum(1 for s in item.trajectory if s.event_type == IntermediateStepType.LLM_END)
        return EvalOutputItem(id=item.id, score=float(num_calls), reasoning={"num_llm_end": num_calls})


class AverageTokensPerLLMEndEvaluator(BaseEvaluator):
    """
    Average total tokens per LLM_END event: sum of prompt and completion tokens if available.
    The score is the average tokens per LLM_END for the item (0 if none).
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating Avg Tokens/LLM_END")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:  # noqa: D401
        totals: list[int] = []
        for step in (IntermediatePropertyAdaptor.from_intermediate_step(s) for s in item.trajectory):
            if step.event_type == IntermediateStepType.LLM_END:
                total_tokens = step.token_usage.total_tokens
                # If framework doesn't set total, compute from prompt+completion
                if total_tokens == 0:
                    total_tokens = step.token_usage.prompt_tokens + step.token_usage.completion_tokens
                totals.append(total_tokens)

        avg_tokens = (sum(totals) / len(totals)) if totals else 0.0
        reasoning = {
            "num_llm_end": len(totals),
            "totals": totals,
        }
        return EvalOutputItem(id=item.id, score=round(avg_tokens, 2), reasoning=reasoning)
