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

from __future__ import annotations

import logging
from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

if TYPE_CHECKING:
    from nat.eval.evaluator.evaluator_model import EvalInputItem

logger = logging.getLogger(__name__)


@dataclass
class EvalResultItem:
    """Per-dataset-item result from evaluation."""
    item_id: Any
    input_obj: Any  # the question / input
    expected_output: Any  # ground truth
    actual_output: Any  # model's answer
    scores: dict[str, float]  # evaluator_name -> score for this item
    reasoning: dict[str, Any]  # evaluator_name -> reasoning/explanation
    total_tokens: int | None = None
    llm_latency: float | None = None  # p95 LLM latency in seconds
    runtime: float | None = None  # total wall-clock time in seconds
    root_span_id: int | None = None  # Pre-generated OTEL root span_id for eager trace linking


@dataclass
class EvalResult:
    """Full result of a single evaluation run.

    The ``metric_scores`` and ``items`` fields are always populated.
    The remaining fields are optional context that exporters (e.g.
    ``FileEvalCallback``) can use to persist richer output without
    breaking callbacks that only inspect scores.
    """

    metric_scores: dict[str, float]  # evaluator_name -> average score
    items: list[EvalResultItem]  # per-item breakdown

    evaluation_outputs: list[tuple[str, Any]] = field(default_factory=list)
    workflow_output_json: str | None = None
    atif_workflow_output_json: str | None = None
    run_config: Any | None = None
    effective_config: Any | None = None
    output_dir: Path | None = None


def build_eval_result(
    *,
    eval_input_items: list,
    evaluation_results: list[tuple[str, Any]],
    metric_scores: dict[str, float],
    usage_stats: Any | None = None,
    item_span_ids: dict[str, int] | None = None,
    workflow_output_json: str | None = None,
    atif_workflow_output_json: str | None = None,
    run_config: Any | None = None,
    effective_config: Any | None = None,
    output_dir: Path | None = None,
) -> EvalResult:
    """Build an EvalResult from raw evaluation data.

    This is the single place that maps eval-input items + evaluator outputs
    into the callback-friendly ``EvalResult`` / ``EvalResultItem`` structure.
    """
    cb_items: list[EvalResultItem] = []
    for input_item in eval_input_items:
        per_item_scores: dict[str, float] = {}
        per_item_reasoning: dict[str, Any] = {}
        for eval_name, eval_output in evaluation_results:
            for output_item in eval_output.eval_output_items:
                if str(output_item.id) == str(input_item.id):
                    score_val = output_item.score
                    if isinstance(score_val, (int, float)):
                        per_item_scores[eval_name] = float(score_val)
                    per_item_reasoning[eval_name] = output_item.reasoning
                    break

        usage_item = None
        if usage_stats is not None:
            usage_item = usage_stats.usage_stats_items.get(input_item.id)

        cb_items.append(
            EvalResultItem(
                item_id=input_item.id,
                input_obj=input_item.input_obj,
                expected_output=input_item.expected_output_obj,
                actual_output=input_item.output_obj,
                scores=per_item_scores,
                reasoning=per_item_reasoning,
                total_tokens=usage_item.total_tokens if usage_item else None,
                llm_latency=usage_item.llm_latency if usage_item else None,
                runtime=usage_item.runtime if usage_item else None,
                root_span_id=(item_span_ids.get(str(input_item.id)) if item_span_ids else None),
            ))
    return EvalResult(
        metric_scores=metric_scores,
        items=cb_items,
        evaluation_outputs=evaluation_results,
        workflow_output_json=workflow_output_json,
        atif_workflow_output_json=atif_workflow_output_json,
        run_config=run_config,
        effective_config=effective_config,
        output_dir=output_dir,
    )


class EvalCallback(Protocol):

    def on_dataset_loaded(self, *, dataset_name: str, items: list[EvalInputItem]) -> None:
        ...

    def on_eval_complete(self, result: EvalResult) -> None:
        ...

    # Optional hooks for provider-specific eval metric exporting.
    # Implementations may provide any subset of these methods.
    def on_eval_started(self, *, workflow_alias: str, eval_input: Any, config: Any, job_id: str | None = None) -> None:
        ...

    def on_prediction(self, *, item: Any, output: Any) -> None:
        ...

    async def a_on_usage_stats(self, *, item: Any, usage_stats_item: Any) -> None:
        ...

    async def a_on_evaluator_score(self, *, eval_output: Any, evaluator_name: str) -> None:
        ...

    async def a_on_export_flush(self) -> None:
        ...

    def on_eval_summary(self, *, usage_stats: Any, evaluation_results: Any, profiler_results: Any) -> None:
        ...

    def evaluation_context(self):
        ...


class EvalCallbackManager:
    """
    Dispatches eval lifecycle callbacks to registered integrations.

    Maintainer note:
    Keep this callback surface stable for provider plugins. If we later adopt
    an internal event-subscriber bus (typed events, async fan-out, retries),
    it can be introduced behind this manager as a near-term design evolution.
    """

    def __init__(self) -> None:
        self._callbacks: list[EvalCallback] = []

    def register(self, callback: EvalCallback) -> None:
        self._callbacks.append(callback)

    @property
    def has_callbacks(self) -> bool:
        return bool(self._callbacks)

    @property
    def needs_root_span_ids(self) -> bool:
        """Check if any registered callback declares it needs pre-generated root span_ids."""
        for cb in self._callbacks:
            if getattr(cb, "needs_root_span_ids", False):
                return True
        return False

    def on_dataset_loaded(self, *, dataset_name: str, items: list[EvalInputItem]) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "on_dataset_loaded", None)
            if not fn:
                continue
            try:
                fn(dataset_name=dataset_name, items=items)
            except Exception:
                logger.exception("EvalCallback %s.on_dataset_loaded failed", type(cb).__name__)

    def on_eval_started(self, *, workflow_alias: str, eval_input: Any, config: Any, job_id: str | None = None) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "on_eval_started", None)
            if not fn:
                continue
            try:
                fn(workflow_alias=workflow_alias, eval_input=eval_input, config=config, job_id=job_id)
            except Exception:
                logger.exception("EvalCallback %s.on_eval_started failed", type(cb).__name__)

    def on_prediction(self, *, item: Any, output: Any) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "on_prediction", None)
            if not fn:
                continue
            try:
                fn(item=item, output=output)
            except Exception:
                logger.exception("EvalCallback %s.on_prediction failed", type(cb).__name__)

    async def a_on_usage_stats(self, *, item: Any, usage_stats_item: Any) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "a_on_usage_stats", None)
            if not fn:
                continue
            try:
                await fn(item=item, usage_stats_item=usage_stats_item)
            except Exception:
                logger.exception("EvalCallback %s.a_on_usage_stats failed", type(cb).__name__)

    async def a_on_evaluator_score(self, *, eval_output: Any, evaluator_name: str) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "a_on_evaluator_score", None)
            if not fn:
                continue
            try:
                await fn(eval_output=eval_output, evaluator_name=evaluator_name)
            except Exception:
                logger.exception("EvalCallback %s.a_on_evaluator_score failed", type(cb).__name__)

    async def a_on_export_flush(self) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "a_on_export_flush", None)
            if not fn:
                continue
            try:
                await fn()
            except Exception:
                logger.exception("EvalCallback %s.a_on_export_flush failed", type(cb).__name__)

    def on_eval_summary(self, *, usage_stats: Any, evaluation_results: Any, profiler_results: Any) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "on_eval_summary", None)
            if not fn:
                continue
            try:
                fn(usage_stats=usage_stats, evaluation_results=evaluation_results, profiler_results=profiler_results)
            except Exception:
                logger.exception("EvalCallback %s.on_eval_summary failed", type(cb).__name__)

    @contextmanager
    def evaluation_context(self):
        with ExitStack() as stack:
            for cb in self._callbacks:
                fn = getattr(cb, "evaluation_context", None)
                if not fn:
                    continue
                try:
                    stack.enter_context(fn())
                except Exception:
                    logger.exception("EvalCallback %s.evaluation_context setup failed", type(cb).__name__)
            yield

    def on_eval_complete(self, result: EvalResult) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "on_eval_complete", None)
            if not fn:
                continue
            try:
                fn(result)
            except Exception:
                logger.exception("EvalCallback %s.on_eval_complete failed", type(cb).__name__)

    def get_eval_project_name(self) -> str | None:
        """Get an eval-specific project name from the first callback that supports it."""
        for cb in self._callbacks:
            fn = getattr(cb, "get_eval_project_name", None)
            if fn:
                try:
                    return fn()
                except Exception:
                    logger.debug("get_eval_project_name failed for %s", type(cb).__name__, exc_info=True)
        return None
