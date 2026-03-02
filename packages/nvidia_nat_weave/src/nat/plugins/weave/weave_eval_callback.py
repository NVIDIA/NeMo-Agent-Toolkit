# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging
from contextlib import contextmanager
from typing import Any

from nat.data_models.evaluate_runtime import ProfilerResults
from nat.data_models.evaluate_runtime import UsageStats
from nat.data_models.evaluate_runtime import UsageStatsItem
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutput

logger = logging.getLogger(__name__)


class WeaveEvaluationCallback:
    """Eval callback that publishes per-item metrics and summary to Weave."""

    def __init__(self, *, project: str):
        self.project = project
        self.client = None
        self.eval_logger = None
        self.pred_loggers: dict[Any, Any] = {}
        self.eval_call = None
        self.evaluation_logger_cls = None
        self.weave_client_context = None
        self.set_call_stack = None

        try:
            from weave import EvaluationLogger
            from weave.trace.context import weave_client_context
            from weave.trace.context.call_context import set_call_stack

            self.evaluation_logger_cls = EvaluationLogger
            self.weave_client_context = weave_client_context
            self.set_call_stack = set_call_stack
        except Exception:
            # If weave import fails at runtime we no-op and let eval continue.
            logger.debug("Weave callback unavailable due to import error", exc_info=True)

    def _is_available(self) -> bool:
        return self.evaluation_logger_cls is not None and self.weave_client_context is not None

    def _initialize_client(self) -> bool:
        if not self._is_available():
            return False

        try:
            self.client = self.weave_client_context.require_weave_client()
            return self.client is not None
        except Exception:
            self.client = None
            return False

    @staticmethod
    def _prediction_inputs(item: EvalInputItem) -> dict[str, Any]:
        include = {"id", "input_obj", "expected_output_obj"}
        return item.model_dump(include=include)

    @staticmethod
    def _weave_dataset(eval_input: EvalInput) -> list[dict[str, Any]]:
        return [item.full_dataset_entry for item in eval_input.eval_input_items]

    def on_eval_started(self,
                        *,
                        workflow_alias: str,
                        eval_input: EvalInput,
                        config: Any,
                        job_id: str | None = None) -> None:
        if not self.client and not self._initialize_client():
            return

        try:
            config_dict = config.model_dump(mode="json")
            config_dict["name"] = workflow_alias

            eval_attributes = {}
            if job_id:
                eval_attributes["job_id"] = job_id

            self.eval_logger = self.evaluation_logger_cls(model=config_dict,
                                                          dataset=self._weave_dataset(eval_input),
                                                          name=workflow_alias,
                                                          eval_attributes=eval_attributes)
            self.pred_loggers = {}
            self.eval_call = getattr(self.eval_logger, "_evaluate_call", None)
        except Exception as e:
            self.eval_logger = None
            logger.warning("Failed to initialize Weave evaluation logger: %s", e)

    @contextmanager
    def evaluation_context(self):
        if self.set_call_stack and self.eval_call:
            try:
                with self.set_call_stack([self.eval_call]):
                    yield
                return
            except Exception:
                logger.warning("Failed to set Weave evaluation call context", exc_info=True)

        yield

    def on_prediction(self, *, item: EvalInputItem, output: Any) -> None:
        if not self.eval_logger:
            return
        self.pred_loggers[item.id] = self.eval_logger.log_prediction(inputs=self._prediction_inputs(item),
                                                                     output=output)

    async def a_on_usage_stats(self, *, item: EvalInputItem, usage_stats_item: UsageStatsItem) -> None:
        if not self.eval_logger or item.id not in self.pred_loggers:
            return

        pred_logger = self.pred_loggers[item.id]
        await pred_logger.alog_score(scorer="wf_runtime", score=usage_stats_item.runtime)
        await pred_logger.alog_score(scorer="wf_tokens", score=usage_stats_item.total_tokens)

    async def a_on_evaluator_score(self, *, eval_output: EvalOutput, evaluator_name: str) -> None:
        if not self.eval_logger:
            return

        coros = []
        for eval_output_item in eval_output.eval_output_items:
            pred_logger = self.pred_loggers.get(eval_output_item.id)
            if pred_logger is None:
                continue

            score_value = {"score": eval_output_item.score}
            if eval_output_item.reasoning is not None:
                score_value["reasoning"] = eval_output_item.reasoning

            coros.append(pred_logger.alog_score(scorer=evaluator_name, score=score_value))

        if coros:
            await asyncio.gather(*coros)

    async def a_on_export_flush(self) -> None:
        if not self.eval_logger:
            return

        async def _finish(pred_logger):
            if getattr(pred_logger, "_has_finished", False):
                return
            await asyncio.to_thread(pred_logger.finish)

        await asyncio.gather(*[_finish(pl) for pl in self.pred_loggers.values()])

    @staticmethod
    def _profiler_metrics(profiler_results: ProfilerResults, usage_stats: UsageStats) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if profiler_results.llm_latency_ci:
            metrics["llm_latency_p95"] = profiler_results.llm_latency_ci.p95
        if profiler_results.workflow_runtime_metrics:
            metrics["wf_runtime_p95"] = profiler_results.workflow_runtime_metrics.p95
        metrics["total_runtime"] = usage_stats.total_runtime
        return metrics

    def on_eval_summary(self,
                        *,
                        usage_stats: UsageStats,
                        evaluation_results: list[tuple[str, EvalOutput]],
                        profiler_results: ProfilerResults) -> None:
        if not self.eval_logger:
            return

        summary = {evaluator_name: eval_output.average_score for evaluator_name, eval_output in evaluation_results}
        summary.update(self._profiler_metrics(profiler_results, usage_stats))
        self.eval_logger.log_summary(summary, auto_summarize=False)
