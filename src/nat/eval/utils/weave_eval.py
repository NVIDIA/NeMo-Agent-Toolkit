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

import asyncio
import logging
from typing import TYPE_CHECKING
from typing import Any

from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.usage_stats import UsageStats
from nat.eval.usage_stats import UsageStatsItem
from nat.profiler.data_models import ProfilerResults

if TYPE_CHECKING:
    from nat.eval.utils.eval_trace_ctx import EvalTraceContext

logger = logging.getLogger(__name__)


class WeaveEvaluationIntegration:  # pylint: disable=too-many-public-methods
    """
    Class to handle all Weave integration functionality.
    """

    def __init__(self, eval_trace_context: "EvalTraceContext"):
        self.available = False
        self.client = None
        self.eval_logger = None
        self.pred_loggers = {}
        self.eval_trace_context = eval_trace_context

        try:
            from weave.flow.eval_imperative import EvaluationLogger
            from weave.flow.eval_imperative import ScoreLogger
            from weave.trace.context import weave_client_context
            self.evaluation_logger_cls = EvaluationLogger  # pylint: disable=invalid-name
            self.score_logger_cls = ScoreLogger  # pylint: disable=invalid-name
            self.weave_client_context = weave_client_context
            self.available = True
        except ImportError:
            self.available = False
            # we simply don't do anything if weave is not available
            pass

    def initialize_client(self):
        """Initialize the Weave client if available."""
        if not self.available:
            return False

        try:
            self.client = self.weave_client_context.require_weave_client()
            return self.client is not None
        except Exception:
            self.client = None
            return False

    def _get_prediction_inputs(self, item: EvalInputItem):
        """Get the inputs for displaying in the UI.
        The following fields are excluded as they are too large to display in the UI:
        - full_dataset_entry
        - expected_trajectory
        - trajectory

        output_obj is excluded because it is displayed separately.
        """
        include = {"id", "input_obj", "expected_output_obj"}
        return item.model_dump(include=include)

    def _get_weave_dataset(self, eval_input: EvalInput):
        """Get the full dataset for Weave."""
        return [item.full_dataset_entry for item in eval_input.eval_input_items]

    def initialize_logger(self, workflow_alias: str, eval_input: EvalInput, config: Any):
        """Initialize the Weave evaluation logger."""
        if not self.client and not self.initialize_client():
            # lazy init the client
            return False

        try:
            weave_dataset = self._get_weave_dataset(eval_input)
            config_dict = config.model_dump(mode="json")
            config_dict["name"] = workflow_alias
            self.eval_logger = self.evaluation_logger_cls(model=config_dict, dataset=weave_dataset)
            self.pred_loggers = {}

            # Capture the current evaluation call for context propagation
            self.eval_trace_context.set_eval_call(self.eval_logger._evaluate_call)

            return True
        except Exception as e:
            self.eval_logger = None
            logger.warning("Failed to initialize Weave `EvaluationLogger`: %s", e)

            return False

    def log_prediction(self, item: EvalInputItem, output: Any):
        """Log a prediction to Weave."""
        if not self.eval_logger:
            return

        pred_logger = self.eval_logger.log_prediction(inputs=self._get_prediction_inputs(item), output=output)
        self.pred_loggers[item.id] = pred_logger

    async def log_usage_stats(self, item: EvalInputItem, usage_stats_item: UsageStatsItem):
        """Log usage stats to Weave."""
        if not self.eval_logger:
            return

        # log each usage stat as a score
        await self.pred_loggers[item.id].alog_score(scorer="wf_runtime", score=usage_stats_item.runtime)

        # log the total tokens for this item, per-llm tokens can be exported later if needed
        await self.pred_loggers[item.id].alog_score(scorer="wf_tokens", score=usage_stats_item.total_tokens)

    async def alog_score(self, eval_output: EvalOutput, evaluator_name: str):
        """Log scores for evaluation outputs."""
        if not self.eval_logger:
            return

        # Create coroutines for all score logging operations
        coros = []
        for eval_output_item in eval_output.eval_output_items:
            if eval_output_item.id in self.pred_loggers:
                coros.append(self.pred_loggers[eval_output_item.id].alog_score(
                    scorer=evaluator_name,
                    score=eval_output_item.score,
                ))

        # Execute all coroutines concurrently
        if coros:
            await asyncio.gather(*coros)

    async def afinish_loggers(self):
        """Finish all prediction loggers and wait for exports."""
        if not self.eval_logger:
            return

        async def _finish_one(pred_logger):
            if hasattr(pred_logger, '_has_finished') and not pred_logger._has_finished:
                return
            # run the *blocking* finish() in a thread so we don't nest loops
            await asyncio.to_thread(pred_logger.finish)

        await asyncio.gather(*[_finish_one(pl) for pl in self.pred_loggers.values()])

    def _log_profiler_metrics(self, profiler_results: ProfilerResults, usage_stats: UsageStats) -> dict[str, Any]:
        """Log profiler metrics to Weave."""
        profile_metrics = {}
        if profiler_results.llm_latency_ci:
            profile_metrics["llm_latency_p95"] = profiler_results.llm_latency_ci.p95
        if profiler_results.workflow_runtime_metrics:
            profile_metrics["wf_runtime_p95"] = profiler_results.workflow_runtime_metrics.p95

        profile_metrics["total_runtime"] = usage_stats.total_runtime

        return profile_metrics

    def log_summary(self,
                    usage_stats: UsageStats,
                    evaluation_results: list[tuple[str, EvalOutput]],
                    profiler_results: ProfilerResults):
        """Log summary statistics to Weave."""
        if not self.eval_logger:
            return

        summary = {}
        # add evaluation results to the summary
        for evaluator_name, eval_output in evaluation_results:
            summary[evaluator_name] = eval_output.average_score

        # add profiler metrics to the summary
        profile_metrics = self._log_profiler_metrics(profiler_results, usage_stats)
        summary.update(profile_metrics)

        # Log the summary to finish the evaluation, disable auto-summarize
        # as we will be adding profiler metrics to the summary
        self.eval_logger.log_summary(summary, auto_summarize=False)
