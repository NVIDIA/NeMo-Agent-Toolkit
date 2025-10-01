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

import logging
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any

import httpx

from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.usage_stats import UsageStats
from nat.profiler.data_models import ProfilerResults

if TYPE_CHECKING:
    from nat.eval.utils.eval_trace_ctx import EvalTraceContext

logger = logging.getLogger(__name__)


class PhoenixEvaluationIntegration:
    """
    Class to handle Arize Phoenix integration for evaluation metrics.

    This integration attempts best-effort logging of per-item evaluator scores
    to Phoenix when Phoenix tracing is configured in the workflow config.
    """

    def __init__(self, eval_trace_context: "EvalTraceContext"):
        self.available = False
        self.client = None
        self.project_name: str | None = None
        self.eval_trace_context = eval_trace_context
        self.run_name: str | None = None
        # Minimal state to match Weave-level complexity
        # Best-effort mapping from eval item id -> input string for span association
        self._id_to_input: dict[str, str] = {}

        try:
            from phoenix.client import Client as _PhoenixClient  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def _extract_phoenix_server_url(self, endpoint: str) -> str:
        """Convert OTLP traces endpoint to Phoenix server URL if needed.

        Example: http://localhost:6006/v1/traces -> http://localhost:6006
        """
        if not endpoint:
            return endpoint
        # strip trailing '/v1/traces' if present
        suffix = "/v1/traces"
        return endpoint[:-len(suffix)] if endpoint.endswith(suffix) else endpoint

    def _find_phoenix_config(self, config: Any) -> tuple[str | None, str | None]:
        """Find Phoenix tracing config (endpoint, project) from full config object."""
        try:
            cfg = config.model_dump(mode="json")
        except AttributeError:
            try:
                # If already a dict
                cfg = dict(config)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None, None

        tracing = (cfg.get("general", {}) or {}).get("telemetry", {}).get("tracing", {})
        phoenix_cfg = tracing.get("phoenix") or tracing.get("Phoenix")
        if not isinstance(phoenix_cfg, dict):
            return None, None

        endpoint = phoenix_cfg.get("endpoint")
        project = phoenix_cfg.get("project")
        return (endpoint, project)

    def _metric_eval_name(self, metric: str) -> str:
        return f"{self.run_name}:{metric}" if self.run_name else metric

    def initialize_logger(self, _workflow_alias: str, _eval_input: EvalInput, config: Any) -> bool:
        """Initialize Phoenix client if Phoenix tracing is configured."""
        if not self.available:
            return False

        endpoint, project = self._find_phoenix_config(config)
        if not endpoint or not project:
            # Phoenix tracing not configured; skip
            return False

        try:
            from phoenix.client import Client as PhoenixClient
        except ImportError as e:
            logger.warning("Failed to import phoenix client: %s", e)
            self.client = None
            self.project_name = None
            return False

        try:
            server_url = self._extract_phoenix_server_url(endpoint)
            self.client = PhoenixClient(base_url=server_url)
            self.project_name = project
            # capture a friendly run label (workflow alias) for evaluations
            self.run_name = _workflow_alias
            # Build id->input mapping for later span matching (best effort)
            if _eval_input and getattr(_eval_input, "eval_input_items", None):
                for it in _eval_input.eval_input_items:
                    item_id = str(it.id)
                    input_val = str(it.input_obj) if it.input_obj is not None else ""
                    if item_id:
                        self._id_to_input[item_id] = input_val
            logger.debug("Initialized Phoenix client for project '%s' at '%s'", project, server_url)
            return True
        except (ValueError, RuntimeError, TypeError) as e:
            logger.warning("Failed to initialize Phoenix client: %s", e)
            self.client = None
            self.project_name = None
            return False

    def log_prediction(self, _item: EvalInputItem, _output: Any):
        """No-op for Phoenix (kept for interface parity)."""
        return

    async def log_usage_stats(self, item: EvalInputItem, usage_stats_item):  # noqa: ANN001
        """Best-effort usage stats logging as span annotations.

        We intentionally keep this lightweight and skip logging if span resolution fails.
        """
        if not self.client:
            return
        span_id = self._resolve_span_id_for_item(str(item.id))
        if not span_id:
            return
        try:
            self.client.annotations.add_span_annotation(
                span_id=span_id,
                annotation_name=self._metric_eval_name("wf_runtime"),
                annotator_kind="LLM",
                label="seconds",
                score=float(getattr(usage_stats_item, "runtime", 0.0) or 0.0),
                explanation=None,
            )
            self.client.annotations.add_span_annotation(
                span_id=span_id,
                annotation_name=self._metric_eval_name("wf_tokens"),
                annotator_kind="LLM",
                label="count",
                score=float(getattr(usage_stats_item, "total_tokens", 0) or 0),
                explanation=None,
            )
        except (ValueError, TypeError, RuntimeError, httpx.HTTPError):
            logger.debug("Phoenix usage stats logging failed")

    async def alog_score(self, eval_output: EvalOutput, evaluator_name: str):
        """Log per-item evaluator scores to Phoenix as span annotations."""
        if not self.client:
            return

        if not eval_output.eval_output_items:
            return

        for eval_output_item in eval_output.eval_output_items:
            span_id = self._resolve_span_id_for_item(str(eval_output_item.id))
            if not span_id:
                continue
            score_val = eval_output_item.score
            try:
                score_val = float(score_val)
            except (TypeError, ValueError):
                # Skip non-numeric scores
                continue
            try:
                self.client.annotations.add_span_annotation(
                    span_id=span_id,
                    annotation_name=self._metric_eval_name(evaluator_name),
                    annotator_kind="LLM",
                    label="score",
                    score=score_val,
                    explanation=None,
                )
            except (ValueError, TypeError, RuntimeError, httpx.HTTPError):
                logger.debug("Phoenix per-item score logging failed")

    async def afinish_loggers(self):
        # No-op for Phoenix integration
        return

    def log_summary(self,
                    _usage_stats: UsageStats,
                    _evaluation_results: list[tuple[str, EvalOutput]],
                    _profiler_results: ProfilerResults):
        """No-op: Phoenix Client annotations are span-based; skip summary logging."""
        return

    def _resolve_span_id_for_item(self, item_id: str) -> str | None:
        """Resolve a Phoenix span id for an evaluation item.

        Keep this best-effort and lightweight: fetch a small recent window of spans
        and match on `input.value`. If unavailable or not found, skip.
        """
        if not self.client or not self.project_name or not item_id:
            return None
        input_value = self._id_to_input.get(item_id)
        if input_value is None:
            return None
        try:
            # Search a narrow window to reduce overhead
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=4)
            spans = self.client.spans.get_spans(
                project_identifier=self.project_name,
                limit=2000,
                start_time=start_time,
                end_time=end_time,
            )
            for span in spans or []:
                sid = span.get("id")
                attrs = span.get("attributes") or {}
                val = attrs.get("input.value")
                if sid and val is not None and str(val) == str(input_value):
                    return str(sid)
        except (ValueError, TypeError, RuntimeError, httpx.HTTPError):
            return None
        return None
