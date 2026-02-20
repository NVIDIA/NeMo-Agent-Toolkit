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
from dataclasses import dataclass
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
    """Full result of a single evaluation run."""
    metric_scores: dict[str, float]  # evaluator_name -> average score
    items: list[EvalResultItem]  # per-item breakdown


class EvalCallback(Protocol):

    def on_dataset_loaded(self, *, dataset_name: str, items: list[EvalInputItem]) -> None:
        ...

    def on_eval_complete(self, result: EvalResult) -> None:
        ...


class EvalCallbackManager:

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
            try:
                cb.on_dataset_loaded(dataset_name=dataset_name, items=items)
            except Exception:
                logger.exception("EvalCallback %s.on_dataset_loaded failed", type(cb).__name__)

    def on_eval_complete(self, result: EvalResult) -> None:
        for cb in self._callbacks:
            try:
                cb.on_eval_complete(result)
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


def get_tracing_configs(config: Any) -> dict[str, Any]:
    """Extract tracing configs from a loaded NAT config object."""
    return getattr(getattr(getattr(config, 'general', None), 'telemetry', None), 'tracing', None) or {}
