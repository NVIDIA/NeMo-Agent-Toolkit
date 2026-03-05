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
    from nat.plugins.eval.evaluator.evaluator_model import EvalInputItem

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    trial_number: int
    parameters: dict[str, Any]
    metric_scores: dict[str, float]
    is_best: bool
    rep_scores: list[list[float]] | None = None
    prompts: dict[str, str] | None = None  # param_name -> prompt text (for prompt GA trials)
    prompt_formats: dict[str, str] | None = None  # param_name -> template format ("jinja2", "f-string", "mustache")
    eval_result: Any | None = None  # EvalResult from nat.eval.eval_callbacks (kept as Any to avoid circular dep)


class OptimizerCallback(Protocol):

    def pre_create_experiment(self, dataset_items: list[EvalInputItem]) -> None:
        ...

    def on_trial_end(self, result: TrialResult) -> None:
        ...

    def on_study_end(self, *, best_trial: TrialResult, total_trials: int) -> None:
        ...


class OptimizerCallbackManager:

    def __init__(self) -> None:
        self._callbacks: list[OptimizerCallback] = []

    def register(self, callback: OptimizerCallback) -> None:
        self._callbacks.append(callback)

    @property
    def has_callbacks(self) -> bool:
        return bool(self._callbacks)

    def set_prompt_param_names(self, names: list[str]) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "set_prompt_param_names", None)
            if fn:
                try:
                    fn(names)
                except Exception:
                    logger.debug("set_prompt_param_names failed for %s", type(cb).__name__, exc_info=True)

    def pre_create_experiment(self, dataset_items: list[EvalInputItem]) -> None:
        for cb in self._callbacks:
            try:
                cb.pre_create_experiment(dataset_items)
            except Exception:
                logger.exception("OptimizerCallback %s.pre_create_experiment failed", type(cb).__name__)

    def on_trial_end(self, result: TrialResult) -> None:
        for cb in self._callbacks:
            try:
                cb.on_trial_end(result)
            except Exception:
                logger.exception("OptimizerCallback %s.on_trial_end failed", type(cb).__name__)

    def get_trial_project_name(self, trial_number: int) -> str | None:
        """Get a trial-specific OTEL project name from the first callback that supports it."""
        for cb in self._callbacks:
            fn = getattr(cb, "get_trial_project_name", None)
            if fn:
                try:
                    return fn(trial_number)
                except Exception:
                    logger.debug("get_trial_project_name failed for %s", type(cb).__name__, exc_info=True)
        return None

    def on_study_end(self, *, best_trial: TrialResult, total_trials: int) -> None:
        for cb in self._callbacks:
            try:
                cb.on_study_end(best_trial=best_trial, total_trials=total_trials)
            except Exception:
                logger.exception("OptimizerCallback %s.on_study_end failed", type(cb).__name__)
