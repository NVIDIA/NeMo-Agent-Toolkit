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
"""Lightweight ATIF-only evaluator harness.

This harness is intentionally narrow in scope:
- it evaluates ATIF-native evaluators only (`evaluate_atif_fn`)
- it runs evaluators concurrently
- it returns per-evaluator `EvalOutput` objects

Example:
    ```python
    harness = EvaluationHarness()
    results = await harness.evaluate(
        evaluators={"trajectory": trajectory_evaluator},
        atif_samples=atif_samples,
    )
    ```
"""

from __future__ import annotations

import asyncio
import logging

from nat.plugins.eval.data_models.evaluator_io import EvalOutput
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSampleList
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvaluator

logger = logging.getLogger(__name__)


class EvaluationHarness:
    """Run ATIF-native evaluators against a shared sample list."""

    def __init__(self, logger_instance: logging.Logger | None = None):
        self._logger = logger_instance or logger

    async def _evaluate_single(self, evaluator_name: str, evaluator: AtifEvaluator,
                               atif_samples: AtifEvalSampleList) -> tuple[str, EvalOutput] | None:
        """Evaluate one evaluator using the ATIF lane.

        Returns:
            A tuple of evaluator name and result on success, otherwise ``None``.
        """
        if not callable(evaluator.evaluate_atif_fn):
            self._logger.warning("Skipping evaluator %s: missing callable evaluate_atif_fn", evaluator_name)
            return None

        try:
            eval_output = await evaluator.evaluate_atif_fn(atif_samples)
            return evaluator_name, eval_output
        except Exception:
            # Best-effort policy: log per-evaluator failure and continue.
            self._logger.exception("An error occurred while running evaluator %s", evaluator_name)
            return None

    async def evaluate(self, evaluators: dict[str, AtifEvaluator],
                       atif_samples: AtifEvalSampleList) -> dict[str, EvalOutput]:
        """Evaluate ATIF-native evaluators concurrently.

        Args:
            evaluators: Evaluators keyed by evaluator name.
            atif_samples: Pre-built ATIF samples shared by all evaluators.

        Returns:
            A mapping of evaluator name to `EvalOutput` for successful evaluators.
        """
        tasks = [
            self._evaluate_single(evaluator_name=name, evaluator=evaluator, atif_samples=atif_samples)
            for name, evaluator in evaluators.items() if evaluator
        ]
        if not tasks:
            return {}

        results = await asyncio.gather(*tasks)
        return {name: output for result in results if result is not None for name, output in [result]}
