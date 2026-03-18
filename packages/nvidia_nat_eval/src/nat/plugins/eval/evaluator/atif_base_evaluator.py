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
"""Reusable ATIF-native evaluator base with concurrent orchestration."""

from __future__ import annotations

import asyncio
from abc import ABC
from abc import abstractmethod

from nat.data_models.evaluator import EvalOutput
from nat.data_models.evaluator import EvalOutputItem
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSampleList


class AtifBaseEvaluator(ABC):
    """Base class for ATIF-native custom evaluators.

    Implementers provide item-level scoring via `evaluate_atif_item`.
    This base handles bounded concurrency, gathers all items asynchronously,
    and computes `EvalOutput.average_score` from numeric per-item scores.
    """

    def __init__(self, max_concurrency: int = 4):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)

    @abstractmethod
    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        """Evaluate one ATIF sample and return a single output item."""

    async def evaluate_atif_fn(self, atif_samples: AtifEvalSampleList) -> EvalOutput:
        """Evaluate ATIF samples concurrently with bounded concurrency."""

        async def wrapped(sample: AtifEvalSample) -> EvalOutputItem:
            async with self.semaphore:
                try:
                    return await self.evaluate_atif_item(sample)
                except Exception as e:
                    return EvalOutputItem(id=sample.item_id, score=0.0, reasoning={"error": f"Evaluator error: {e}"})

        output_items = await asyncio.gather(*[wrapped(sample) for sample in atif_samples])
        numeric_scores = [item.score for item in output_items if isinstance(item.score, int | float)]
        avg_score = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None
        return EvalOutput(average_score=avg_score, eval_output_items=output_items)
