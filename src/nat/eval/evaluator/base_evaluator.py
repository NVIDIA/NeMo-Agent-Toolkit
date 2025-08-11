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
from abc import ABC
from abc import abstractmethod

from tqdm import tqdm

from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.eval.utils.tqdm_position_registry import TqdmPositionRegistry


class BaseEvaluator(ABC):
    """
    Base class for custom evaluators.

    .. warning::
        **Experimental Feature**: The Evaluation API is experimental and may change in future releases.
        Future versions may introduce breaking changes without notice.

    Each custom evaluator must implement the `evaluate_item` method which is used to evaluate a
    single EvalInputItem.
    """

    def __init__(self, max_concurrency: int = 4, tqdm_desc: str = "Evaluating"):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.tqdm_desc = tqdm_desc

    @abstractmethod
    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Each evaluator must implement this for item-level evaluation"""
        pass

    async def evaluate(self, eval_input: EvalInput) -> EvalOutput:
        pbar = None
        try:
            tqdm_position = TqdmPositionRegistry.claim()
            pbar = tqdm(total=len(eval_input.eval_input_items), desc=self.tqdm_desc, position=tqdm_position)

            async def wrapped(item):
                async with self.semaphore:
                    try:
                        output_item = await self.evaluate_item(item)
                        pbar.update(1)
                        return output_item
                    except Exception as e:
                        # If the evaluator fails, return an error item with a score of 0.0
                        pbar.update(1)
                        return EvalOutputItem(id=item.id, score=0.0, reasoning={"error": f"Evaluator error: {str(e)}"})

            output_items = await asyncio.gather(*[wrapped(item) for item in eval_input.eval_input_items])
        finally:
            pbar.close()
            TqdmPositionRegistry.release(tqdm_position)

        # Compute average if possible
        numeric_scores = [item.score for item in output_items if isinstance(item.score, (int, float))]
        avg_score = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None

        return EvalOutput(average_score=avg_score, eval_output_items=output_items)
