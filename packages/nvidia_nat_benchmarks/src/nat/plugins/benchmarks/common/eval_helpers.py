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
"""Shared evaluation helpers for benchmark evaluators.

Provides a common evaluator loop pattern used across all benchmark evaluators
to reduce boilerplate and ensure consistent error handling.
"""

import logging
from collections.abc import Callable

from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutput
from nat.data_models.evaluator import EvalOutputItem

logger = logging.getLogger(__name__)


def run_evaluator_loop(
    eval_input: EvalInput,
    evaluate_item_fn: Callable[[EvalInputItem], EvalOutputItem],
    benchmark_name: str,
) -> EvalOutput:
    """Run the standard evaluator loop: score each item, handle errors, compute average.

    This is the shared pattern used by all benchmark evaluators. It:
    1. Iterates over all items in the eval input
    2. Calls evaluate_item_fn for each item
    3. Catches and logs exceptions, assigning score=0.0 for failed items
    4. Computes the average score across all items
    5. Logs a summary

    Args:
        eval_input: The evaluation input containing all items to evaluate.
        evaluate_item_fn: A callable that takes an EvalInputItem and returns an EvalOutputItem.
        benchmark_name: Name of the benchmark (for logging).

    Returns:
        EvalOutput with average score and per-item results.
    """
    eval_output_items: list[EvalOutputItem] = []

    for item in eval_input.eval_input_items:
        try:
            output_item = evaluate_item_fn(item)
        except Exception as e:
            logger.exception("Error evaluating %s item %s: %s", benchmark_name, item.id, e)
            output_item = EvalOutputItem(
                id=item.id,
                score=0.0,
                reasoning={"error": str(e)},
            )
        eval_output_items.append(output_item)

    scores = [item.score for item in eval_output_items if isinstance(item.score, (int, float))]
    average_score = sum(scores) / len(scores) if scores else 0.0

    logger.info(
        "%s evaluation complete: avg_score=%.3f across %d items",
        benchmark_name,
        average_score,
        len(scores),
    )

    return EvalOutput(average_score=average_score, eval_output_items=eval_output_items)
