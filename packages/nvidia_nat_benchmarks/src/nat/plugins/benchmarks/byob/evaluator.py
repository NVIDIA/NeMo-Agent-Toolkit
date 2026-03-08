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
"""BYOB benchmark evaluator.

Calls bench.scorer_fn(ScorerInput(...)) directly in-process.
model_call_fn=None is safe for all built-in scorers (exact_match, contains,
f1_token, bleu, rouge, regex_match).
"""

import json
import logging
from functools import partial

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutput
from nat.data_models.evaluator import EvalOutputItem

from ..common.eval_helpers import run_evaluator_loop
from .config import BYOBEvaluatorConfig

logger = logging.getLogger(__name__)


def _evaluate_single(
    item: EvalInputItem,
    scorer_fn,
    target_field: str,
    score_field: str,
    extra_config: dict,
) -> EvalOutputItem:
    """Evaluate a single item using the BYOB scorer."""
    from nemo_evaluator.contrib.byob.decorators import ScorerInput

    if item.output_obj is None:
        return EvalOutputItem(
            id=item.id,
            score=0.0,
            reasoning={"error": "No workflow output (output_obj is None)"},
        )

    # Reconstruct metadata from the full dataset entry
    if isinstance(item.full_dataset_entry, str):
        try:
            metadata = json.loads(item.full_dataset_entry)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    elif isinstance(item.full_dataset_entry, dict):
        metadata = item.full_dataset_entry
    else:
        metadata = {}

    # Get target value
    target = item.expected_output_obj
    if isinstance(target, str):
        try:
            target = json.loads(target)
        except (json.JSONDecodeError, ValueError):
            pass  # Keep as string

    scorer_input = ScorerInput(
        response=str(item.output_obj),
        target=target,
        metadata=metadata,
        model_call_fn=None,
        config=extra_config,
    )

    try:
        scores = scorer_fn(scorer_input)
    except Exception as e:
        return EvalOutputItem(
            id=item.id,
            score=0.0,
            reasoning={"error": f"Scorer failed: {e}"},
        )

    # Extract the primary score
    primary_score = scores.get(score_field, 0.0)
    if isinstance(primary_score, bool):
        primary_score = 1.0 if primary_score else 0.0

    return EvalOutputItem(id=item.id, score=float(primary_score), reasoning=scores)


@register_evaluator(config_type=BYOBEvaluatorConfig)
async def byob_evaluator_function(config: BYOBEvaluatorConfig, builder: EvalBuilder):
    """Register the BYOB benchmark evaluator."""
    from nemo_evaluator.contrib.byob.eval_logic import import_benchmark

    bench = import_benchmark(config.benchmark_module, config.benchmark_name)
    logger.info(
        "BYOB evaluator loaded benchmark '%s' with scorer '%s'",
        bench.name,
        getattr(bench.scorer_fn, '__name__', 'unknown'),
    )

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        """Evaluate all items using the BYOB scorer."""
        return run_evaluator_loop(
            eval_input,
            evaluate_item_fn=partial(
                _evaluate_single,
                scorer_fn=bench.scorer_fn,
                target_field=bench.target_field,
                score_field=config.score_field,
                extra_config=bench.extra_config,
            ),
            benchmark_name=f"BYOB ({config.benchmark_name})",
        )

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluate_fn,
        description=f"BYOB evaluator for '{config.benchmark_name}'",
    )
