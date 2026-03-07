# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BYOB benchmark evaluator.

Calls bench.scorer_fn(ScorerInput(...)) directly in-process.
model_call_fn=None is safe for all built-in scorers (exact_match, contains,
f1_token, bleu, rouge, regex_match).
"""

import json
import logging

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalInput, EvalInputItem, EvalOutput, EvalOutputItem

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
            id=item.id, score=0.0,
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
            id=item.id, score=0.0,
            reasoning={"error": f"Scorer failed: {e}"},
        )

    # Extract the primary score
    primary_score = scores.get(score_field, 0.0)
    if isinstance(primary_score, bool):
        primary_score = 1.0 if primary_score else 0.0

    return EvalOutputItem(id=item.id, score=float(primary_score), reasoning=scores)


@register_evaluator(config_type=BYOBEvaluatorConfig)
async def byob_evaluator_function(config: BYOBEvaluatorConfig, builder: EvalBuilder):
    from nemo_evaluator.contrib.byob.eval_logic import import_benchmark

    bench = import_benchmark(config.benchmark_module, config.benchmark_name)
    logger.info("BYOB evaluator loaded benchmark '%s' with scorer '%s'",
                bench.name, getattr(bench.scorer_fn, '__name__', 'unknown'))

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        eval_output_items = []

        for item in eval_input.eval_input_items:
            try:
                output_item = _evaluate_single(
                    item, bench.scorer_fn, bench.target_field,
                    config.score_field, bench.extra_config,
                )
            except Exception as e:
                logger.exception("Error evaluating BYOB item %s: %s", item.id, e)
                output_item = EvalOutputItem(
                    id=item.id, score=0.0, reasoning={"error": str(e)},
                )
            eval_output_items.append(output_item)

        scores = [i.score for i in eval_output_items if isinstance(i.score, (int, float))]
        average_score = sum(scores) / len(scores) if scores else 0.0

        logger.info(
            "BYOB evaluation complete: avg_%s=%.3f (%d items)",
            config.score_field, average_score, len(scores),
        )

        return EvalOutput(average_score=average_score, eval_output_items=eval_output_items)

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluate_fn,
        description=f"BYOB evaluator for '{config.benchmark_name}'",
    )
