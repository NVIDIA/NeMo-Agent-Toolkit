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
"""ToolTalk benchmark evaluator.

Uses ToolTalk's built-in evaluation logic to compare predicted API calls against
ground truth, producing metrics: recall, action_precision, bad_action_rate, success.
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
from .config import ToolTalkEvaluatorConfig

logger = logging.getLogger(__name__)


def _evaluate_single(item: EvalInputItem, database_dir: str) -> EvalOutputItem:
    """Evaluate a single ToolTalk conversation using ToolTalk's built-in metrics."""
    from tooltalk.evaluation.tool_executor import ToolExecutor

    if not item.output_obj:
        return EvalOutputItem(
            id=item.id,
            score=0.0,
            reasoning={"error": "No workflow output (output_obj is empty or None)"},
        )

    try:
        conversation_with_predictions = json.loads(item.output_obj)
    except (json.JSONDecodeError, TypeError) as e:
        return EvalOutputItem(
            id=item.id,
            score=0.0,
            reasoning={"error": f"Invalid JSON in output_obj: {e}"},
        )

    tool_executor = ToolExecutor(init_database_dir=database_dir)
    result = tool_executor.evaluate_predictions(conversation_with_predictions)
    metrics = result.get("metrics", {})

    success = float(metrics.get("success", 0.0))
    reasoning = {
        "predictions": metrics.get("predictions", 0),
        "ground_truths": metrics.get("ground_truths", 0),
        "matches": metrics.get("matches", 0),
        "recall": metrics.get("recall", 0.0),
        "actions": metrics.get("actions", 0),
        "valid_actions": metrics.get("valid_actions", 0),
        "bad_actions": metrics.get("bad_actions", 0),
        "action_precision": metrics.get("action_precision", 0.0),
        "bad_action_rate": metrics.get("bad_action_rate", 0.0),
        "success": success,
        "soft_success": metrics.get("soft_success", 0.0),
    }

    return EvalOutputItem(id=item.id, score=success, reasoning=reasoning)


@register_evaluator(config_type=ToolTalkEvaluatorConfig)
async def tooltalk_evaluator_function(config: ToolTalkEvaluatorConfig, builder: EvalBuilder):
    """Register the ToolTalk benchmark evaluator."""

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        """Evaluate all items using ToolTalk's built-in metrics."""
        return run_evaluator_loop(
            eval_input,
            evaluate_item_fn=partial(_evaluate_single, database_dir=config.database_dir),
            benchmark_name="ToolTalk",
        )

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluate_fn,
        description="ToolTalk benchmark evaluator (recall, action_precision, success)",
    )
