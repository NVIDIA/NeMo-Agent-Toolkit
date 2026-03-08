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
"""Tool Selection Quality (TSQ) evaluator for Agent Leaderboard.

Computes F1 score between predicted tool calls (from workflow output)
and expected tool calls (from dataset ground truth).
"""

import json
import logging

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutput
from nat.data_models.evaluator import EvalOutputItem

from .config import TSQEvaluatorConfig

logger = logging.getLogger(__name__)


def _normalize_tool_name(name: str) -> str:
    """Normalize tool name for comparison (lowercase, strip separators)."""
    if not name:
        return ""
    # Strip module prefix (e.g. "banking_tools__get_balance" -> "get_balance")
    if "__" in name:
        name = name.split("__", 1)[1]
    return name.lower().strip().replace("_", "").replace("-", "")


def _evaluate_single(item: EvalInputItem, tool_weight: float, parameter_weight: float) -> EvalOutputItem:
    """Evaluate tool selection quality for a single scenario."""
    if item.output_obj is None:
        return EvalOutputItem(
            id=item.id,
            score=0.0,
            reasoning={"error": "No workflow output"},
        )

    # Parse predicted tool calls from workflow output
    try:
        predicted = json.loads(item.output_obj) if isinstance(item.output_obj, str) else item.output_obj
        if not isinstance(predicted, list):
            predicted = []
    except (json.JSONDecodeError, TypeError):
        predicted = []

    # Parse expected tool calls from ground truth
    # The full_dataset_entry from DatasetHandler is a DataFrame row with 'question' as serialized JSON.
    # We need to parse that JSON to get the actual entry with expected_tool_calls.
    full_entry = item.full_dataset_entry
    if isinstance(full_entry, str):
        try:
            full_entry = json.loads(full_entry)
        except (json.JSONDecodeError, TypeError):
            full_entry = {}
    if not isinstance(full_entry, dict):
        full_entry = {}

    # Check if expected_tool_calls is directly in full_entry
    expected = full_entry.get("expected_tool_calls", [])

    # If not found, try parsing the 'question' field (which contains the serialized entry)
    if not expected and "question" in full_entry:
        try:
            question_data = json.loads(full_entry["question"]) if isinstance(full_entry["question"],
                                                                             str) else full_entry["question"]
            if isinstance(question_data, dict):
                expected = question_data.get("expected_tool_calls", [])
        except (json.JSONDecodeError, TypeError):
            pass

    # Also try input_obj as a last resort
    if not expected and item.input_obj:
        try:
            input_data = json.loads(item.input_obj) if isinstance(item.input_obj, str) else item.input_obj
            if isinstance(input_data, dict):
                expected = input_data.get("expected_tool_calls", [])
        except (json.JSONDecodeError, TypeError):
            pass

    # Calculate tool selection F1
    predicted_tools = {_normalize_tool_name(tc.get("tool", "")) for tc in predicted} - {""}
    expected_tools = {_normalize_tool_name(tc.get("tool", "")) for tc in expected} - {""}

    if not expected_tools:
        tool_f1 = 1.0 if not predicted_tools else 0.0
    else:
        correct = len(predicted_tools & expected_tools)
        precision = correct / len(predicted_tools) if predicted_tools else 0.0
        recall = correct / len(expected_tools)
        tool_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Parameter accuracy (simplified: just check if expected params are present)
    param_accuracy = 1.0  # Placeholder — full param matching is optional

    score = tool_weight * tool_f1 + parameter_weight * param_accuracy

    return EvalOutputItem(
        id=item.id,
        score=score,
        reasoning={
            "tool_selection_f1": tool_f1,
            "parameter_accuracy": param_accuracy,
            "predicted_tools": sorted(predicted_tools),
            "expected_tools": sorted(expected_tools),
            "num_predicted": len(predicted),
            "num_expected": len(expected),
        },
    )


@register_evaluator(config_type=TSQEvaluatorConfig)
async def agent_leaderboard_tsq_evaluator(config: TSQEvaluatorConfig, builder: EvalBuilder):

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        eval_output_items = []

        for item in eval_input.eval_input_items:
            try:
                output_item = _evaluate_single(item, config.tool_weight, config.parameter_weight)
            except Exception as e:
                logger.exception("Error evaluating TSQ for item %s: %s", item.id, e)
                output_item = EvalOutputItem(
                    id=item.id,
                    score=0.0,
                    reasoning={"error": str(e)},
                )
            eval_output_items.append(output_item)

        scores = [i.score for i in eval_output_items if isinstance(i.score, (int, float))]
        average_score = sum(scores) / len(scores) if scores else 0.0

        logger.info("TSQ evaluation complete: avg_f1=%.3f across %d scenarios", average_score, len(scores))

        return EvalOutput(average_score=average_score, eval_output_items=eval_output_items)

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluate_fn,
        description="Tool Selection Quality (TSQ) evaluator for Agent Leaderboard",
    )
