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
"""
Tool Selection Quality (TSQ) Evaluator for Agent Leaderboard benchmarks.

This evaluator assesses:
1. Tool selection accuracy - Did the agent select the correct tools?
2. Parameter usage correctness - Were the tool parameters used correctly?
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import Field

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_evaluator
from nat.data_models.component_ref import LLMRef
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem

logger = logging.getLogger(__name__)


class TSQEvaluatorConfig(EvaluatorBaseConfig, name="tsq_evaluator"):
    """Configuration for Tool Selection Quality evaluator."""

    llm_name: LLMRef | None = Field(
        default=None,
        description="Optional LLM to use for semantic parameter comparison",
    )
    strict_mode: bool = Field(
        default=False,
        description="If True, requires exact tool and parameter matches. If False, allows semantic similarity.",
    )
    tool_weight: float = Field(
        default=1.0,
        description="Weight for tool selection accuracy in final score (0-1)",
    )
    parameter_weight: float = Field(
        default=0.0,
        description="Weight for parameter correctness in final score (0-1)",
    )


@register_evaluator(config_type=TSQEvaluatorConfig)
async def tsq_evaluator_function(config: TSQEvaluatorConfig, builder: EvalBuilder) -> AsyncIterator[EvaluatorInfo]:
    """
    Register the Tool Selection Quality (TSQ) evaluator.

    The TSQ metric evaluates:
    1. Tool Selection Accuracy: % of correctly selected tools
    2. Parameter Usage Correctness: % of correctly used parameters

    Final TSQ score = (tool_weight * tool_accuracy) + (parameter_weight * param_accuracy)
    """
    # Unused: builder is available if needed for future enhancements
    del builder

    def extract_tool_calls_from_trajectory(trajectory: list[dict[str, Any] | Any]) -> list[dict[str, Any]]:
        """
        Extract tool calls from agent trajectory.

        Handles multiple data formats:
        1. Flat structure with event_type at top level (legacy)
        2. Nested structure with payload containing event_type (profiler format)
        3. LangChain action/action_input format
        4. IntermediateStep Pydantic objects

        Args:
            trajectory: List of trajectory steps (can be dicts or IntermediateStep objects)

        Returns:
            List of extracted tool calls with format [{"tool": "name", "parameters": {...}}]
        """
        tool_calls = []
        for step in trajectory:
            # Convert to dict if it's an IntermediateStep or similar Pydantic model
            if hasattr(step, "model_dump"):
                try:
                    step = step.model_dump()
                except (TypeError, ValueError) as exc:
                    logger.warning("Failed to convert step to dict: %s", exc)
                    continue
            elif not isinstance(step, dict):
                logger.warning("Skipping non-dict, non-Pydantic step: %s", type(step))
                continue

            # Try multiple extraction strategies
            tool_call = None

            # Strategy 1: Nested payload structure (profiler format)
            # Structure: {"payload": {"event_type": "TOOL_START", "name": "tool_name", "data": {...}}}
            payload = step.get("payload", {})
            if isinstance(payload, dict) and payload.get("event_type") == "TOOL_START":
                tool_name = payload.get("name", "")
                # Extract parameters from data.input or data.input_params
                data = payload.get("data", {})
                if isinstance(data, dict):
                    params = data.get("input_params", data.get("input", {}))
                    if isinstance(params, dict):
                        # Handle nested input_params structure
                        params = params.get("input_params", params)
                else:
                    params = {}
                tool_call = {"tool": tool_name, "parameters": params if isinstance(params, dict) else {}}

            # Strategy 2: Flat structure with event_type at top level (legacy format)
            elif step.get("event_type") == "TOOL_START":
                tool_call = {
                    "tool": step.get("tool_name", step.get("name", "")),
                    "parameters": step.get("tool_input", step.get("input", {})),
                }

            # Strategy 3: LangChain action format
            elif "action" in step and "action_input" in step:
                tool_call = {
                    "tool": step.get("action", ""),
                    "parameters": step.get("action_input", {}),
                }

            if tool_call and tool_call.get("tool"):
                tool_calls.append(tool_call)

        logger.debug("Extracted %d tool calls from trajectory", len(tool_calls))
        return tool_calls

    def normalize_tool_name(tool_name: str) -> str:
        """
        Normalize tool names for comparison.

        Handles:
        - Case normalization (lowercase)
        - Underscore and dash removal
        - Module prefix stripping (e.g., 'banking_tools.report_lost_stolen_card' -> 'reportloststolencard')

        Args:
            tool_name: Raw tool name from trajectory or expected list

        Returns:
            Normalized tool name for comparison
        """
        if not tool_name:
            return ""

        # Strip module prefix (e.g., "banking_tools.report_lost_stolen_card" -> "report_lost_stolen_card")
        if FunctionGroup.SEPARATOR in tool_name:
            _, tool_name = FunctionGroup.decompose(tool_name)

        return tool_name.lower().strip().replace("_", "").replace("-", "")

    def calculate_tool_accuracy(actual: list[dict], expected: list[dict]) -> float:
        """Calculate tool selection accuracy."""
        if not expected:
            return 1.0 if not actual else 0.0

        actual_tools = {normalize_tool_name(tc["tool"]) for tc in actual}
        expected_tools = {normalize_tool_name(tc["tool"]) for tc in expected}

        if not expected_tools:
            return 1.0

        # Calculate precision and recall
        correct = len(actual_tools.intersection(expected_tools))
        precision = correct / len(actual_tools) if actual_tools else 0.0
        recall = correct / len(expected_tools) if expected_tools else 0.0

        # F1 score
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_parameter_accuracy(actual: list[dict], expected: list[dict]) -> float:
        """Calculate parameter usage accuracy."""
        if not expected:
            return 1.0

        # Group by tool name
        actual_by_tool = {normalize_tool_name(tc["tool"]): tc["parameters"] for tc in actual}
        expected_by_tool = {normalize_tool_name(tc["tool"]): tc["parameters"] for tc in expected}

        if not expected_by_tool:
            return 1.0

        total_params = 0
        correct_params = 0

        for tool, expected_params in expected_by_tool.items():
            if tool not in actual_by_tool:
                total_params += len(expected_params)
                continue

            actual_params = actual_by_tool[tool]

            for param_name, expected_value in expected_params.items():
                total_params += 1
                actual_value = actual_params.get(param_name)

                # Exact match or type match
                if actual_value == expected_value:
                    correct_params += 1
                elif isinstance(expected_value, type(actual_value)) or isinstance(actual_value, type(expected_value)):
                    # For non-strict mode, give partial credit for type match
                    if not config.strict_mode:
                        correct_params += 0.5

        return correct_params / total_params if total_params > 0 else 1.0

    async def evaluate_single_item(item: EvalInputItem) -> EvalOutputItem:
        """
        Evaluate Tool Selection Quality for a single item.

        Args:
            item: Evaluation input item with trajectory and expected tool calls

        Returns:
            EvalOutputItem with TSQ score and reasoning
        """
        try:
            # Debug: Log what we receive
            logger.info("Evaluating item %s", item.id)
            logger.debug("  Trajectory type: %s, length: %d",
                         type(item.trajectory),
                         len(item.trajectory) if item.trajectory else 0)

            # Extract actual tool calls from trajectory
            actual_tool_calls = extract_tool_calls_from_trajectory(item.trajectory)

            logger.info("  Extracted %d tool calls from trajectory", len(actual_tool_calls))

            # In decision-only mode, also check for tool intents in metadata
            # (This would be populated by the tool intent buffer)
            if hasattr(item, "metadata") and isinstance(item.metadata, dict):
                tool_intents = item.metadata.get("tool_intents", [])
                if tool_intents:
                    logger.info("Found %d tool intents in metadata for item %s", len(tool_intents), item.id)
                    # Merge intents with trajectory-extracted calls
                    actual_tool_calls.extend(tool_intents)

            # FALLBACK: Access global intent registry
            # This is a workaround for decision-only mode where intents are stored globally
            if len(actual_tool_calls) == 0:
                try:
                    from react_benchmark_agent.tool_intent_stubs import clear_global_intents
                    from react_benchmark_agent.tool_intent_stubs import get_global_intents

                    # Try with scenario ID first, then fallback to "current"
                    scenario_intents = get_global_intents(str(item.id))
                    if not scenario_intents:
                        scenario_intents = get_global_intents("current")

                    if scenario_intents:
                        logger.info("Retrieved %d intents from global registry for item %s",
                                    len(scenario_intents),
                                    item.id)
                        actual_tool_calls = scenario_intents
                        # Clear for next scenario
                        clear_global_intents("current")
                        clear_global_intents(str(item.id))
                    else:
                        logger.warning("No intents found in global registry for item %s", item.id)

                except (ImportError, AttributeError, KeyError) as exc:
                    logger.warning("Failed to retrieve intents from global registry: %s", exc)

            # Get expected tool calls from full dataset entry
            full_entry = item.full_dataset_entry if isinstance(item.full_dataset_entry, dict) else {}
            expected_tool_calls = full_entry.get("expected_tool_calls", [])

            # Calculate component scores
            tool_accuracy = calculate_tool_accuracy(actual_tool_calls, expected_tool_calls)
            param_accuracy = calculate_parameter_accuracy(actual_tool_calls, expected_tool_calls)

            # Calculate weighted TSQ score
            tsq_score = (config.tool_weight * tool_accuracy) + (config.parameter_weight * param_accuracy)

            reasoning = {
                "tool_selection_accuracy": tool_accuracy,
                "parameter_usage_accuracy": param_accuracy,
                "actual_tool_calls": len(actual_tool_calls),
                "expected_tool_calls": len(expected_tool_calls),
                "details": {
                    "actual_tools": [tc["tool"] for tc in actual_tool_calls],
                    "expected_tools": [tc["tool"] for tc in expected_tool_calls],
                },
            }

            logger.debug("TSQ evaluation for item %s: score=%.3f", item.id, tsq_score)
            return EvalOutputItem(id=item.id, score=tsq_score, reasoning=reasoning)

        except Exception:
            logger.exception("Error evaluating TSQ for item %s", item.id)
            return EvalOutputItem(
                id=item.id,
                score=0.0,
                reasoning={
                    "error": "Evaluation failed", "tool_selection_accuracy": 0.0, "parameter_usage_accuracy": 0.0
                },
            )

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        """
        Evaluate Tool Selection Quality for all items in the dataset.

        Args:
            eval_input: Evaluation input containing all items

        Returns:
            EvalOutput with average TSQ score and per-item results
        """
        eval_output_items = []

        for item in eval_input.eval_input_items:
            output_item = await evaluate_single_item(item)
            eval_output_items.append(output_item)

        # Calculate average score
        scores = [item.score for item in eval_output_items if isinstance(item.score, int | float)]
        average_score = sum(scores) / len(scores) if scores else 0.0

        logger.info("TSQ Evaluation complete: average_score=%.3f across %d items", average_score, len(scores))

        return EvalOutput(average_score=average_score, eval_output_items=eval_output_items)

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluate_fn,
        description="Tool Selection Quality (TSQ) evaluator for agent leaderboard benchmarks",
    )
