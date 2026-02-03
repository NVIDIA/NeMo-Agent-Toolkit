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
Action Completion (AC) Evaluator for Agent Leaderboard benchmarks.

This evaluator assesses whether the agent's final response addresses all user goals.
It checks if the agent completed all required actions to satisfy the user's request.
"""

import logging
from typing import Any

from pydantic import Field

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.component_ref import LLMRef
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem

logger = logging.getLogger(__name__)


class ActionCompletionEvaluatorConfig(EvaluatorBaseConfig, name="action_completion_evaluator"):
    """Configuration for Action Completion evaluator."""

    llm_name: LLMRef | None = Field(
        default=None,
        description="Optional LLM to use for semantic goal completion checking",
    )
    strict_mode: bool = Field(
        default=False,
        description="If True, requires all goals to be explicitly mentioned. If False, uses semantic matching.",
    )


@register_evaluator(config_type=ActionCompletionEvaluatorConfig)
async def action_completion_evaluator_function(config: ActionCompletionEvaluatorConfig, builder: EvalBuilder):
    """
    Register the Action Completion (AC) evaluator.

    The AC metric evaluates whether the agent's final response addresses all user goals.

    Score calculation:
    - AC score = (goals_addressed / total_goals)
    - A score of 1.0 means all goals were addressed
    - A score of 0.0 means no goals were addressed
    """
    # Get LLM if specified for semantic evaluation
    llm = None
    if config.llm_name:
        llm = await builder.get_llm(config.llm_name)

    def extract_final_response(trajectory: list[dict[str, Any] | Any]) -> str:
        """
        Extract the final response from agent trajectory.

        Args:
            trajectory: List of trajectory steps

        Returns:
            Final response text
        """
        # Look for the last agent response
        for step in reversed(trajectory):
            # Convert to dict if it's a Pydantic model
            if hasattr(step, "model_dump"):
                try:
                    step = step.model_dump()
                except Exception:
                    continue
            elif not isinstance(step, dict):
                continue

            # Check for various response formats
            if step.get("event_type") == "AGENT_RESPONSE":
                return step.get("response", "")
            elif "output" in step:
                return step.get("output", "")
            elif "observation" in step:
                # Last observation might be the final response
                return step.get("observation", "")

        return ""

    def check_goal_completion_simple(response: str, goal: str) -> bool:
        """
        Simple keyword-based goal completion check.

        Args:
            response: Agent's final response
            goal: User goal to check

        Returns:
            True if goal appears to be addressed in response
        """
        response_lower = response.lower()
        goal_lower = goal.lower()

        # Extract key action words from goal
        action_words = [
            "check",
            "transfer",
            "pay",
            "send",
            "block",
            "unblock",
            "update",
            "change",
            "view",
            "get",
            "set",
            "cancel",
            "increase",
            "decrease",
            "report",
            "dispute"
        ]

        # Check if any key words from goal appear in response
        goal_keywords = [word for word in action_words if word in goal_lower]

        if not goal_keywords:
            # If no action words found, do simple substring check
            return any(word in response_lower for word in goal_lower.split() if len(word) > 3)

        # Check if action words from goal are in response
        return any(keyword in response_lower for keyword in goal_keywords)

    async def check_goal_completion_llm(response: str, goal: str) -> bool:
        """
        LLM-based semantic goal completion check.

        Args:
            response: Agent's final response
            goal: User goal to check

        Returns:
            True if goal is addressed in response according to LLM
        """
        if not llm:
            return check_goal_completion_simple(response, goal)

        prompt = f"""Given the following user goal and agent response, determine if the goal was addressed.

User Goal: {goal}

Agent Response: {response}

Was the user goal addressed in the agent's response? Respond with only "YES" or "NO".
"""

        try:
            result = await llm.ainvoke(prompt)
            result_text = str(result).strip().upper()
            return "YES" in result_text
        except Exception:
            logger.exception("LLM evaluation failed, falling back to simple check")
            return check_goal_completion_simple(response, goal)

    async def evaluate_single_item(item: EvalInputItem) -> EvalOutputItem:
        """
        Evaluate Action Completion for a single item.

        Args:
            item: Evaluation input item with trajectory and user goals

        Returns:
            EvalOutputItem with AC score and reasoning
        """
        try:
            # Extract final response from trajectory
            final_response = extract_final_response(item.trajectory)

            # Get user goals from full dataset entry
            full_entry = item.full_dataset_entry if isinstance(item.full_dataset_entry, dict) else {}
            user_goals = full_entry.get("user_goals", [])

            if not user_goals:
                logger.warning("No user_goals found for item %s, defaulting to score 1.0", item.id)
                return EvalOutputItem(
                    id=item.id,
                    score=1.0,
                    reasoning={
                        "error": "No user goals provided", "goals_addressed": 0, "total_goals": 0
                    },
                )

            # Check each goal
            goals_addressed = 0
            goal_results = []

            for goal in user_goals:
                if config.strict_mode or not llm:
                    is_addressed = check_goal_completion_simple(final_response, goal)
                else:
                    is_addressed = await check_goal_completion_llm(final_response, goal)

                if is_addressed:
                    goals_addressed += 1

                goal_results.append({"goal": goal, "addressed": is_addressed})

            # Calculate AC score
            ac_score = goals_addressed / len(user_goals) if user_goals else 0.0

            reasoning = {
                "goals_addressed": goals_addressed,
                "total_goals": len(user_goals),
                "completion_rate": ac_score,
                "goal_details": goal_results,
                "final_response_preview": final_response[:200] + "..." if len(final_response) > 200 else final_response,
            }

            logger.debug("AC evaluation for item %s: score=%.3f (%d/%d goals)",
                         item.id,
                         ac_score,
                         goals_addressed,
                         len(user_goals))
            return EvalOutputItem(id=item.id, score=ac_score, reasoning=reasoning)

        except Exception as e:
            logger.exception("Error evaluating AC for item %s: %s", item.id, e)
            return EvalOutputItem(
                id=item.id,
                score=0.0,
                reasoning={
                    "error": str(e), "goals_addressed": 0, "total_goals": 0
                },
            )

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        """
        Evaluate Action Completion for all items in the dataset.

        Args:
            eval_input: Evaluation input containing all items

        Returns:
            EvalOutput with average AC score and per-item results
        """
        eval_output_items = []

        for item in eval_input.eval_input_items:
            output_item = await evaluate_single_item(item)
            eval_output_items.append(output_item)

        # Calculate average score
        scores = [item.score for item in eval_output_items if isinstance(item.score, int | float)]
        average_score = sum(scores) / len(scores) if scores else 0.0

        logger.info("AC Evaluation complete: average_score=%.3f across %d items", average_score, len(scores))

        return EvalOutput(average_score=average_score, eval_output_items=eval_output_items)

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluate_fn,
        description="Action Completion (AC) evaluator for agent leaderboard benchmarks",
    )
