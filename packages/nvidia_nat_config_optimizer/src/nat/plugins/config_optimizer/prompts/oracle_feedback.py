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
Oracle feedback utilities for prompt optimization.

This module provides functions to extract, format, and inject failure reasoning
from evaluation results into the prompt optimization genetic algorithm. The oracle
feedback system enables context-grounded prompt evolution by learning from specific
evaluation failures.
"""

import json
import statistics
from typing import Any

from pydantic import BaseModel as PydanticBaseModel


def build_oracle_feedback(reasoning_list: list[str], max_chars: int) -> str | None:
    """
    Build truncated feedback string from worst items reasoning.

    Args:
        reasoning_list: List of reasoning strings from worst-performing items.
        max_chars: Maximum characters for the output.

    Returns:
        Formatted feedback string, or None if no reasoning available.
    """
    if not reasoning_list:
        return None

    feedback_parts: list[str] = []
    current_length = 0

    truncated = False
    for i, reasoning in enumerate(reasoning_list, 1):
        entry = f"{i}. {reasoning}\n"
        if current_length + len(entry) > max_chars:
            remaining = max_chars - current_length
            if remaining > 20:  # Only add if meaningful space left
                feedback_parts.append(entry[:remaining - 3] + "...")
            else:
                truncated = True
            break
        feedback_parts.append(entry)
        current_length += len(entry)

    if not feedback_parts:
        return None

    result = "".join(feedback_parts)
    # Add truncation indicator if items were skipped without partial inclusion
    if truncated and not result.endswith("..."):
        # Trim trailing newline if present, add truncation marker
        result = result.rstrip("\n") + "...\n"

    return result


def should_inject_feedback(
    *,
    mode: str,
    scalar_fitness: float,
    fitness_threshold: float,
    adaptive_enabled: bool,
) -> bool:
    """
    Determine if oracle feedback should be injected for this mutation.

    Args:
        mode: Feedback mode ('never', 'always', 'failing_only', 'adaptive').
        scalar_fitness: The individual's normalized fitness score.
        fitness_threshold: Threshold for 'failing_only' mode.
        adaptive_enabled: Whether adaptive feedback has been triggered.

    Returns:
        True if feedback should be injected, False otherwise.
    """
    if mode == "never":
        return False

    if mode == "always":
        return True

    if mode == "failing_only":
        return scalar_fitness < fitness_threshold

    if mode == "adaptive":
        return adaptive_enabled

    return False


def check_adaptive_triggers(
    *,
    best_fitness_history: list[float],
    population_fitness_values: list[float],
    population_prompt_keys: list[tuple[Any, ...]],
    stagnation_generations: int,
    fitness_variance_threshold: float,
    diversity_threshold: float,
) -> dict[str, Any]:
    """
    Check if adaptive feedback should be triggered.

    Args:
        best_fitness_history: History of best fitness values per generation.
        population_fitness_values: Current population's fitness values.
        population_prompt_keys: Hashable keys representing each individual's prompts.
        stagnation_generations: Generations without improvement to trigger.
        fitness_variance_threshold: Variance threshold for collapse detection.
        diversity_threshold: Prompt duplication ratio threshold.

    Returns:
        Dict with 'triggered' bool and 'reason' string if triggered.
    """
    # Check stagnation
    if len(best_fitness_history) >= stagnation_generations:
        recent = best_fitness_history[-stagnation_generations:]
        if (max(recent) - min(recent)) < 0.001:  # Consider stagnant if fitness varies by less than 0.1%
            return {"triggered": True, "reason": "stagnation"}

    # Check fitness variance collapse
    if len(population_fitness_values) > 1:
        variance = statistics.variance(population_fitness_values)
        if variance < fitness_variance_threshold:
            return {"triggered": True, "reason": "fitness_variance_collapse"}

    # Check diversity collapse
    if population_prompt_keys:
        unique_ratio = len(set(population_prompt_keys)) / len(population_prompt_keys)
        if unique_ratio < (1.0 - diversity_threshold):
            return {"triggered": True, "reason": "diversity_collapse"}

    return {"triggered": False, "reason": None}


def _reasoning_to_string(reasoning: Any) -> str:
    """
    Convert reasoning to a string, handling various types.

    Args:
        reasoning: The reasoning value (str, dict, list, BaseModel, etc.)

    Returns:
        String representation of the reasoning.
    """
    if reasoning is None:
        return ""
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, PydanticBaseModel):
        return reasoning.model_dump_json()
    if isinstance(reasoning, dict | list):
        return json.dumps(reasoning)
    return str(reasoning)


def extract_worst_reasoning(
    *,
    evaluation_results: list[tuple[str, Any]],
    weights_by_name: dict[str, float],
    directions_by_name: dict[str, str],
    worst_n: int,
) -> list[str]:
    """
    Extract reasoning from worst-performing evaluation items.

    Args:
        evaluation_results: List of (evaluator_name, EvalOutput) tuples.
        weights_by_name: Metric weights by evaluator name.
        directions_by_name: Optimization direction ('maximize' or 'minimize') by evaluator name.
        worst_n: Number of worst items to extract.

    Returns:
        List of formatted reasoning strings with evaluator labels.
    """
    # Collect items with evaluator weights: (priority_score, reasoning, evaluator_name)
    weighted_items: list[tuple[float, str, str]] = []

    for name, result in evaluation_results:
        evaluator_weight = weights_by_name.get(name, 1.0)
        direction = directions_by_name.get(name, "maximize")

        for item in result.eval_output_items:
            if not item.reasoning:
                continue

            # Convert reasoning to string (handles dict, BaseModel, list, etc.)
            reasoning_str = _reasoning_to_string(item.reasoning)
            if not reasoning_str:
                continue

            score = float(item.score)
            # For maximize: lower is worse, use score directly (low values sort first)
            # For minimize: higher is worse, negate so high values sort first
            if direction == "minimize":
                score = -score
                # For negative scores, multiply so higher weight increases priority (more negative -> earlier)
                priority_score = score * max(evaluator_weight, 0.01)
            else:
                # For positive scores, divide so higher weight increases priority (smaller -> earlier)
                priority_score = score / max(evaluator_weight, 0.01)
            weighted_items.append((priority_score, reasoning_str, name))

    # Sort by priority (worst weighted failures first)
    weighted_items.sort(key=lambda x: x[0])
    worst = weighted_items[:worst_n]

    # Format with evaluator context
    return [f"[{evaluator}] {reasoning}" for _, reasoning, evaluator in worst]
