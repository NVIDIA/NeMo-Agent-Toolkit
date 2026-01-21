# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import statistics
from typing import Any


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
        if (max(recent) - min(recent)) < 0.001:
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
