# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nat.profiler.parameter_optimization.oracle_feedback import build_oracle_feedback
from nat.profiler.parameter_optimization.oracle_feedback import check_adaptive_triggers
from nat.profiler.parameter_optimization.oracle_feedback import should_inject_feedback


class TestBuildOracleFeedback:
    """Tests for build_oracle_feedback function."""

    def test_empty_reasoning_returns_none(self):
        """Returns None when no reasoning provided."""
        result = build_oracle_feedback([], max_chars=4000)
        assert result is None

    def test_single_reasoning(self):
        """Formats single reasoning item correctly."""
        result = build_oracle_feedback(["Failed to answer question"], max_chars=4000)
        assert result == "1. Failed to answer question\n"

    def test_multiple_reasoning(self):
        """Formats multiple reasoning items with numbers."""
        reasons = ["First failure", "Second failure", "Third failure"]
        result = build_oracle_feedback(reasons, max_chars=4000)
        assert result == "1. First failure\n2. Second failure\n3. Third failure\n"

    def test_truncation_at_char_limit(self):
        """Truncates reasoning to fit within max_chars."""
        reasons = ["A" * 100, "B" * 100, "C" * 100]
        result = build_oracle_feedback(reasons, max_chars=120)
        # Should include first item and partial second
        assert result is not None
        assert len(result) <= 120
        assert "1. " in result
        assert "..." in result  # Truncation indicator

    def test_skips_entry_if_no_meaningful_space(self):
        """Skips entries when remaining space is too small."""
        reasons = ["A" * 50]
        result = build_oracle_feedback(reasons, max_chars=10)
        # Not enough space for even "1. " + content
        assert result is None or len(result) <= 10

    def test_preserves_evaluator_labels(self):
        """Preserves evaluator labels in reasoning."""
        reasons = ["[Accuracy] Score too low", "[Relevance] Off topic"]
        result = build_oracle_feedback(reasons, max_chars=4000)
        assert "[Accuracy]" in result
        assert "[Relevance]" in result


class TestShouldInjectFeedback:
    """Tests for should_inject_feedback function."""

    def test_never_mode_returns_false(self):
        """Never mode always returns False."""
        assert should_inject_feedback(
            mode="never",
            scalar_fitness=0.1,
            fitness_threshold=0.3,
            adaptive_enabled=True,
        ) is False

    def test_always_mode_returns_true(self):
        """Always mode always returns True."""
        assert should_inject_feedback(
            mode="always",
            scalar_fitness=0.9,
            fitness_threshold=0.3,
            adaptive_enabled=False,
        ) is True

    def test_failing_only_below_threshold(self):
        """Failing_only returns True when below threshold."""
        assert should_inject_feedback(
            mode="failing_only",
            scalar_fitness=0.2,
            fitness_threshold=0.3,
            adaptive_enabled=False,
        ) is True

    def test_failing_only_above_threshold(self):
        """Failing_only returns False when above threshold."""
        assert should_inject_feedback(
            mode="failing_only",
            scalar_fitness=0.5,
            fitness_threshold=0.3,
            adaptive_enabled=False,
        ) is False

    def test_adaptive_when_enabled(self):
        """Adaptive returns True when adaptive_enabled is True."""
        assert should_inject_feedback(
            mode="adaptive",
            scalar_fitness=0.9,
            fitness_threshold=0.3,
            adaptive_enabled=True,
        ) is True

    def test_adaptive_when_not_enabled(self):
        """Adaptive returns False when adaptive_enabled is False."""
        assert should_inject_feedback(
            mode="adaptive",
            scalar_fitness=0.1,
            fitness_threshold=0.3,
            adaptive_enabled=False,
        ) is False

    def test_unknown_mode_returns_false(self):
        """Unknown mode returns False as safe default."""
        assert should_inject_feedback(
            mode="unknown",
            scalar_fitness=0.5,
            fitness_threshold=0.3,
            adaptive_enabled=True,
        ) is False


class TestCheckAdaptiveTriggers:
    """Tests for adaptive trigger detection."""

    def test_no_trigger_with_improving_fitness(self):
        """No trigger when fitness is improving."""
        result = check_adaptive_triggers(
            best_fitness_history=[0.5, 0.6, 0.7, 0.8],
            population_fitness_values=[0.5, 0.7, 0.9, 0.6],  # variance ~0.029 > 0.01
            population_prompt_keys=[("a", ), ("b", ), ("c", ), ("d", )],
            stagnation_generations=3,
            fitness_variance_threshold=0.01,
            diversity_threshold=0.5,
        )
        assert result["triggered"] is False

    def test_stagnation_trigger(self):
        """Triggers when fitness stagnates."""
        result = check_adaptive_triggers(
            best_fitness_history=[0.5, 0.5, 0.5, 0.5],
            population_fitness_values=[0.4, 0.45, 0.5, 0.48],
            population_prompt_keys=[("a", ), ("b", ), ("c", ), ("d", )],
            stagnation_generations=3,
            fitness_variance_threshold=0.01,
            diversity_threshold=0.5,
        )
        assert result["triggered"] is True
        assert result["reason"] == "stagnation"

    def test_fitness_variance_collapse_trigger(self):
        """Triggers when fitness variance collapses."""
        result = check_adaptive_triggers(
            best_fitness_history=[0.5, 0.6, 0.7],
            population_fitness_values=[0.7, 0.7, 0.7, 0.7],  # No variance
            population_prompt_keys=[("a", ), ("b", ), ("c", ), ("d", )],
            stagnation_generations=3,
            fitness_variance_threshold=0.01,
            diversity_threshold=0.5,
        )
        assert result["triggered"] is True
        assert result["reason"] == "fitness_variance_collapse"

    def test_diversity_collapse_trigger(self):
        """Triggers when prompt diversity collapses."""
        result = check_adaptive_triggers(
            best_fitness_history=[0.5, 0.6, 0.7],
            population_fitness_values=[0.3, 0.6, 0.9, 0.5],  # variance ~0.063 > 0.01
            population_prompt_keys=[("a", ), ("a", ), ("a", ), ("a", )],  # 100% duplicates, unique_ratio=0.25
            stagnation_generations=3,
            fitness_variance_threshold=0.01,
            diversity_threshold=0.5,
        )
        assert result["triggered"] is True
        assert result["reason"] == "diversity_collapse"

    def test_insufficient_history_no_stagnation_check(self):
        """No stagnation check with insufficient history."""
        result = check_adaptive_triggers(
            best_fitness_history=[0.5, 0.5],  # Only 2 generations
            population_fitness_values=[0.3, 0.5, 0.7, 0.6],  # variance ~0.029 > 0.01
            population_prompt_keys=[("a", ), ("b", ), ("c", ), ("d", )],
            stagnation_generations=3,
            fitness_variance_threshold=0.01,
            diversity_threshold=0.5,
        )
        assert result["triggered"] is False
