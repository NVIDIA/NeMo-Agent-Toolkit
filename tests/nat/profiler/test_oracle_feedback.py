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
Unit tests for oracle feedback functionality.

Tests cover feedback extraction, formatting, injection logic, adaptive triggers,
and type conversions for various reasoning formats.
"""

from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.profiler.parameter_optimization.oracle_feedback import _reasoning_to_string
from nat.profiler.parameter_optimization.oracle_feedback import build_oracle_feedback
from nat.profiler.parameter_optimization.oracle_feedback import check_adaptive_triggers
from nat.profiler.parameter_optimization.oracle_feedback import extract_worst_reasoning
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
        assert (should_inject_feedback(
            mode="never",
            scalar_fitness=0.1,
            fitness_threshold=0.3,
            adaptive_enabled=True,
        ) is False)

    def test_always_mode_returns_true(self):
        """Always mode always returns True."""
        assert (should_inject_feedback(
            mode="always",
            scalar_fitness=0.9,
            fitness_threshold=0.3,
            adaptive_enabled=False,
        ) is True)

    def test_failing_only_below_threshold(self):
        """Failing_only returns True when below threshold."""
        assert (should_inject_feedback(
            mode="failing_only",
            scalar_fitness=0.2,
            fitness_threshold=0.3,
            adaptive_enabled=False,
        ) is True)

    def test_failing_only_above_threshold(self):
        """Failing_only returns False when above threshold."""
        assert (should_inject_feedback(
            mode="failing_only",
            scalar_fitness=0.5,
            fitness_threshold=0.3,
            adaptive_enabled=False,
        ) is False)

    def test_adaptive_when_enabled(self):
        """Adaptive returns True when adaptive_enabled is True."""
        assert (should_inject_feedback(
            mode="adaptive",
            scalar_fitness=0.9,
            fitness_threshold=0.3,
            adaptive_enabled=True,
        ) is True)

    def test_adaptive_when_not_enabled(self):
        """Adaptive returns False when adaptive_enabled is False."""
        assert (should_inject_feedback(
            mode="adaptive",
            scalar_fitness=0.1,
            fitness_threshold=0.3,
            adaptive_enabled=False,
        ) is False)

    def test_unknown_mode_returns_false(self):
        """Unknown mode returns False as safe default."""
        assert (should_inject_feedback(
            mode="unknown",
            scalar_fitness=0.5,
            fitness_threshold=0.3,
            adaptive_enabled=True,
        ) is False)


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


class TestReasoningToString:
    """Tests for _reasoning_to_string helper."""

    def test_none_returns_empty_string(self):
        assert _reasoning_to_string(None) == ""

    def test_string_returns_unchanged(self):
        assert _reasoning_to_string("test") == "test"

    def test_dict_returns_json(self):
        result = _reasoning_to_string({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_list_returns_json(self):
        result = _reasoning_to_string(["a", "b"])
        assert '"a"' in result
        assert '"b"' in result

    def test_basemodel_returns_json(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            field: str

        result = _reasoning_to_string(TestModel(field="test"))
        assert "field" in result
        assert "test" in result

    def test_other_types_use_str(self):
        assert _reasoning_to_string(123) == "123"
        assert _reasoning_to_string(45.67) == "45.67"


class TestExtractWorstReasoning:
    """Tests for extracting reasoning from worst-performing items."""

    def test_empty_results_returns_empty(self):
        """Returns empty list when no results."""
        result = extract_worst_reasoning(
            evaluation_results=[],
            weights_by_name={},
            directions_by_name={},
            worst_n=5,
        )
        assert result == []

    def test_extracts_reasoning_from_lowest_scores(self):
        """Extracts reasoning from lowest-scoring items."""
        items = [
            EvalOutputItem(id=1, score=0.9, reasoning="Good answer"),
            EvalOutputItem(id=2, score=0.2, reasoning="Bad answer"),
            EvalOutputItem(id=3, score=0.5, reasoning="Medium answer"),
        ]
        eval_output = EvalOutput(average_score=0.53, eval_output_items=items)

        result = extract_worst_reasoning(
            evaluation_results=[("Accuracy", eval_output)],
            weights_by_name={"Accuracy": 1.0},
            directions_by_name={"Accuracy": "maximize"},
            worst_n=2,
        )
        assert len(result) == 2
        assert "[Accuracy] Bad answer" in result[0]
        assert "[Accuracy] Medium answer" in result[1]

    def test_skips_items_without_reasoning(self):
        """Skips items that have no reasoning."""
        items = [
            EvalOutputItem(id=1, score=0.2, reasoning=None),
            EvalOutputItem(id=2, score=0.3, reasoning="Has reasoning"),
        ]
        eval_output = EvalOutput(average_score=0.25, eval_output_items=items)

        result = extract_worst_reasoning(
            evaluation_results=[("Accuracy", eval_output)],
            weights_by_name={"Accuracy": 1.0},
            directions_by_name={"Accuracy": "maximize"},
            worst_n=5,
        )
        assert len(result) == 1
        assert "Has reasoning" in result[0]

    def test_converts_dict_reasoning_to_string(self):
        """Converts dict reasoning to JSON string."""
        items = [
            EvalOutputItem(id=1, score=0.2, reasoning={
                "error": "Failed", "details": "Missing info"
            }),
        ]
        eval_output = EvalOutput(average_score=0.2, eval_output_items=items)

        result = extract_worst_reasoning(
            evaluation_results=[("Accuracy", eval_output)],
            weights_by_name={"Accuracy": 1.0},
            directions_by_name={"Accuracy": "maximize"},
            worst_n=5,
        )
        assert len(result) == 1
        assert "error" in result[0]
        assert "Failed" in result[0]

    def test_converts_basemodel_reasoning_to_string(self):
        """Converts Pydantic BaseModel reasoning to JSON string."""
        from pydantic import BaseModel

        class ReasoningModel(BaseModel):
            error: str
            score_breakdown: dict[str, float]

        reasoning_obj = ReasoningModel(error="Failed validation", score_breakdown={"accuracy": 0.2})
        items = [
            EvalOutputItem(id=1, score=0.2, reasoning=reasoning_obj),
        ]
        eval_output = EvalOutput(average_score=0.2, eval_output_items=items)

        result = extract_worst_reasoning(
            evaluation_results=[("Accuracy", eval_output)],
            weights_by_name={"Accuracy": 1.0},
            directions_by_name={"Accuracy": "maximize"},
            worst_n=5,
        )
        assert len(result) == 1
        assert "Failed validation" in result[0]

    def test_handles_list_reasoning(self):
        """Converts list reasoning to string."""
        items = [
            EvalOutputItem(id=1, score=0.2, reasoning=["Error 1", "Error 2"]),
        ]
        eval_output = EvalOutput(average_score=0.2, eval_output_items=items)

        result = extract_worst_reasoning(
            evaluation_results=[("Accuracy", eval_output)],
            weights_by_name={"Accuracy": 1.0},
            directions_by_name={"Accuracy": "maximize"},
            worst_n=5,
        )
        assert len(result) == 1
        assert "Error 1" in result[0]
        assert "Error 2" in result[0]

    def test_weights_affect_priority(self):
        """Higher-weighted evaluator failures appear first."""
        items_acc = [EvalOutputItem(id=1, score=0.3, reasoning="Accuracy fail")]
        items_rel = [EvalOutputItem(id=2, score=0.3, reasoning="Relevance fail")]
        eval_acc = EvalOutput(average_score=0.3, eval_output_items=items_acc)
        eval_rel = EvalOutput(average_score=0.3, eval_output_items=items_rel)

        result = extract_worst_reasoning(
            evaluation_results=[("Accuracy", eval_acc), ("Relevance", eval_rel)],
            weights_by_name={
                "Accuracy": 2.0, "Relevance": 1.0
            },
            directions_by_name={
                "Accuracy": "maximize", "Relevance": "maximize"
            },
            worst_n=2,
        )
        # Higher weight means more important, so Accuracy fail should be first
        assert "Accuracy fail" in result[0]
        assert "Relevance fail" in result[1]

    def test_minimize_direction_handled(self):
        """Handles minimize direction correctly (lower is better)."""
        items = [
            EvalOutputItem(id=1, score=0.1, reasoning="Low score"),
            EvalOutputItem(id=2, score=0.9, reasoning="High score"),
        ]
        eval_output = EvalOutput(average_score=0.5, eval_output_items=items)

        result = extract_worst_reasoning(
            evaluation_results=[("Latency", eval_output)],
            weights_by_name={"Latency": 1.0},
            directions_by_name={"Latency": "minimize"},  # Lower is better
            worst_n=1,
        )
        # For minimize, high score is worst
        assert "High score" in result[0]
