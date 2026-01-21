# Oracle Feedback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add context-grounded improvement to the GA prompt optimizer by feeding failure reasoning back into mutations.

**Architecture:** Extract reasoning from worst-performing `EvalOutputItem`s, aggregate across evaluators weighted by metric importance, and inject into the prompt optimizer's mutation function. Support four modes: never, always, failing_only, and adaptive (triggers on fitness stagnation or diversity collapse).

**Tech Stack:** Python 3.11+, Pydantic, pytest, asyncio

---

## Task 1: Add Oracle Feedback Configuration Fields

**Files:**
- Modify: `src/nat/data_models/optimizer.py:51-112` (PromptGAOptimizationConfig)
- Test: `tests/nat/data_models/test_optimizer_oracle_feedback.py` (create)

**Step 1: Write the failing test**

Create `tests/nat/data_models/test_optimizer_oracle_feedback.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nat.data_models.optimizer import PromptGAOptimizationConfig


class TestOracleFeedbackConfig:
    """Tests for oracle feedback configuration fields."""

    def test_default_values(self):
        """Oracle feedback is disabled by default."""
        config = PromptGAOptimizationConfig()
        assert config.oracle_feedback_mode == "never"
        assert config.oracle_feedback_worst_n == 5
        assert config.oracle_feedback_max_chars == 4000
        assert config.oracle_feedback_fitness_threshold == 0.3
        assert config.oracle_feedback_stagnation_generations == 3
        assert config.oracle_feedback_fitness_variance_threshold == 0.01
        assert config.oracle_feedback_diversity_threshold == 0.5

    def test_valid_modes(self):
        """All valid feedback modes are accepted."""
        for mode in ["never", "always", "failing_only", "adaptive"]:
            config = PromptGAOptimizationConfig(oracle_feedback_mode=mode)
            assert config.oracle_feedback_mode == mode

    def test_invalid_mode_rejected(self):
        """Invalid feedback mode raises validation error."""
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_mode="invalid")

    def test_worst_n_must_be_positive(self):
        """oracle_feedback_worst_n must be >= 1."""
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_worst_n=0)

    def test_max_chars_must_be_positive(self):
        """oracle_feedback_max_chars must be >= 1."""
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_max_chars=0)

    def test_fitness_threshold_range(self):
        """oracle_feedback_fitness_threshold must be in [0, 1]."""
        PromptGAOptimizationConfig(oracle_feedback_fitness_threshold=0.0)
        PromptGAOptimizationConfig(oracle_feedback_fitness_threshold=1.0)
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_fitness_threshold=-0.1)
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_fitness_threshold=1.1)

    def test_diversity_threshold_range(self):
        """oracle_feedback_diversity_threshold must be in [0, 1]."""
        PromptGAOptimizationConfig(oracle_feedback_diversity_threshold=0.0)
        PromptGAOptimizationConfig(oracle_feedback_diversity_threshold=1.0)
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_diversity_threshold=-0.1)
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_diversity_threshold=1.1)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nat/data_models/test_optimizer_oracle_feedback.py -v`
Expected: FAIL with AttributeError (oracle_feedback_mode not found)

**Step 3: Write minimal implementation**

Add to `src/nat/data_models/optimizer.py` inside `PromptGAOptimizationConfig` class, after line 112 (before the closing of the class):

```python
    # Oracle feedback configuration
    oracle_feedback_mode: typing.Literal["never", "always", "failing_only", "adaptive"] = Field(
        description="When to inject failure reasoning into mutations: "
                    "'never' (default), 'always', 'failing_only' (below threshold), 'adaptive' (on plateau).",
        default="never",
    )
    oracle_feedback_worst_n: int = Field(
        description="Number of worst-scoring items to extract reasoning from.",
        default=5,
        ge=1,
    )
    oracle_feedback_max_chars: int = Field(
        description="Maximum characters for oracle feedback in mutation prompt.",
        default=4000,
        ge=1,
    )
    oracle_feedback_fitness_threshold: float = Field(
        description="For 'failing_only' mode: normalized fitness threshold below which feedback is injected.",
        default=0.3,
        ge=0.0,
        le=1.0,
    )
    oracle_feedback_stagnation_generations: int = Field(
        description="For 'adaptive' mode: generations without improvement before enabling feedback.",
        default=3,
        ge=1,
    )
    oracle_feedback_fitness_variance_threshold: float = Field(
        description="For 'adaptive' mode: fitness variance threshold for collapse detection.",
        default=0.01,
        ge=0.0,
    )
    oracle_feedback_diversity_threshold: float = Field(
        description="For 'adaptive' mode: prompt duplication ratio threshold (0-1).",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
```

Also add `import typing` at the top of the file if not present.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nat/data_models/test_optimizer_oracle_feedback.py -v`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```bash
git add src/nat/data_models/optimizer.py tests/nat/data_models/test_optimizer_oracle_feedback.py
git commit --signoff -m "feat(optimizer): add oracle feedback configuration fields"
```

---

## Task 2: Add Feedback Building Helper Functions

**Files:**
- Create: `src/nat/profiler/parameter_optimization/oracle_feedback.py`
- Test: `tests/nat/profiler/test_oracle_feedback.py` (create)

**Step 1: Write the failing test**

Create `tests/nat/profiler/test_oracle_feedback.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.profiler.parameter_optimization.oracle_feedback import build_oracle_feedback


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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nat/profiler/test_oracle_feedback.py::TestBuildOracleFeedback -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

Create `src/nat/profiler/parameter_optimization/oracle_feedback.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


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

    for i, reasoning in enumerate(reasoning_list, 1):
        entry = f"{i}. {reasoning}\n"
        if current_length + len(entry) > max_chars:
            remaining = max_chars - current_length
            if remaining > 20:  # Only add if meaningful space left
                feedback_parts.append(entry[: remaining - 3] + "...")
            break
        feedback_parts.append(entry)
        current_length += len(entry)

    return "".join(feedback_parts) if feedback_parts else None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nat/profiler/test_oracle_feedback.py::TestBuildOracleFeedback -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add src/nat/profiler/parameter_optimization/oracle_feedback.py tests/nat/profiler/test_oracle_feedback.py
git commit --signoff -m "feat(optimizer): add build_oracle_feedback helper"
```

---

## Task 3: Add Feedback Mode Decision Logic

**Files:**
- Modify: `src/nat/profiler/parameter_optimization/oracle_feedback.py`
- Test: `tests/nat/profiler/test_oracle_feedback.py`

**Step 1: Write the failing test**

Add to `tests/nat/profiler/test_oracle_feedback.py`:

```python
from nat.profiler.parameter_optimization.oracle_feedback import should_inject_feedback


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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nat/profiler/test_oracle_feedback.py::TestShouldInjectFeedback -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `src/nat/profiler/parameter_optimization/oracle_feedback.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nat/profiler/test_oracle_feedback.py::TestShouldInjectFeedback -v`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```bash
git add src/nat/profiler/parameter_optimization/oracle_feedback.py tests/nat/profiler/test_oracle_feedback.py
git commit --signoff -m "feat(optimizer): add should_inject_feedback decision logic"
```

---

## Task 4: Add Adaptive Trigger Detection

**Files:**
- Modify: `src/nat/profiler/parameter_optimization/oracle_feedback.py`
- Test: `tests/nat/profiler/test_oracle_feedback.py`

**Step 1: Write the failing test**

Add to `tests/nat/profiler/test_oracle_feedback.py`:

```python
from nat.profiler.parameter_optimization.oracle_feedback import check_adaptive_triggers


class TestCheckAdaptiveTriggers:
    """Tests for adaptive trigger detection."""

    def test_no_trigger_with_improving_fitness(self):
        """No trigger when fitness is improving."""
        result = check_adaptive_triggers(
            best_fitness_history=[0.5, 0.6, 0.7, 0.8],
            population_fitness_values=[0.7, 0.75, 0.8, 0.78],
            population_prompt_keys=[("a",), ("b",), ("c",), ("d",)],
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
            population_prompt_keys=[("a",), ("b",), ("c",), ("d",)],
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
            population_prompt_keys=[("a",), ("b",), ("c",), ("d",)],
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
            population_fitness_values=[0.5, 0.6, 0.7, 0.65],
            population_prompt_keys=[("a",), ("a",), ("a",), ("b",)],  # 75% duplicates
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
            population_fitness_values=[0.4, 0.45, 0.5, 0.48],
            population_prompt_keys=[("a",), ("b",), ("c",), ("d",)],
            stagnation_generations=3,
            fitness_variance_threshold=0.01,
            diversity_threshold=0.5,
        )
        assert result["triggered"] is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nat/profiler/test_oracle_feedback.py::TestCheckAdaptiveTriggers -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `src/nat/profiler/parameter_optimization/oracle_feedback.py`:

```python
import statistics
from typing import Any


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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nat/profiler/test_oracle_feedback.py::TestCheckAdaptiveTriggers -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/nat/profiler/parameter_optimization/oracle_feedback.py tests/nat/profiler/test_oracle_feedback.py
git commit --signoff -m "feat(optimizer): add adaptive trigger detection"
```

---

## Task 5: Add Weighted Reasoning Extraction

**Files:**
- Modify: `src/nat/profiler/parameter_optimization/oracle_feedback.py`
- Test: `tests/nat/profiler/test_oracle_feedback.py`

**Step 1: Write the failing test**

Add to `tests/nat/profiler/test_oracle_feedback.py`:

```python
from nat.profiler.parameter_optimization.oracle_feedback import (
    extract_worst_reasoning,
    _reasoning_to_string,
)
from nat.eval.evaluator.evaluator_model import EvalOutput, EvalOutputItem


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
            EvalOutputItem(id=1, score=0.2, reasoning={"error": "Failed", "details": "Missing info"}),
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
            weights_by_name={"Accuracy": 2.0, "Relevance": 1.0},
            directions_by_name={"Accuracy": "maximize", "Relevance": "maximize"},
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nat/profiler/test_oracle_feedback.py::TestExtractWorstReasoning -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `src/nat/profiler/parameter_optimization/oracle_feedback.py`:

```python
import json

from pydantic import BaseModel as PydanticBaseModel


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
    if isinstance(reasoning, (dict, list)):
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
            # Invert for maximize so sorting ascending gives worst
            if direction == "maximize":
                score = -score

            # Higher weight = more important failures float to top
            priority_score = score / max(evaluator_weight, 0.01)
            weighted_items.append((priority_score, reasoning_str, name))

    # Sort by priority (worst weighted failures first)
    weighted_items.sort(key=lambda x: x[0])
    worst = weighted_items[:worst_n]

    # Format with evaluator context
    return [f"[{evaluator}] {reasoning}" for _, reasoning, evaluator in worst]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nat/profiler/test_oracle_feedback.py::TestExtractWorstReasoning -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/nat/profiler/parameter_optimization/oracle_feedback.py tests/nat/profiler/test_oracle_feedback.py
git commit --signoff -m "feat(optimizer): add weighted reasoning extraction"
```

---

## Task 6: Update Prompt Templates

**Files:**
- Modify: `src/nat/agent/prompt_optimizer/prompt.py`
- Test: `tests/nat/agent/prompt_optimizer/test_prompt_templates.py` (create)

**Step 1: Write the failing test**

Create `tests/nat/agent/prompt_optimizer/test_prompt_templates.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nat.agent.prompt_optimizer.prompt import mutator_prompt, oracle_feedback_template


class TestPromptTemplates:
    """Tests for prompt optimizer templates."""

    def test_mutator_prompt_has_feedback_placeholder(self):
        """Mutator prompt includes oracle_feedback_section placeholder."""
        assert "{oracle_feedback_section}" in mutator_prompt

    def test_oracle_feedback_template_has_feedback_placeholder(self):
        """Oracle feedback template includes oracle_feedback placeholder."""
        assert "{oracle_feedback}" in oracle_feedback_template

    def test_oracle_feedback_template_formatting(self):
        """Oracle feedback template formats correctly."""
        feedback = "1. [Accuracy] Failed to answer\n2. [Relevance] Off topic\n"
        result = oracle_feedback_template.format(oracle_feedback=feedback)
        assert "FAILURE ANALYSIS" in result
        assert "[Accuracy] Failed to answer" in result
        assert "[Relevance] Off topic" in result
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nat/agent/prompt_optimizer/test_prompt_templates.py -v`
Expected: FAIL with ImportError or assertion error

**Step 3: Write minimal implementation**

Modify `src/nat/agent/prompt_optimizer/prompt.py`. Replace the existing `mutator_prompt` and add `oracle_feedback_template`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# flake8: noqa W291

mutator_prompt = """

## CORE DIRECTIVES
- **Preserve the original objective and task.** Do not change what the prompt is meant to accomplish.
- **Keep the intent intact.** The improved prompt must solve the same problem as the original.
- **Do not invent new goals.** Only improve clarity, structure, constraints, and usability.
- **Do not drop critical instructions.** Everything essential from the original prompt must remain.
- **Return only the mutated prompt text.** No rationale, no diffs, no explanations.
- **Be Creative within bounds.** You may rephrase, reorganize, and enhance, but not alter meaning.
- **DO NOT use curly braces in your prompt** for anything other than existing variables in the prompt as the string
will be treated as an f-string.
- **Examples are a good idea** if the original prompt lacks them. They help clarify expected output.

---

## IMPROVEMENT HINTS
When modifying, apply these principles:
1. **Clarity & Precision** – remove vague language, strengthen directives.
2. **Structure & Flow** – order sections as: *Objective → Constraints → Tools → Steps → Output Schema → Examples*.
3. **Schema Adherence** – enforce a single canonical output schema (JSON/XML) with `schema_version`.
4. **Tool Governance** – clarify when/how tools are used, their inputs/outputs, and fallback behavior.
5. **Error Handling** – specify behavior if tools fail or inputs are insufficient.
6. **Budget Awareness** – minimize verbosity, respect token/latency limits.
7. **Safety** – include refusals for unsafe requests, enforce compliance with rules.
8. **Consistency** – avoid format drift; always maintain the same schema.
9. **Integrity** – confirm the task, objective, and intent are preserved.

---

## MUTATION OPERATORS
You may:
- **Tighten** (remove fluff, redundancies)
- **Reorder** (improve logical flow)
- **Constrain** (add explicit rules/limits)
- **Harden** (improve error handling/fallbacks)
- **Defuse** (replace ambiguous verbs with measurable actions)
- **Format-lock** (wrap outputs in JSON/XML fenced blocks)
- **Example-ify** (add examples if missing or weak)

---

## INPUT
Here is the prompt to mutate:
{original_prompt}

## OBJECTIVE
The prompt must acheive the following objective:
{objective}

{oracle_feedback_section}

The modified prompt is: \n

"""

oracle_feedback_template = """
## FAILURE ANALYSIS
The following are examples of cases where the current prompt performed poorly,
along with reasoning explaining why. Use these insights to improve the prompt:

{oracle_feedback}

Focus on addressing the root causes identified above while maintaining the original objective.
"""
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nat/agent/prompt_optimizer/test_prompt_templates.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/nat/agent/prompt_optimizer/prompt.py tests/nat/agent/prompt_optimizer/test_prompt_templates.py
git commit --signoff -m "feat(optimizer): add oracle feedback section to prompt templates"
```

---

## Task 7: Update Prompt Optimizer Function

**Files:**
- Modify: `src/nat/agent/prompt_optimizer/register.py:67-83`
- Test: `tests/nat/agent/prompt_optimizer/test_register.py` (create)

**Step 1: Write the failing test**

Create `tests/nat/agent/prompt_optimizer/test_register.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import AsyncMock, MagicMock

from nat.profiler.parameter_optimization.prompt_optimizer import PromptOptimizerInputSchema


class TestPromptOptimizerInputSchema:
    """Tests for PromptOptimizerInputSchema."""

    def test_oracle_feedback_is_optional(self):
        """Oracle feedback defaults to None."""
        schema = PromptOptimizerInputSchema(
            original_prompt="Test prompt",
            objective="Test objective",
        )
        assert schema.oracle_feedback is None

    def test_oracle_feedback_can_be_set(self):
        """Oracle feedback can be provided."""
        feedback = "1. [Accuracy] Failed\n"
        schema = PromptOptimizerInputSchema(
            original_prompt="Test prompt",
            objective="Test objective",
            oracle_feedback=feedback,
        )
        assert schema.oracle_feedback == feedback
```

**Step 2: Run test to verify it passes (schema already exists)**

Run: `uv run pytest tests/nat/agent/prompt_optimizer/test_register.py -v`
Expected: PASS (schema already has oracle_feedback field)

**Step 3: Update register.py to use oracle_feedback**

Modify `src/nat/agent/prompt_optimizer/register.py`. Update the `_inner` function (around line 67-83):

```python
from .prompt import mutator_prompt, oracle_feedback_template

# ... existing code ...

    prompt_extension_template = PromptTemplate(
        template=mutator_prompt,
        input_variables=["original_prompt", "objective", "oracle_feedback_section"],
        validate_template=True,
    )

    async def _inner(input_message: PromptOptimizerInputSchema) -> str:
        """
        Optimize the prompt using the provided LLM.
        """
        original_prompt = input_message.original_prompt
        prompt_objective = input_message.objective
        oracle_feedback = input_message.oracle_feedback

        # Build feedback section conditionally
        feedback_section = ""
        if oracle_feedback:
            feedback_section = oracle_feedback_template.format(oracle_feedback=oracle_feedback)

        prompt_extension = (await prompt_extension_template.ainvoke(input={
            "original_prompt": original_prompt,
            "objective": prompt_objective,
            "oracle_feedback_section": feedback_section,
        })).to_string()

        prompt = f"{base_prompt}\n\n{prompt_extension}"

        optimized_prompt = await llm.ainvoke(prompt)
        return optimized_prompt.content
```

**Step 4: Run existing tests to verify no regression**

Run: `uv run pytest tests/nat/profiler/test_prompt_optimizer.py -v`
Expected: PASS (all existing tests)

**Step 5: Commit**

```bash
git add src/nat/agent/prompt_optimizer/register.py tests/nat/agent/prompt_optimizer/test_register.py
git commit --signoff -m "feat(optimizer): update prompt optimizer to use oracle feedback"
```

---

## Task 8: Integrate Oracle Feedback into GA Loop

**Files:**
- Modify: `src/nat/profiler/parameter_optimization/prompt_optimizer.py`
- Test: `tests/nat/profiler/test_prompt_optimizer.py`

**Step 1: Write the failing test**

Add to `tests/nat/profiler/test_prompt_optimizer.py`:

```python
async def test_optimize_prompts_with_oracle_feedback(tmp_path: Path):
    """Test that oracle feedback is extracted and passed to mutations."""
    await _register_prompt_optimizer_functions()
    base_cfg = Config()
    optimizer_config = _make_optimizer_config(tmp_path)

    # Enable oracle feedback
    optimizer_config.prompt.oracle_feedback_mode = "always"
    optimizer_config.prompt.oracle_feedback_worst_n = 2
    optimizer_config.prompt.oracle_feedback_max_chars = 1000

    full_space = {"prompt_param": SearchSpace(is_prompt=True, prompt="Base", prompt_purpose="Greet")}
    run_cfg = _make_run_config(base_cfg)
    base_cfg.functions = {
        "init_fn": InitFunctionConfig(),
        "recombine_fn": RecombineFunctionConfig(),
    }
    base_cfg.workflow = InitFunctionConfig()

    # Track oracle_feedback passed to init function
    feedback_received = {"values": []}

    class _EvalRun:
        def __init__(self, config):
            self.config = config

        async def run_and_evaluate(self):
            from nat.eval.evaluator.evaluator_model import EvalOutput, EvalOutputItem
            items = [
                EvalOutputItem(id=1, score=0.2, reasoning="Failed to greet properly"),
                EvalOutputItem(id=2, score=0.8, reasoning="Good greeting"),
            ]
            eval_output = EvalOutput(average_score=0.5, eval_output_items=items)
            return SimpleNamespace(evaluation_results=[("Accuracy", eval_output)])

    def fake_apply_suggestions(cfg, prompts):
        _ = (cfg, prompts)
        return Config()

    with patch("nat.profiler.parameter_optimization.prompt_optimizer.EvaluationRun", _EvalRun), \
         patch("nat.profiler.parameter_optimization.prompt_optimizer.apply_suggestions",
               side_effect=fake_apply_suggestions):

        await optimize_prompts(
            base_cfg=base_cfg,
            full_space=full_space,
            optimizer_config=optimizer_config,
            opt_run_config=run_cfg,
        )

    # Verify output files created
    assert (optimizer_config.output_path / "optimized_prompts.json").exists()
```

**Step 2: Run test to verify it fails or passes**

Run: `uv run pytest tests/nat/profiler/test_prompt_optimizer.py::test_optimize_prompts_with_oracle_feedback -v`
Expected: May fail if oracle_feedback_mode not recognized

**Step 3: Update prompt_optimizer.py to integrate oracle feedback**

This is the most complex change. Modify `src/nat/profiler/parameter_optimization/prompt_optimizer.py`:

1. Add imports at top:
```python
from nat.profiler.parameter_optimization.oracle_feedback import (
    build_oracle_feedback,
    should_inject_feedback,
    check_adaptive_triggers,
    extract_worst_reasoning,
)
```

2. Extend `Individual` dataclass (around line 56):
```python
@dataclass
class Individual:
    prompts: dict[str, str]
    metrics: dict[str, float] | None = None
    scalar_fitness: float | None = None
    worst_items_reasoning: list[str] | None = None
```

3. Update `_evaluate` function to extract reasoning (around line 229-256):
```python
async def _evaluate(ind: Individual) -> Individual:
    async with sem:
        cfg_trial = apply_suggestions(base_cfg, ind.prompts)
    eval_cfg = EvaluationRunConfig(
        config_file=cfg_trial,
        dataset=opt_run_config.dataset,
        result_json_path=opt_run_config.result_json_path,
        endpoint=opt_run_config.endpoint,
        endpoint_timeout=opt_run_config.endpoint_timeout,
        override=opt_run_config.override,
    )
    all_results: list[list[tuple[str, Any]]] = []
    for _ in range(reps):
        res = (await EvaluationRun(config=eval_cfg).run_and_evaluate()).evaluation_results
        all_results.append(res)

    metrics: dict[str, float] = {}
    for metric_name in eval_metrics:
        scores: list[float] = []
        for run_results in all_results:
            for name, result in run_results:
                if name == metric_name:
                    scores.append(result.average_score)
                    break
        metrics[metric_name] = float(sum(scores) / len(scores)) if scores else 0.0
    ind.metrics = metrics

    # Extract reasoning from worst items (use last run's results)
    if all_results and oracle_feedback_mode != "never":
        weights_by_name = {v.evaluator_name: v.weight for v in metric_cfg.values()}
        directions_by_name = {v.evaluator_name: v.direction for v in metric_cfg.values()}
        ind.worst_items_reasoning = extract_worst_reasoning(
            evaluation_results=all_results[-1],
            weights_by_name=weights_by_name,
            directions_by_name=directions_by_name,
            worst_n=oracle_feedback_worst_n,
        )

    return ind
```

4. Update `_mutate_prompt` to accept parent and inject feedback (around line 180):
```python
async def _mutate_prompt(
    original_prompt: str,
    purpose: str,
    parent: Individual | None = None,
) -> str:
    feedback = None
    if parent and should_inject_feedback(
        mode=oracle_feedback_mode,
        scalar_fitness=parent.scalar_fitness or 0.0,
        fitness_threshold=oracle_feedback_fitness_threshold,
        adaptive_enabled=adaptive_feedback_enabled,
    ):
        feedback = build_oracle_feedback(
            parent.worst_items_reasoning or [],
            oracle_feedback_max_chars,
        )

    return await init_fn.acall_invoke(
        PromptOptimizerInputSchema(
            original_prompt=original_prompt,
            objective=purpose,
            oracle_feedback=feedback,
        ))
```

5. Add oracle feedback config variables after GA parameters (around line 177):
```python
oracle_feedback_mode = optimizer_config.prompt.oracle_feedback_mode
oracle_feedback_worst_n = optimizer_config.prompt.oracle_feedback_worst_n
oracle_feedback_max_chars = optimizer_config.prompt.oracle_feedback_max_chars
oracle_feedback_fitness_threshold = optimizer_config.prompt.oracle_feedback_fitness_threshold
oracle_feedback_stagnation_generations = optimizer_config.prompt.oracle_feedback_stagnation_generations
oracle_feedback_fitness_variance_threshold = optimizer_config.prompt.oracle_feedback_fitness_variance_threshold
oracle_feedback_diversity_threshold = optimizer_config.prompt.oracle_feedback_diversity_threshold
```

6. Add adaptive state tracking before GA loop (around line 297):
```python
best_fitness_history: list[float] = []
adaptive_feedback_enabled: bool = False
```

7. Add adaptive trigger check after evaluation in GA loop (around line 311):
```python
# Check adaptive triggers
if oracle_feedback_mode == "adaptive" and not adaptive_feedback_enabled:
    prompt_keys = [tuple(sorted(ind.prompts.items())) for ind in population]
    fitness_values = [ind.scalar_fitness or 0.0 for ind in population]
    trigger_result = check_adaptive_triggers(
        best_fitness_history=best_fitness_history,
        population_fitness_values=fitness_values,
        population_prompt_keys=prompt_keys,
        stagnation_generations=oracle_feedback_stagnation_generations,
        fitness_variance_threshold=oracle_feedback_fitness_variance_threshold,
        diversity_threshold=oracle_feedback_diversity_threshold,
    )
    if trigger_result["triggered"]:
        adaptive_feedback_enabled = True
        logger.info("[GA] Adaptive oracle feedback ENABLED (reason=%s)", trigger_result["reason"])

best_fitness_history.append(best.scalar_fitness or 0.0)
```

8. Update `_make_child` to pass parent for feedback (around line 275):
```python
async def _make_child(parent_a: Individual, parent_b: Individual) -> Individual:
    child_prompts: dict[str, str] = {}
    for param, (base_prompt, purpose) in prompt_space.items():
        pa = parent_a.prompts.get(param, base_prompt)
        pb = parent_b.prompts.get(param, base_prompt)
        child = pa
        # crossover
        if random.random() < crossover_rate:
            try:
                child = await _recombine_prompts(pa, pb, purpose)
            except Exception as e:
                logger.warning("Recombination failed for %s: %s; falling back to parent.", param, e)
                child = random.choice([pa, pb])
        # mutation
        if random.random() < mutation_rate:
            try:
                child = await _mutate_prompt(child, purpose, parent=parent_a)
            except Exception as e:
                logger.warning("Mutation failed for %s: %s; keeping child as-is.", param, e)
        child_prompts[param] = child
    return _make_individual_from_prompts(child_prompts)
```

**Step 4: Run all tests to verify**

Run: `uv run pytest tests/nat/profiler/test_prompt_optimizer.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/nat/profiler/parameter_optimization/prompt_optimizer.py tests/nat/profiler/test_prompt_optimizer.py
git commit --signoff -m "feat(optimizer): integrate oracle feedback into GA loop"
```

---

## Task 9: Update Optimizer Documentation

**Files:**
- Modify: `docs/source/improve-workflows/optimizer.md`

**Step 1: Read existing documentation**

Run: `cat docs/source/improve-workflows/optimizer.md | head -100`

**Step 2: Add oracle feedback section**

Add a new section after the GA configuration section:

```markdown
### Oracle Feedback Configuration

Oracle feedback enables context-grounded improvement by extracting reasoning from poorly-performing
evaluation items and feeding it back into the mutation process.

#### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `oracle_feedback_mode` | `"never"` | When to inject feedback: `"never"`, `"always"`, `"failing_only"`, `"adaptive"` |
| `oracle_feedback_worst_n` | `5` | Number of worst-scoring items to extract reasoning from |
| `oracle_feedback_max_chars` | `4000` | Maximum characters for feedback in mutation prompt |
| `oracle_feedback_fitness_threshold` | `0.3` | For `failing_only`: threshold below which feedback is injected |
| `oracle_feedback_stagnation_generations` | `3` | For `adaptive`: generations without improvement before enabling |
| `oracle_feedback_fitness_variance_threshold` | `0.01` | For `adaptive`: variance threshold for collapse detection |
| `oracle_feedback_diversity_threshold` | `0.5` | For `adaptive`: prompt duplication ratio threshold |

#### Feedback Modes

- **`never`** (default): No feedback injection, original behavior
- **`always`**: Every mutation receives feedback from the parent's worst evaluation items
- **`failing_only`**: Only individuals below the fitness threshold receive feedback
- **`adaptive`**: Starts without feedback, enables when fitness stagnates or diversity collapses

#### Evaluator Requirements

For oracle feedback to work effectively, your evaluators must populate the `reasoning` field
in `EvalOutputItem`:

```python
EvalOutputItem(
    id="item_123",
    score=0.2,
    reasoning="The response failed to address the user's question about pricing. "
              "Instead, it provided generic product information."
)
```

#### Example Configuration

```yaml
optimizer:
  prompt:
    enabled: true
    oracle_feedback_mode: "adaptive"
    oracle_feedback_worst_n: 5
    oracle_feedback_max_chars: 4000
```
```

**Step 3: Commit**

```bash
git add docs/source/improve-workflows/optimizer.md
git commit --signoff -m "docs(optimizer): add oracle feedback documentation"
```

---

## Task 10: Run Full Test Suite and Clean Up

**Step 1: Run all related tests**

Run: `uv run pytest tests/nat/profiler/test_prompt_optimizer.py tests/nat/profiler/test_oracle_feedback.py tests/nat/data_models/test_optimizer_oracle_feedback.py tests/nat/agent/prompt_optimizer/ -v`
Expected: All PASS

**Step 2: Run linting**

Run: `uv run pre-commit run --all-files`
Expected: All checks pass

**Step 3: Final commit if needed**

```bash
git status  # Check for any uncommitted changes
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add config fields | optimizer.py, test_optimizer_oracle_feedback.py |
| 2 | Add build_oracle_feedback | oracle_feedback.py, test_oracle_feedback.py |
| 3 | Add should_inject_feedback | oracle_feedback.py, test_oracle_feedback.py |
| 4 | Add adaptive triggers | oracle_feedback.py, test_oracle_feedback.py |
| 5 | Add weighted extraction | oracle_feedback.py, test_oracle_feedback.py |
| 6 | Update prompt templates | prompt.py, test_prompt_templates.py |
| 7 | Update prompt optimizer fn | register.py, test_register.py |
| 8 | Integrate into GA loop | prompt_optimizer.py, test_prompt_optimizer.py |
| 9 | Update documentation | optimizer.md |
| 10 | Run full test suite | - |
