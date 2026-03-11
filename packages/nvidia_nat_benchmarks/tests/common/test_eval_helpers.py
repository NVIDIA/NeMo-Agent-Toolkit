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
"""Tests for common.eval_helpers — the shared evaluator loop."""

from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutputItem
from nat.plugins.benchmarks.common.eval_helpers import run_evaluator_loop


def _make_item(item_id: str, output: str = "output", expected: str = "expected") -> EvalInputItem:
    """Create a minimal EvalInputItem for testing."""
    return EvalInputItem(
        id=item_id,
        input_obj="input",
        output_obj=output,
        expected_output_obj=expected,
        full_dataset_entry={"id": item_id},
    )


class TestRunEvaluatorLoop:
    """Tests for run_evaluator_loop."""

    def test_basic_scoring(self):
        """All items score 1.0 => average is 1.0."""
        items = [_make_item("a"), _make_item("b"), _make_item("c")]
        eval_input = EvalInput(eval_input_items=items)

        def score_fn(item: EvalInputItem) -> EvalOutputItem:
            return EvalOutputItem(id=item.id, score=1.0, reasoning={"ok": True})

        result = run_evaluator_loop(eval_input, score_fn, "TestBench")

        assert result.average_score == 1.0
        assert len(result.eval_output_items) == 3
        assert all(item.score == 1.0 for item in result.eval_output_items)

    def test_mixed_scores(self):
        """Average of [1.0, 0.0, 0.5] => 0.5."""
        items = [_make_item("a"), _make_item("b"), _make_item("c")]
        eval_input = EvalInput(eval_input_items=items)
        scores = [1.0, 0.0, 0.5]

        def score_fn(item: EvalInputItem) -> EvalOutputItem:
            idx = ["a", "b", "c"].index(item.id)
            return EvalOutputItem(id=item.id, score=scores[idx], reasoning={})

        result = run_evaluator_loop(eval_input, score_fn, "TestBench")

        assert result.average_score == 0.5
        assert len(result.eval_output_items) == 3

    def test_empty_input(self):
        """Empty input => average 0.0, no items."""
        eval_input = EvalInput(eval_input_items=[])

        def score_fn(item: EvalInputItem) -> EvalOutputItem:
            raise AssertionError("Should not be called")

        result = run_evaluator_loop(eval_input, score_fn, "TestBench")

        assert result.average_score == 0.0
        assert len(result.eval_output_items) == 0

    def test_error_handling_gives_zero_score(self):
        """When evaluate_item_fn raises, that item gets score=0.0 with error in reasoning."""
        items = [_make_item("good"), _make_item("bad"), _make_item("good2")]
        eval_input = EvalInput(eval_input_items=items)

        def score_fn(item: EvalInputItem) -> EvalOutputItem:
            if item.id == "bad":
                raise ValueError("Scoring exploded")
            return EvalOutputItem(id=item.id, score=1.0, reasoning={"ok": True})

        result = run_evaluator_loop(eval_input, score_fn, "TestBench")

        assert len(result.eval_output_items) == 3
        # Good items scored 1.0, bad item scored 0.0
        scores_by_id = {item.id: item.score for item in result.eval_output_items}
        assert scores_by_id["good"] == 1.0
        assert scores_by_id["bad"] == 0.0
        assert scores_by_id["good2"] == 1.0
        # Average: (1.0 + 0.0 + 1.0) / 3
        assert abs(result.average_score - 2.0 / 3.0) < 1e-9

        # Error item has error in reasoning
        bad_item = next(i for i in result.eval_output_items if i.id == "bad")
        assert "error" in bad_item.reasoning
        assert "Scoring exploded" in bad_item.reasoning["error"]

    def test_all_errors(self):
        """When all items fail, average is 0.0."""
        items = [_make_item("a"), _make_item("b")]
        eval_input = EvalInput(eval_input_items=items)

        def score_fn(item: EvalInputItem) -> EvalOutputItem:
            raise RuntimeError("boom")

        result = run_evaluator_loop(eval_input, score_fn, "TestBench")

        assert result.average_score == 0.0
        assert all(item.score == 0.0 for item in result.eval_output_items)

    def test_preserves_item_ids(self):
        """Output items have the same IDs as input items, in order."""
        items = [_make_item("x"), _make_item("y"), _make_item("z")]
        eval_input = EvalInput(eval_input_items=items)

        def score_fn(item: EvalInputItem) -> EvalOutputItem:
            return EvalOutputItem(id=item.id, score=0.5, reasoning={})

        result = run_evaluator_loop(eval_input, score_fn, "TestBench")

        assert [item.id for item in result.eval_output_items] == ["x", "y", "z"]

    def test_single_item(self):
        """Single item => average equals that item's score."""
        eval_input = EvalInput(eval_input_items=[_make_item("only")])

        def score_fn(item: EvalInputItem) -> EvalOutputItem:
            return EvalOutputItem(id=item.id, score=0.75, reasoning={"f1": 0.75})

        result = run_evaluator_loop(eval_input, score_fn, "TestBench")

        assert result.average_score == 0.75
        assert len(result.eval_output_items) == 1
