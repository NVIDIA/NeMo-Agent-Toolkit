# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BFCL evaluator — no LLM required.

Uses pre-constructed model outputs to validate that ast_checker scoring works
correctly through the evaluator.
"""

import json
import os

import pytest

from nat.data_models.evaluator import EvalInputItem

try:
    from bfcl.eval_checker.ast_eval.ast_checker import ast_checker
    _HAS_BFCL = True
except ImportError:
    _HAS_BFCL = False

pytestmark = pytest.mark.skipif(not _HAS_BFCL, reason="bfcl not installed")


def _make_simple_entry() -> dict:
    """BFCL simple test: calculate_triangle_area(base=10, height=5)."""
    return {
        "id": "simple_0",
        "question": [[{"role": "user", "content": "Find area of triangle with base 10, height 5"}]],
        "function": [{
            "name": "calculate_triangle_area",
            "description": "Calculate the area of a triangle.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "base": {"type": "integer", "description": "The base"},
                    "height": {"type": "integer", "description": "The height"},
                    "unit": {"type": "string", "description": "Unit of measure"},
                },
                "required": ["base", "height"],
            },
        }],
    }


def _make_simple_answer() -> dict:
    return {
        "id": "simple_0",
        "ground_truth": [{
            "calculate_triangle_area": {
                "base": [10],
                "height": [5],
                "unit": ["units", ""],
            },
        }],
    }


def _make_eval_item(entry: dict, answer: dict, model_output: str) -> EvalInputItem:
    return EvalInputItem(
        id=entry["id"],
        input_obj=json.dumps(entry),
        expected_output_obj=json.dumps(answer),
        output_obj=model_output,
        full_dataset_entry=entry,
    )


class TestBFCLEvaluator:

    def test_correct_ast_output_scores_1(self):
        """Model outputs correct function call text → score 1.0."""
        from nat.plugins.benchmarks.bfcl.evaluator import _evaluate_single

        entry = _make_simple_entry()
        answer = _make_simple_answer()
        model_output = "calculate_triangle_area(base=10, height=5)"

        item = _make_eval_item(entry, answer, model_output)
        result = _evaluate_single(item, "simple", "Python")

        assert result.score == 1.0
        assert result.reasoning["valid"] is True

    def test_correct_with_optional_param(self):
        """Model includes optional param with valid value → score 1.0."""
        from nat.plugins.benchmarks.bfcl.evaluator import _evaluate_single

        entry = _make_simple_entry()
        answer = _make_simple_answer()
        model_output = 'calculate_triangle_area(base=10, height=5, unit="units")'

        item = _make_eval_item(entry, answer, model_output)
        result = _evaluate_single(item, "simple", "Python")

        assert result.score == 1.0

    def test_wrong_param_value_scores_0(self):
        """Wrong parameter value → score 0.0."""
        from nat.plugins.benchmarks.bfcl.evaluator import _evaluate_single

        entry = _make_simple_entry()
        answer = _make_simple_answer()
        model_output = "calculate_triangle_area(base=99, height=5)"

        item = _make_eval_item(entry, answer, model_output)
        result = _evaluate_single(item, "simple", "Python")

        assert result.score == 0.0
        assert result.reasoning["valid"] is False

    def test_invalid_syntax_scores_0(self):
        """Garbage output that can't be AST parsed → score 0.0."""
        from nat.plugins.benchmarks.bfcl.evaluator import _evaluate_single

        entry = _make_simple_entry()
        answer = _make_simple_answer()
        model_output = "I cannot do that, sorry."

        item = _make_eval_item(entry, answer, model_output)
        result = _evaluate_single(item, "simple", "Python")

        assert result.score == 0.0

    def test_none_output_scores_0(self):
        """Null output → score 0.0."""
        from nat.plugins.benchmarks.bfcl.evaluator import _evaluate_single

        entry = _make_simple_entry()
        answer = _make_simple_answer()

        item = _make_eval_item(entry, answer, None)
        result = _evaluate_single(item, "simple", "Python")

        assert result.score == 0.0
        assert "error" in result.reasoning


class TestBFCLOutputFormatting:
    """Test that workflow output formats are compatible with the evaluator."""

    def test_fc_workflow_format_parseable(self):
        """FC workflow output format: [func(param=val)] is AST-parseable."""
        from nat.plugins.benchmarks.bfcl.evaluator import _evaluate_single

        entry = _make_simple_entry()
        answer = _make_simple_answer()
        # This is the format our FC workflow produces
        model_output = "[calculate_triangle_area(base=10, height=5)]"

        item = _make_eval_item(entry, answer, model_output)
        result = _evaluate_single(item, "simple", "Python")

        assert result.score == 1.0

    def test_ast_workflow_format_parseable(self):
        """AST workflow output: raw text like func(param=val) is parseable."""
        from nat.plugins.benchmarks.bfcl.evaluator import _evaluate_single

        entry = _make_simple_entry()
        answer = _make_simple_answer()
        model_output = "calculate_triangle_area(base=10, height=5)"

        item = _make_eval_item(entry, answer, model_output)
        result = _evaluate_single(item, "simple", "Python")

        assert result.score == 1.0
