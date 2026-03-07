# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BYOB evaluator — no LLM required."""

import json
import os
import sys

import pytest

try:
    from nemo_evaluator.contrib.byob.decorators import ScorerInput
    from nemo_evaluator.contrib.byob.scorers import exact_match, contains, f1_token
    _HAS_BYOB = True
except ImportError:
    _HAS_BYOB = False

pytestmark = pytest.mark.skipif(not _HAS_BYOB, reason="nemo_evaluator BYOB not installed")


from nat.data_models.evaluator import EvalInputItem


class TestBYOBEvaluator:

    def test_exact_match_correct(self):
        """Exact match scorer: correct answer scores 1.0."""
        from nat.plugins.benchmarks.byob.evaluator import _evaluate_single

        item = EvalInputItem(
            id="0", input_obj='{"question": "2+2?"}',
            expected_output_obj="4", output_obj="4",
            full_dataset_entry={"question": "2+2?", "target": "4"},
        )
        result = _evaluate_single(item, exact_match, "target", "correct", {})

        assert result.score == 1.0
        assert result.reasoning["correct"] is True

    def test_exact_match_wrong(self):
        """Wrong answer scores 0.0."""
        from nat.plugins.benchmarks.byob.evaluator import _evaluate_single

        item = EvalInputItem(
            id="1", input_obj='{"question": "2+2?"}',
            expected_output_obj="4", output_obj="5",
            full_dataset_entry={"question": "2+2?", "target": "4"},
        )
        result = _evaluate_single(item, exact_match, "target", "correct", {})

        assert result.score == 0.0
        assert result.reasoning["correct"] is False

    def test_contains_scorer(self):
        """Contains scorer: target substring in response."""
        from nat.plugins.benchmarks.byob.evaluator import _evaluate_single

        item = EvalInputItem(
            id="2", input_obj='{}',
            expected_output_obj="Paris", output_obj="The capital of France is Paris.",
            full_dataset_entry={"target": "Paris"},
        )
        result = _evaluate_single(item, contains, "target", "correct", {})

        assert result.score == 1.0

    def test_f1_token_scorer(self):
        """F1 token scorer returns float score."""
        from nat.plugins.benchmarks.byob.evaluator import _evaluate_single

        item = EvalInputItem(
            id="3", input_obj='{}',
            expected_output_obj="the quick brown fox",
            output_obj="the quick brown dog",
            full_dataset_entry={"target": "the quick brown fox"},
        )
        result = _evaluate_single(item, f1_token, "target", "f1", {})

        assert 0.0 < result.score < 1.0  # Partial match
        assert "f1" in result.reasoning

    def test_none_output_scores_0(self):
        """Null output scores 0.0."""
        from nat.plugins.benchmarks.byob.evaluator import _evaluate_single

        item = EvalInputItem(
            id="4", input_obj='{}',
            expected_output_obj="answer", output_obj=None,
            full_dataset_entry={},
        )
        result = _evaluate_single(item, exact_match, "target", "correct", {})

        assert result.score == 0.0


class TestBYOBBenchmarkImport:

    def test_import_sample_benchmark(self):
        """Can import and use the sample benchmark definition."""
        from nemo_evaluator.contrib.byob.eval_logic import import_benchmark

        fixture_path = os.path.join(
            os.path.dirname(__file__), "fixtures", "sample_benchmark.py"
        )
        bench = import_benchmark(fixture_path, "test_exact_match")

        assert bench.name == "test-exact-match"
        assert bench.target_field == "target"
        assert bench.scorer_fn is not None

        # Test the scorer
        scorer_input = ScorerInput(
            response="4", target="4", metadata={},
        )
        result = bench.scorer_fn(scorer_input)
        assert result["correct"] is True

    def test_import_and_load_dataset(self):
        """Can load dataset from sample benchmark."""
        from nat.plugins.benchmarks.byob.dataset import load_byob_dataset

        fixture_path = os.path.join(
            os.path.dirname(__file__), "fixtures", "sample_benchmark.py"
        )
        df = load_byob_dataset(
            file_path="ignored",
            benchmark_module=fixture_path,
            benchmark_name="test_exact_match",
        )

        assert len(df) == 3
        assert set(df.columns) >= {"id", "question", "answer"}
        assert df.iloc[0]["answer"] == "4"
