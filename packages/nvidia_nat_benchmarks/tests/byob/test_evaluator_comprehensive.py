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
"""Comprehensive tests for BYOB evaluator — covers score_field, metadata, error handling."""

import json

import pytest

try:
    from nemo_evaluator.contrib.byob.decorators import ScorerInput
    from nemo_evaluator.contrib.byob.scorers import exact_match, contains, f1_token
    _HAS_BYOB = True
except ImportError:
    _HAS_BYOB = False

pytestmark = pytest.mark.skipif(not _HAS_BYOB, reason="nemo_evaluator BYOB not installed")

from nat.data_models.evaluator import EvalInputItem
from nat.plugins.benchmarks.byob.evaluator import _evaluate_single


class TestScoreFieldSelection:
    """Verify score_field selects the correct key from scorer output."""

    def test_default_correct_field(self):
        item = EvalInputItem(
            id="0", input_obj="{}", expected_output_obj="hello",
            output_obj="hello", full_dataset_entry={"target": "hello"},
        )
        result = _evaluate_single(item, exact_match, "target", "correct", {})
        assert result.score == 1.0

    def test_f1_score_field(self):
        item = EvalInputItem(
            id="0", input_obj="{}", expected_output_obj="the quick brown fox",
            output_obj="the quick brown dog",
            full_dataset_entry={"target": "the quick brown fox"},
        )
        result = _evaluate_single(item, f1_token, "target", "f1", {})
        assert 0.0 < result.score < 1.0
        assert "f1" in result.reasoning
        assert "precision" in result.reasoning
        assert "recall" in result.reasoning

    def test_precision_score_field(self):
        item = EvalInputItem(
            id="0", input_obj="{}", expected_output_obj="a b c",
            output_obj="a b c d e",
            full_dataset_entry={"target": "a b c"},
        )
        result = _evaluate_single(item, f1_token, "target", "precision", {})
        # precision = 3/5 = 0.6
        assert result.score == pytest.approx(0.6)

    def test_missing_score_field_returns_0(self):
        item = EvalInputItem(
            id="0", input_obj="{}", expected_output_obj="hello",
            output_obj="hello", full_dataset_entry={"target": "hello"},
        )
        result = _evaluate_single(item, exact_match, "target", "nonexistent_field", {})
        assert result.score == 0.0

    def test_boolean_score_converted_to_float(self):
        item = EvalInputItem(
            id="0", input_obj="{}", expected_output_obj="yes",
            output_obj="yes", full_dataset_entry={"target": "yes"},
        )
        result = _evaluate_single(item, exact_match, "target", "correct", {})
        assert isinstance(result.score, float)
        assert result.score == 1.0


class TestMetadataPassthrough:
    """Verify full_dataset_entry is passed as metadata to the scorer."""

    def test_metadata_available_to_scorer(self):
        """Custom scorer that reads metadata."""
        def metadata_checker(sample: ScorerInput) -> dict:
            has_extra = "extra_field" in sample.metadata
            return {"correct": has_extra}

        item = EvalInputItem(
            id="0", input_obj="{}", expected_output_obj="x",
            output_obj="x",
            full_dataset_entry={"target": "x", "extra_field": "present"},
        )
        result = _evaluate_single(item, metadata_checker, "target", "correct", {})
        assert result.score == 1.0

    def test_string_full_entry_parsed_to_dict(self):
        """full_dataset_entry as JSON string should be parsed."""
        item = EvalInputItem(
            id="0", input_obj="{}", expected_output_obj="hello",
            output_obj="hello",
            full_dataset_entry=json.dumps({"target": "hello", "category": "test"}),
        )
        result = _evaluate_single(item, exact_match, "target", "correct", {})
        assert result.score == 1.0


class TestExtraConfig:
    """Verify extra_config is passed to the scorer."""

    def test_config_available_to_scorer(self):
        def config_scorer(sample: ScorerInput) -> dict:
            threshold = sample.config.get("threshold", 0.5)
            return {"correct": len(sample.response) > threshold}

        item = EvalInputItem(
            id="0", input_obj="{}", expected_output_obj="x",
            output_obj="hello world",
            full_dataset_entry={"target": "x"},
        )
        result = _evaluate_single(item, config_scorer, "target", "correct", {"threshold": 3})
        assert result.score == 1.0


class TestScorerErrors:
    """Verify error handling when the scorer raises exceptions."""

    def test_scorer_exception_returns_0(self):
        def bad_scorer(sample: ScorerInput) -> dict:
            raise ValueError("Scorer crashed!")

        item = EvalInputItem(
            id="0", input_obj="{}", expected_output_obj="x",
            output_obj="hello", full_dataset_entry={"target": "x"},
        )
        result = _evaluate_single(item, bad_scorer, "target", "correct", {})
        assert result.score == 0.0
        assert "Scorer failed" in result.reasoning["error"]


class TestTargetParsing:
    """Verify target values are correctly parsed from expected_output_obj."""

    def test_string_target(self):
        item = EvalInputItem(
            id="0", input_obj="{}", expected_output_obj="Paris",
            output_obj="Paris", full_dataset_entry={"target": "Paris"},
        )
        result = _evaluate_single(item, exact_match, "target", "correct", {})
        assert result.score == 1.0

    def test_json_target_parsed(self):
        """JSON-encoded target should be parsed."""
        def list_scorer(sample: ScorerInput) -> dict:
            return {"correct": isinstance(sample.target, list)}

        item = EvalInputItem(
            id="0", input_obj="{}", expected_output_obj='["a", "b"]',
            output_obj="x", full_dataset_entry={"target": ["a", "b"]},
        )
        result = _evaluate_single(item, list_scorer, "target", "correct", {})
        assert result.score == 1.0
