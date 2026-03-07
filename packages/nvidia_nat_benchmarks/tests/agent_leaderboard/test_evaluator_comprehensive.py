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
"""Comprehensive tests for Agent Leaderboard TSQ evaluator."""

import json

import pytest

from nat.data_models.evaluator import EvalInputItem
from nat.plugins.benchmarks.agent_leaderboard.evaluator import _evaluate_single


def _make_item(expected_tools, predicted_tools, via_input_obj=False):
    """Create a test item. If via_input_obj, put expected_tool_calls in input_obj instead of full_dataset_entry."""
    expected_calls = [{"tool": t, "parameters": {}} for t in expected_tools]
    predicted_calls = [{"tool": t, "parameters": {}} for t in predicted_tools]

    entry = {
        "id": "test_001",
        "question": json.dumps({"expected_tool_calls": expected_calls}),
        "expected_tool_calls": expected_calls if not via_input_obj else [],
    }

    return EvalInputItem(
        id="test_001",
        input_obj=json.dumps({"expected_tool_calls": expected_calls}) if via_input_obj else "{}",
        expected_output_obj=json.dumps(expected_calls),
        output_obj=json.dumps(predicted_calls),
        full_dataset_entry=entry,
    )


class TestWeightedScoring:

    def test_tool_weight_only(self):
        """tool_weight=1.0, parameter_weight=0.0 → score = tool_f1."""
        item = _make_item(["tool_a"], ["tool_a"])
        result = _evaluate_single(item, tool_weight=1.0, parameter_weight=0.0)
        assert result.score == 1.0

    def test_parameter_weight_only(self):
        """tool_weight=0.0, parameter_weight=1.0 → score = param_accuracy (placeholder=1.0)."""
        item = _make_item(["tool_a"], ["wrong_tool"])
        result = _evaluate_single(item, tool_weight=0.0, parameter_weight=1.0)
        assert result.score == 1.0  # param_accuracy placeholder is always 1.0

    def test_mixed_weights(self):
        """50/50 weighting."""
        item = _make_item(["tool_a"], ["tool_a"])
        result = _evaluate_single(item, tool_weight=0.5, parameter_weight=0.5)
        # tool_f1=1.0, param=1.0 → 0.5*1.0 + 0.5*1.0 = 1.0
        assert result.score == 1.0

    def test_mixed_weights_partial_match(self):
        item = _make_item(["tool_a", "tool_b"], ["tool_a", "tool_c"])
        result = _evaluate_single(item, tool_weight=0.6, parameter_weight=0.4)
        # tool_f1: precision=0.5, recall=0.5, f1=0.5
        # param=1.0 (placeholder)
        # score = 0.6*0.5 + 0.4*1.0 = 0.7
        assert result.score == pytest.approx(0.7)


class TestInputObjFallback:
    """Verify expected_tool_calls are found via input_obj when not in full_dataset_entry."""

    def test_reads_from_input_obj(self):
        item = _make_item(["tool_a"], ["tool_a"], via_input_obj=True)
        result = _evaluate_single(item, tool_weight=1.0, parameter_weight=0.0)
        assert result.score == 1.0
        assert result.reasoning["expected_tools"] == ["toola"]


class TestReasoningFields:
    """Verify all reasoning dict fields are present and correct."""

    def test_all_fields_present(self):
        item = _make_item(["get_balance", "transfer"], ["get_balance"])
        result = _evaluate_single(item, tool_weight=1.0, parameter_weight=0.0)
        r = result.reasoning

        assert "tool_selection_f1" in r
        assert "parameter_accuracy" in r
        assert "predicted_tools" in r
        assert "expected_tools" in r
        assert "num_predicted" in r
        assert "num_expected" in r
        assert r["num_predicted"] == 1
        assert r["num_expected"] == 2
        assert sorted(r["predicted_tools"]) == ["getbalance"]
        assert sorted(r["expected_tools"]) == ["getbalance", "transfer"]

    def test_f1_precision_recall_math(self):
        """2 expected, 3 predicted, 1 overlap → precision=1/3, recall=1/2, F1=2/5."""
        item = _make_item(["a", "b"], ["a", "c", "d"])
        result = _evaluate_single(item, 1.0, 0.0)
        # precision = 1/3, recall = 1/2, f1 = 2*(1/3)*(1/2)/((1/3)+(1/2)) = 2/5 = 0.4
        assert result.reasoning["tool_selection_f1"] == pytest.approx(0.4)


class TestEdgeCases:

    def test_malformed_json_output(self):
        entry = {"expected_tool_calls": [{"tool": "t", "parameters": {}}]}
        item = EvalInputItem(
            id="bad", input_obj="{}", expected_output_obj="[]",
            output_obj="not json", full_dataset_entry=entry,
        )
        result = _evaluate_single(item, 1.0, 0.0)
        assert result.score == 0.0

    def test_output_is_dict_not_list(self):
        entry = {"expected_tool_calls": [{"tool": "t", "parameters": {}}]}
        item = EvalInputItem(
            id="bad", input_obj="{}", expected_output_obj="[]",
            output_obj='{"tool": "t"}', full_dataset_entry=entry,
        )
        result = _evaluate_single(item, 1.0, 0.0)
        # dict is not a list → predicted = [] → score 0
        assert result.score == 0.0

    def test_duplicate_predicted_tools(self):
        """Duplicate predictions shouldn't inflate precision (set-based comparison)."""
        item = _make_item(["tool_a"], ["tool_a", "tool_a", "tool_a"])
        result = _evaluate_single(item, 1.0, 0.0)
        assert result.score == 1.0  # Set-based: {tool_a} matches {tool_a}
