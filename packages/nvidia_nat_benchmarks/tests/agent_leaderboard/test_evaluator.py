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
"""Tests for the Agent Leaderboard TSQ evaluator — no LLM required."""

import json

import pytest

from nat.data_models.evaluator import EvalInputItem
from nat.plugins.benchmarks.agent_leaderboard.evaluator import _evaluate_single, _normalize_tool_name


def _make_entry(expected_tools: list[str], predicted_tools: list[str]) -> tuple[EvalInputItem, list]:
    """Create a test item with expected and predicted tool calls."""
    expected_calls = [{"tool": t, "parameters": {}} for t in expected_tools]
    predicted_calls = [{"tool": t, "parameters": {}} for t in predicted_tools]

    entry = {
        "id": "test_001",
        "question": "Do something",
        "expected_tool_calls": expected_calls,
        "available_tools": [],
    }

    item = EvalInputItem(
        id="test_001",
        input_obj=json.dumps(entry),
        expected_output_obj=json.dumps(expected_calls),
        output_obj=json.dumps(predicted_calls),
        full_dataset_entry=entry,
    )
    return item


class TestNormalizeToolName:

    def test_basic(self):
        assert _normalize_tool_name("get_account_balance") == "getaccountbalance"

    def test_with_prefix(self):
        assert _normalize_tool_name("banking_tools__get_account_balance") == "getaccountbalance"

    def test_case_insensitive(self):
        assert _normalize_tool_name("GetAccountBalance") == "getaccountbalance"

    def test_empty(self):
        assert _normalize_tool_name("") == ""


class TestTSQEvaluator:

    def test_perfect_match(self):
        """All predicted tools match expected → F1 = 1.0."""
        item = _make_entry(
            expected_tools=["get_balance", "transfer_funds"],
            predicted_tools=["get_balance", "transfer_funds"],
        )
        result = _evaluate_single(item, tool_weight=1.0, parameter_weight=0.0)

        assert result.score == 1.0
        assert result.reasoning["tool_selection_f1"] == 1.0

    def test_partial_match(self):
        """Only some tools match → 0 < F1 < 1."""
        item = _make_entry(
            expected_tools=["get_balance", "transfer_funds", "get_history"],
            predicted_tools=["get_balance", "check_loan"],
        )
        result = _evaluate_single(item, tool_weight=1.0, parameter_weight=0.0)

        assert 0.0 < result.score < 1.0

    def test_no_match(self):
        """No predicted tools match expected → F1 = 0."""
        item = _make_entry(
            expected_tools=["get_balance"],
            predicted_tools=["schedule_appointment"],
        )
        result = _evaluate_single(item, tool_weight=1.0, parameter_weight=0.0)

        assert result.score == 0.0

    def test_no_predictions(self):
        """No predictions when tools expected → score 0."""
        item = _make_entry(
            expected_tools=["get_balance"],
            predicted_tools=[],
        )
        result = _evaluate_single(item, tool_weight=1.0, parameter_weight=0.0)

        assert result.score == 0.0

    def test_no_expected_no_predicted(self):
        """No expected and no predicted → perfect score."""
        item = _make_entry(expected_tools=[], predicted_tools=[])
        result = _evaluate_single(item, tool_weight=1.0, parameter_weight=0.0)

        assert result.score == 1.0

    def test_extra_predictions_reduce_precision(self):
        """Extra predicted tools reduce precision → lower F1."""
        item_exact = _make_entry(
            expected_tools=["get_balance"],
            predicted_tools=["get_balance"],
        )
        item_extra = _make_entry(
            expected_tools=["get_balance"],
            predicted_tools=["get_balance", "extra_tool_1", "extra_tool_2"],
        )

        result_exact = _evaluate_single(item_exact, 1.0, 0.0)
        result_extra = _evaluate_single(item_extra, 1.0, 0.0)

        assert result_exact.score > result_extra.score

    def test_none_output(self):
        """None output → score 0."""
        entry = {"expected_tool_calls": [{"tool": "t", "parameters": {}}]}
        item = EvalInputItem(
            id="test", input_obj="{}", expected_output_obj="[]",
            output_obj=None, full_dataset_entry=entry,
        )
        result = _evaluate_single(item, 1.0, 0.0)
        assert result.score == 0.0

    def test_name_normalization_across_formats(self):
        """Tool names with different formats should still match."""
        item = _make_entry(
            expected_tools=["get_account_balance"],
            predicted_tools=["GetAccountBalance"],
        )
        result = _evaluate_single(item, 1.0, 0.0)
        assert result.score == 1.0


class TestTSQDatasetFormat:

    def test_handles_full_entry_as_string(self):
        """full_dataset_entry can be a JSON string."""
        entry = {"expected_tool_calls": [{"tool": "get_balance", "parameters": {}}]}
        item = EvalInputItem(
            id="test", input_obj="{}",
            expected_output_obj="[]",
            output_obj=json.dumps([{"tool": "get_balance", "parameters": {}}]),
            full_dataset_entry=json.dumps(entry),
        )
        result = _evaluate_single(item, 1.0, 0.0)
        assert result.score == 1.0
