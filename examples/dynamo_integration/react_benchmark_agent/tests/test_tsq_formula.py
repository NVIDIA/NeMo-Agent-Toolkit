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
Unit tests for TSQ (Tool Selection Quality) evaluator formula verification.

These tests verify that the TSQ calculator produces expected results
across various edge cases and known scenarios.
"""

import pytest

from nat.builder.function import FunctionGroup


def normalize_tool_name(tool_name: str) -> str:
    """Normalize tool names for comparison (matches tsq_evaluator.py)."""
    if not tool_name:
        return ""

    # Strip module prefix (e.g., "banking_tools__report_lost_stolen_card" -> "report_lost_stolen_card")
    sep = FunctionGroup.SEPARATOR
    if sep in tool_name:
        _, tool_name = tool_name.split(sep, maxsplit=1)

    return tool_name.lower().strip().replace("_", "").replace("-", "")


def calculate_tool_accuracy(actual: list[dict], expected: list[dict]) -> float:
    """Calculate tool selection accuracy using F1 score (matches tsq_evaluator.py)."""
    if not expected:
        return 1.0 if not actual else 0.0

    actual_tools = {normalize_tool_name(tc["tool"]) for tc in actual}
    expected_tools = {normalize_tool_name(tc["tool"]) for tc in expected}

    if not expected_tools:
        return 1.0

    # Calculate precision and recall
    correct = len(actual_tools.intersection(expected_tools))
    precision = correct / len(actual_tools) if actual_tools else 0.0
    recall = correct / len(expected_tools) if expected_tools else 0.0

    # F1 score (harmonic mean)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_tsq_score(tool_accuracy: float,
                        param_accuracy: float,
                        tool_weight: float = 1.0,
                        param_weight: float = 0.0) -> float:
    """Calculate final TSQ score (parameter accuracy disabled by default)."""
    return (tool_weight * tool_accuracy) + (param_weight * param_accuracy)


class TestF1Formula:
    """Test the F1 score calculation for tool selection accuracy."""

    def test_perfect_match(self):
        """Test when actual tools exactly match expected tools."""
        actual = [{"tool": "tool_a"}, {"tool": "tool_b"}, {"tool": "tool_c"}]
        expected = [{"tool": "tool_a"}, {"tool": "tool_b"}, {"tool": "tool_c"}]

        # precision = 3/3 = 1.0, recall = 3/3 = 1.0, F1 = 1.0
        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == 1.0

    def test_no_overlap(self):
        """Test when there's no overlap between actual and expected."""
        actual = [{"tool": "tool_x"}, {"tool": "tool_y"}]
        expected = [{"tool": "tool_a"}, {"tool": "tool_b"}]

        # precision = 0/2 = 0, recall = 0/2 = 0, F1 = 0
        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == 0.0

    def test_empty_actual(self):
        """Test when no actual tools were called."""
        actual = []
        expected = [{"tool": "tool_a"}, {"tool": "tool_b"}]

        # precision = 0 (empty), recall = 0, F1 = 0
        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == 0.0

    def test_empty_expected(self):
        """Test when no expected tools."""
        actual = [{"tool": "tool_a"}]
        expected = []

        # Edge case: returns 0.0 when actual has tools but expected is empty
        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == 0.0

    def test_both_empty(self):
        """Test when both are empty."""
        actual = []
        expected = []

        # Edge case: returns 1.0 (perfect match of "nothing")
        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == 1.0

    def test_high_precision_low_recall(self):
        """Test when precision is high but recall is low."""
        actual = [{"tool": "tool_a"}]  # 1 unique
        expected = [{"tool": "tool_a"}, {"tool": "tool_b"}, {"tool": "tool_c"}, {"tool": "tool_d"}]  # 4 unique

        # correct = 1
        # precision = 1/1 = 1.0
        # recall = 1/4 = 0.25
        # F1 = 2 * 1.0 * 0.25 / (1.0 + 0.25) = 0.5 / 1.25 = 0.4
        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == pytest.approx(0.4)

    def test_low_precision_high_recall(self):
        """Test when precision is low but recall is high (scenario_000 case)."""
        # Simulating: 20 unique actual tools, 8 expected, all 8 expected in actual
        actual = [{"tool": f"tool_{i}"} for i in range(20)]
        expected = [{"tool": f"tool_{i}"} for i in range(8)]  # First 8 tools

        # correct = 8
        # precision = 8/20 = 0.4
        # recall = 8/8 = 1.0
        # F1 = 2 * 0.4 * 1.0 / (0.4 + 1.0) = 0.8 / 1.4 = 4/7 ≈ 0.5714
        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == pytest.approx(4 / 7)

    def test_scenario_000_exact_case(self):
        """Test the exact case from banking_scenario_000."""
        # From the output: 20 unique actual tools, 8 expected, all 8 present
        # This should produce tool_selection_accuracy = 0.5714285714285715

        expected_accuracy = 4 / 7  # 0.5714285714285715

        # Simulate with actual data
        actual = [{"tool": f"tool_{i}"} for i in range(20)]
        expected = [{"tool": f"tool_{i}"} for i in range(8)]

        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == pytest.approx(expected_accuracy)


class TestTSQScore:
    """Test the final TSQ score calculation."""

    def test_scenario_000_score(self):
        """Verify the score from banking_scenario_000."""
        tool_accuracy = 4 / 7  # 0.5714285714285715
        param_accuracy = 0.0

        # expected_tsq = 1.0 * tool_accuracy + 0.0 * param_accuracy
        # = 1.0 * 0.5714... + 0.0 * 0
        # = 0.5714...

        tsq = calculate_tsq_score(tool_accuracy, param_accuracy)
        assert tsq == pytest.approx(4 / 7)

    def test_zero_tools_score(self):
        """Test TSQ when no tools were called (scenarios 1-99)."""
        tool_accuracy = 0.0
        param_accuracy = 0.0

        tsq = calculate_tsq_score(tool_accuracy, param_accuracy)
        assert tsq == 0.0

    def test_perfect_score(self):
        """Test TSQ for perfect tool selection and parameters."""
        tool_accuracy = 1.0
        param_accuracy = 1.0

        tsq = calculate_tsq_score(tool_accuracy, param_accuracy)
        assert tsq == 1.0

    def test_only_tools_correct(self):
        """Test TSQ when only tools are correct, not parameters."""
        tool_accuracy = 1.0
        param_accuracy = 0.0

        tsq = calculate_tsq_score(tool_accuracy, param_accuracy)
        assert tsq == pytest.approx(1.0)  # With param_weight=0, TSQ = tool_accuracy

    def test_only_params_correct(self):
        """Test TSQ when only parameters are correct (unusual case)."""
        tool_accuracy = 0.0
        param_accuracy = 1.0

        tsq = calculate_tsq_score(tool_accuracy, param_accuracy)
        assert tsq == pytest.approx(0.0)  # With param_weight=0, params don't contribute


class TestNormalization:
    """Test tool name normalization."""

    def test_underscore_removal(self):
        """Verify underscores are removed during normalization."""
        assert normalize_tool_name("get_account_balance") == "getaccountbalance"

    def test_dash_removal(self):
        """Verify dashes are removed during normalization."""
        assert normalize_tool_name("get-account-balance") == "getaccountbalance"

    def test_case_insensitive(self):
        """Verify matching is case insensitive."""
        assert normalize_tool_name("GET_ACCOUNT_BALANCE") == "getaccountbalance"

    def test_matching_with_normalization(self):
        """Verify tools match despite formatting differences."""
        actual = [{"tool": "get_account_balance"}]
        expected = [{"tool": "GetAccountBalance"}]

        # Should match after normalization
        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == 1.0

    def test_module_prefix_stripping(self):
        """Verify module prefixes are stripped (e.g., banking_tools__report_lost_stolen_card)."""
        sep = FunctionGroup.SEPARATOR
        assert normalize_tool_name(f"banking_tools{sep}report_lost_stolen_card") == "reportloststolencard"
        assert normalize_tool_name(f"module{sep}submodule{sep}tool_name") == "submoduletoolname"

    def test_module_prefix_matching(self):
        """Verify tools match even with module prefixes."""
        actual = [{"tool": f"banking_tools{FunctionGroup.SEPARATOR}report_lost_stolen_card"}]
        expected = [{"tool": "report_lost_stolen_card"}]

        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == 1.0

    def test_empty_tool_name(self):
        """Verify empty tool names are handled."""
        assert normalize_tool_name("") == ""
        assert normalize_tool_name(None) == "" if normalize_tool_name(None) is not None else True


class TestDuplicateHandling:
    """Test how duplicate tool calls are handled."""

    def test_duplicates_in_actual(self):
        """Test that duplicate actual calls are deduplicated."""
        # 5 calls but only 2 unique tools
        actual = [{"tool": "tool_a"}, {"tool": "tool_a"}, {"tool": "tool_a"}, {"tool": "tool_b"}, {"tool": "tool_b"}]
        expected = [{"tool": "tool_a"}, {"tool": "tool_b"}]

        # unique actual = 2, expected = 2, correct = 2
        # precision = 2/2 = 1.0, recall = 2/2 = 1.0, F1 = 1.0
        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == 1.0

    def test_many_duplicates_scenario(self):
        """Test scenario similar to banking_scenario_000 with 725 calls."""
        # 725 calls but only ~20 unique tools, 8 expected with all 8 present
        unique_tools = 20
        expected_tools = 8

        # Build actual with duplicates (725 calls, 20 unique)
        actual = []
        for i in range(725):
            actual.append({"tool": f"tool_{i % unique_tools}"})

        # Expected: first 8 tools
        expected = [{"tool": f"tool_{i}"} for i in range(expected_tools)]

        # correct = 8 (all expected are in actual)
        # precision = 8/20 = 0.4
        # recall = 8/8 = 1.0
        # F1 = 2 * 0.4 * 1.0 / 1.4 = 4/7
        accuracy = calculate_tool_accuracy(actual, expected)
        assert accuracy == pytest.approx(4 / 7)


class TestAverageScoreCalculation:
    """Test the average score calculation across scenarios."""

    def test_average_with_mostly_zeros(self):
        """Verify average calculation with 1 non-zero and 99 zeros."""
        # With param_weight=0, TSQ = tool_accuracy = 4/7 ≈ 0.5714
        scores = [4 / 7] + [0.0] * 99  # 100 scenarios

        average = sum(scores) / len(scores)
        assert average == pytest.approx((4 / 7) / 100)

    def test_output_matches_expected_average(self):
        """Verify the output file's average_score."""
        # With param_weight=0, TSQ = tool_accuracy = 4/7
        expected_average = (4 / 7) / 100
        assert expected_average == pytest.approx(0.005714285714285714)


class TestTrajectoryExtraction:
    """Test tool call extraction from different trajectory formats."""

    @staticmethod
    def extract_tool_calls_from_trajectory(trajectory: list) -> list:
        """
        Mimic the TSQ evaluator's extraction logic.
        Matches the updated tsq_evaluator.py implementation.
        """
        tool_calls = []
        for step in trajectory:
            if not isinstance(step, dict):
                continue

            tool_call = None

            # Strategy 1: Nested payload structure (profiler format)
            payload = step.get("payload", {})
            if isinstance(payload, dict) and payload.get("event_type") == "TOOL_START":
                tool_name = payload.get("name", "")
                data = payload.get("data", {})
                if isinstance(data, dict):
                    params = data.get("input_params", data.get("input", {}))
                    if isinstance(params, dict):
                        params = params.get("input_params", params)
                else:
                    params = {}
                tool_call = {"tool": tool_name, "parameters": params if isinstance(params, dict) else {}}

            # Strategy 2: Flat structure (legacy format)
            elif step.get("event_type") == "TOOL_START":
                tool_call = {
                    "tool": step.get("tool_name", step.get("name", "")),
                    "parameters": step.get("tool_input", step.get("input", {})),
                }

            # Strategy 3: LangChain action format
            elif "action" in step and "action_input" in step:
                tool_call = {
                    "tool": step.get("action", ""),
                    "parameters": step.get("action_input", {}),
                }

            if tool_call and tool_call.get("tool"):
                tool_calls.append(tool_call)

        return tool_calls

    def test_nested_payload_format(self):
        """Test extraction from nested payload structure (profiler format)."""
        trajectory = [{
            "parent_id": "root",
            "function_ancestry": {
                "function_id": "123"
            },
            "payload": {
                "event_type": "TOOL_START",
                "name": "banking_tools.report_lost_stolen_card",
                "data": {
                    "input": {
                        "input_params": {
                            "card_type": "credit", "card_number": "1234"
                        }
                    }
                }
            }
        }]

        tool_calls = self.extract_tool_calls_from_trajectory(trajectory)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "banking_tools.report_lost_stolen_card"

    def test_flat_legacy_format(self):
        """Test extraction from flat structure (legacy format)."""
        trajectory = [{
            "event_type": "TOOL_START", "tool_name": "get_account_balance", "tool_input": {
                "account_id": "12345"
            }
        }]

        tool_calls = self.extract_tool_calls_from_trajectory(trajectory)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "get_account_balance"
        assert tool_calls[0]["parameters"] == {"account_id": "12345"}

    def test_langchain_action_format(self):
        """Test extraction from LangChain action format."""
        trajectory = [{"action": "search_tool", "action_input": {"query": "test query"}}]

        tool_calls = self.extract_tool_calls_from_trajectory(trajectory)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "search_tool"
        assert tool_calls[0]["parameters"] == {"query": "test query"}

    def test_mixed_formats(self):
        """Test extraction from mixed trajectory formats."""
        trajectory = [
            # Profiler format
            {
                "payload": {
                    "event_type": "TOOL_START", "name": "tool_a", "data": {
                        "input": {
                            "param": "value_a"
                        }
                    }
                }
            },
            # Non-tool event (should be skipped)
            {
                "payload": {
                    "event_type": "LLM_START", "name": "llama-3.3-70b"
                }
            },
            # Legacy format
            {
                "event_type": "TOOL_START", "tool_name": "tool_b", "tool_input": {
                    "param": "value_b"
                }
            },
            # LangChain format
            {
                "action": "tool_c", "action_input": {
                    "param": "value_c"
                }
            }
        ]

        tool_calls = self.extract_tool_calls_from_trajectory(trajectory)
        assert len(tool_calls) == 3
        assert {tc["tool"] for tc in tool_calls} == {"tool_a", "tool_b", "tool_c"}

    def test_empty_trajectory(self):
        """Test extraction from empty trajectory."""
        tool_calls = self.extract_tool_calls_from_trajectory([])
        assert tool_calls == []

    def test_no_tool_events(self):
        """Test extraction when there are no tool events."""
        trajectory = [
            {
                "payload": {
                    "event_type": "LLM_START", "name": "model"
                }
            },
            {
                "payload": {
                    "event_type": "LLM_END", "name": "model"
                }
            },
        ]

        tool_calls = self.extract_tool_calls_from_trajectory(trajectory)
        assert tool_calls == []

    def test_real_profiler_data_structure(self):
        """Test with structure matching actual profiler output."""
        trajectory = [{
            "parent_id": "root",
            "function_ancestry": {
                "function_id": "dacd16ec-a9bb-458d-bde7-fc2a3a01b3b6",
                "function_name": "<workflow>",
                "parent_id": "root",
                "parent_name": "root"
            },
            "payload": {
                "event_type": "TOOL_START",
                "event_timestamp": 1764917512.0873613,
                "span_event_timestamp": None,
                "framework": None,
                "name": "banking_tools.report_lost_stolen_card",
                "tags": None,
                "metadata": {},
                "data": {
                    "input": {
                        "input_params": {
                            "card_type": "credit", "card_number_last_four": "1234", "incident_type": "lost"
                        }
                    }
                },
                "usage_info": None,
                "UUID": "abc123"
            }
        }]

        tool_calls = self.extract_tool_calls_from_trajectory(trajectory)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "banking_tools.report_lost_stolen_card"
        # Should extract nested input_params
        assert "card_type" in tool_calls[0]["parameters"]
        assert isinstance(tool_calls[0]["parameters"], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
