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
"""Comprehensive evaluator tests covering all BFCL scoring paths."""

import json

import pytest

try:
    from bfcl.eval_checker.ast_eval.ast_checker import ast_checker
    _HAS_BFCL = True
except ImportError:
    _HAS_BFCL = False

pytestmark = pytest.mark.skipif(not _HAS_BFCL, reason="bfcl not installed")

from nat.data_models.evaluator import EvalInputItem
from nat.plugins.benchmarks.bfcl.evaluator import (
    _evaluate_single,
    _extract_function_call,
    _try_convert_json_to_call,
    _extract_calls_from_code,
)


def _make_item(entry, answer, output):
    return EvalInputItem(
        id=entry["id"], input_obj=json.dumps(entry),
        expected_output_obj=json.dumps(answer),
        output_obj=output, full_dataset_entry=entry,
    )


def _simple_entry():
    return {
        "id": "simple_0",
        "question": [[{"role": "user", "content": "Calculate area"}]],
        "function": [{
            "name": "calc_area",
            "description": "Calculate area",
            "parameters": {
                "type": "dict",
                "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}},
                "required": ["base", "height"],
            },
        }],
    }


def _simple_answer():
    return {"id": "simple_0", "ground_truth": [{"calc_area": {"base": [10], "height": [5]}}]}


class TestIrrelevanceMode:
    """Tests for irrelevance test category — model should NOT produce function calls."""

    def test_no_function_call_scores_1(self):
        entry = _simple_entry()
        answer = {"id": "simple_0", "ground_truth": []}
        item = _make_item(entry, answer, "I cannot help with that request.")
        result = _evaluate_single(item, "irrelevance", "Python")
        assert result.score == 1.0

    def test_valid_function_call_scores_0(self):
        entry = _simple_entry()
        answer = {"id": "simple_0", "ground_truth": []}
        item = _make_item(entry, answer, "calc_area(base=10, height=5)")
        result = _evaluate_single(item, "irrelevance", "Python")
        assert result.score == 0.0

    def test_garbage_output_scores_1(self):
        """Unparseable output = no function call = correct for irrelevance."""
        entry = _simple_entry()
        answer = {"id": "simple_0", "ground_truth": []}
        item = _make_item(entry, answer, "Sorry, I don't know how to do that.")
        result = _evaluate_single(item, "irrelevance", "Python")
        assert result.score == 1.0


class TestRelevanceMode:
    """Tests for relevance test category — model SHOULD produce a function call."""

    def test_valid_function_call_scores_1(self):
        entry = _simple_entry()
        answer = {"id": "simple_0", "ground_truth": []}
        item = _make_item(entry, answer, "calc_area(base=10, height=5)")
        result = _evaluate_single(item, "live_relevance", "Python")
        assert result.score == 1.0

    def test_no_function_call_scores_0(self):
        entry = _simple_entry()
        answer = {"id": "simple_0", "ground_truth": []}
        item = _make_item(entry, answer, "I'm not sure what you need.")
        result = _evaluate_single(item, "live_relevance", "Python")
        assert result.score == 0.0


class TestParallelCategory:
    """Tests for parallel test category — multiple function calls in one response."""

    def test_parallel_calls_scored(self):
        entry = {
            "id": "parallel_0",
            "question": [[{"role": "user", "content": "Get area and perimeter"}]],
            "function": [
                {"name": "calc_area", "description": "Area", "parameters": {
                    "type": "dict", "properties": {"x": {"type": "integer"}}, "required": ["x"]}},
                {"name": "calc_perimeter", "description": "Perimeter", "parameters": {
                    "type": "dict", "properties": {"x": {"type": "integer"}}, "required": ["x"]}},
            ],
        }
        answer = {"id": "parallel_0", "ground_truth": [
            {"calc_area": {"x": [5]}},
            {"calc_perimeter": {"x": [5]}},
        ]}
        item = _make_item(entry, answer, "[calc_area(x=5), calc_perimeter(x=5)]")
        result = _evaluate_single(item, "parallel", "Python")
        assert result.score == 1.0


class TestTryConvertJsonToCall:
    """Tests for JSON tool-call format conversion."""

    def test_converts_json_with_name_and_parameters(self):
        text = '{"name": "func", "parameters": {"x": 10}}'
        assert _try_convert_json_to_call(text) == "func(x=10)"

    def test_converts_json_with_arguments_key(self):
        text = '{"name": "func", "arguments": {"a": 1, "b": 2}}'
        result = _try_convert_json_to_call(text)
        assert "func(" in result
        assert "a=1" in result
        assert "b=2" in result

    def test_returns_none_for_non_tool_json(self):
        assert _try_convert_json_to_call('{"key": "value"}') is None

    def test_returns_none_for_invalid_json(self):
        assert _try_convert_json_to_call("not json at all") is None

    def test_returns_none_for_list(self):
        assert _try_convert_json_to_call("[1, 2, 3]") is None


class TestExtractCallsFromCode:
    """Tests for extracting function calls from Python code blocks."""

    def test_extracts_assigned_call(self):
        code = "import math\nresult = math.hypot(4, 5)\nprint(result)"
        result = _extract_calls_from_code(code)
        assert result == "math.hypot(4, 5)"

    def test_extracts_bare_call(self):
        code = "calculate_area(base=10, height=5)"
        result = _extract_calls_from_code(code)
        assert result == "calculate_area(base=10, height=5)"

    def test_skips_imports_and_prints(self):
        code = "import math\nfrom os import path\nresult = func(x=1)\nprint(result)"
        result = _extract_calls_from_code(code)
        assert result == "func(x=1)"

    def test_returns_none_for_no_calls(self):
        code = "import math\nprint('hello')"
        result = _extract_calls_from_code(code)
        assert result is None

    def test_multiple_calls(self):
        code = "a = func_a(x=1)\nb = func_b(y=2)"
        result = _extract_calls_from_code(code)
        assert "func_a(x=1)" in result
        assert "func_b(y=2)" in result


class TestExtractFunctionCallEdgeCases:
    """Additional edge cases for the extraction pipeline."""

    def test_json_code_block(self):
        raw = '```json\n{"name": "func", "parameters": {"x": 10}}\n```'
        assert _extract_function_call(raw) == "func(x=10)"

    def test_python_code_with_imports(self):
        raw = "import math\nresult = math.factorial(5)\nprint(result)"
        result = _extract_function_call(raw)
        assert "math.factorial(5)" in result

    def test_tools_prefix_in_code_block(self):
        raw = "```python\ntools.func(x=1)\n```"
        # Code block extraction returns "tools.func(x=1)", then func call extraction strips it
        result = _extract_function_call(raw)
        assert "func(x=1)" in result or "tools.func(x=1)" in result
