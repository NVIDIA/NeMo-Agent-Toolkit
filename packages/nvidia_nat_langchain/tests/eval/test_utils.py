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
"""Tests for shared conversion utilities (utils.py)."""

import pytest
from langsmith.schemas import Example
from langsmith.schemas import Run

from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.plugins.langchain.eval.utils import eval_input_item_to_openevals_kwargs
from nat.plugins.langchain.eval.utils import eval_input_item_to_run_and_example
from nat.plugins.langchain.eval.utils import langsmith_result_to_eval_output_item


@pytest.fixture
def sample_item():
    return EvalInputItem(
        id="test_1",
        input_obj="What is AI?",
        expected_output_obj="Artificial Intelligence",
        output_obj="AI stands for Artificial Intelligence",
        trajectory=[],
        expected_trajectory=[],
        full_dataset_entry={},
    )


# --------------------------------------------------------------------------- #
# eval_input_item_to_openevals_kwargs
# --------------------------------------------------------------------------- #


def test_openevals_kwargs_maps_fields(sample_item):
    kwargs = eval_input_item_to_openevals_kwargs(sample_item)

    assert kwargs["inputs"] == "What is AI?"
    assert kwargs["outputs"] == "AI stands for Artificial Intelligence"
    assert kwargs["reference_outputs"] == "Artificial Intelligence"


def test_openevals_kwargs_handles_none_expected():
    item = EvalInputItem(
        id="test_none",
        input_obj="question",
        expected_output_obj=None,
        output_obj="answer",
        trajectory=[],
        expected_trajectory=[],
        full_dataset_entry={},
    )
    kwargs = eval_input_item_to_openevals_kwargs(item)

    assert kwargs["inputs"] == "question"
    assert kwargs["outputs"] == "answer"
    assert kwargs["reference_outputs"] is None


# --------------------------------------------------------------------------- #
# eval_input_item_to_run_and_example
# --------------------------------------------------------------------------- #


def test_run_and_example_types(sample_item):
    run, example = eval_input_item_to_run_and_example(sample_item)

    assert isinstance(run, Run)
    assert isinstance(example, Example)


def test_run_contains_correct_data(sample_item):
    run, _ = eval_input_item_to_run_and_example(sample_item)

    assert run.inputs == {"input": "What is AI?"}
    assert run.outputs == {"output": "AI stands for Artificial Intelligence"}
    assert run.run_type == "chain"


def test_example_contains_correct_data(sample_item):
    _, example = eval_input_item_to_run_and_example(sample_item)

    assert example.inputs == {"input": "What is AI?"}
    assert example.outputs == {"output": "Artificial Intelligence"}


# --------------------------------------------------------------------------- #
# langsmith_result_to_eval_output_item
# --------------------------------------------------------------------------- #


def test_dict_result_conversion():
    result = {"key": "accuracy", "score": 0.95, "comment": "Mostly correct", "metadata": None}
    output = langsmith_result_to_eval_output_item("item_1", result)

    assert isinstance(output, EvalOutputItem)
    assert output.id == "item_1"
    assert output.score == 0.95
    assert output.reasoning["key"] == "accuracy"
    assert output.reasoning["comment"] == "Mostly correct"


def test_dict_result_with_bool_score():
    result = {"key": "exact_match", "score": True, "comment": None}
    output = langsmith_result_to_eval_output_item("item_2", result)

    assert output.score is True


def test_dict_result_with_metadata():
    result = {"key": "custom", "score": 0.5, "comment": "OK", "metadata": {"model": "gpt-4"}}
    output = langsmith_result_to_eval_output_item("item_3", result)

    assert output.reasoning["metadata"] == {"model": "gpt-4"}


def test_unexpected_result_type():
    output = langsmith_result_to_eval_output_item("item_4", 42)

    assert output.score == 0.0
    assert "Unexpected result type" in output.reasoning["error"]


def test_evaluation_result_object():
    """Test conversion of a langsmith EvaluationResult object."""
    from langsmith.evaluation.evaluator import EvaluationResult

    result = EvaluationResult(key="test_eval", score=0.8, comment="Good result")
    output = langsmith_result_to_eval_output_item("item_5", result)

    assert output.id == "item_5"
    assert output.score == 0.8
    assert output.reasoning["key"] == "test_eval"
    assert output.reasoning["comment"] == "Good result"
