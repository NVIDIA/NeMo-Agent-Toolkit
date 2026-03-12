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

from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutputItem
from nat.plugins.langchain.eval.langsmith_evaluator_adapter import LangSmithEvaluatorAdapter
from nat.plugins.langchain.eval.utils import _extract_field
from nat.plugins.langchain.eval.utils import eval_input_item_to_openevals_kwargs
from nat.plugins.langchain.eval.utils import eval_input_item_to_run_and_example
from nat.plugins.langchain.eval.utils import langsmith_result_to_eval_output_item


@pytest.fixture(name="sample_item")
def fixture_sample_item():
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
# eval_input_item_to_openevals_kwargs -- extra_fields
# --------------------------------------------------------------------------- #


class TestExtraFieldsMapping:
    """Tests for extra_fields on eval_input_item_to_openevals_kwargs."""

    def test_extra_fields_adds_context(self, item_with_context):
        """extra_fields pulls values from full_dataset_entry."""
        kwargs = eval_input_item_to_openevals_kwargs(
            item_with_context,
            extra_fields={"context": "retrieved_context"},
        )
        assert kwargs["context"] == "Doodads are small mechanical gadgets used in workshops."
        assert kwargs["inputs"] == "What is a doodad?"
        assert kwargs["outputs"] == "A doodad is a kitten"
        assert kwargs["reference_outputs"] == "A small gadget"

    def test_extra_fields_multiple_mappings(self, item_with_context):
        """Multiple extra_fields are all included."""
        kwargs = eval_input_item_to_openevals_kwargs(
            item_with_context,
            extra_fields={
                "context": "retrieved_context", "plan": "agent_plan"
            },
        )
        assert kwargs["context"] == "Doodads are small mechanical gadgets used in workshops."
        assert kwargs["plan"] == "Step 1: look it up. Step 2: summarize."

    def test_extra_fields_missing_dataset_key_raises(self, item_with_context):
        """KeyError raised when dataset field doesn't exist."""
        with pytest.raises(KeyError, match="nonexistent_field"):
            eval_input_item_to_openevals_kwargs(
                item_with_context,
                extra_fields={"context": "nonexistent_field"},
            )

    def test_extra_fields_conflicts_with_standard_raises(self, item_with_context):
        """ValueError raised when extra_fields key conflicts with standard params."""
        with pytest.raises(ValueError, match="conflicts with a standard"):
            eval_input_item_to_openevals_kwargs(
                item_with_context,
                extra_fields={"inputs": "retrieved_context"},
            )

    def test_extra_fields_none_is_noop(self, item_with_context):
        """None extra_fields produces standard 3-key dict."""
        kwargs = eval_input_item_to_openevals_kwargs(item_with_context, extra_fields=None)
        assert set(kwargs.keys()) == {"inputs", "outputs", "reference_outputs"}


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
# _extract_field
# --------------------------------------------------------------------------- #


class TestExtractField:
    """Tests for the _extract_field dot-notation helper."""

    def test_flat_field(self):
        assert _extract_field({"score": 0.8}, "score") == 0.8

    def test_nested_field(self):
        data = {"analysis": {"reasoning": "good", "score": 0.9}}
        assert _extract_field(data, "analysis.score") == 0.9

    def test_deeply_nested(self):
        data = {"a": {"b": {"c": 42}}}
        assert _extract_field(data, "a.b.c") == 42

    def test_missing_field_raises_key_error(self):
        with pytest.raises(KeyError, match="nonexistent"):
            _extract_field({"score": 1.0}, "nonexistent")

    def test_non_dict_intermediate_raises_type_error(self):
        with pytest.raises(TypeError, match="non-dict"):
            _extract_field({"analysis": "not_a_dict"}, "analysis.score")


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
    assert "Unexpected result type" in output.error


def test_evaluation_result_object():
    """Test conversion of a langsmith EvaluationResult object."""
    from langsmith.evaluation.evaluator import EvaluationResult

    result = EvaluationResult(key="test_eval", score=0.8, comment="Good result")
    output = langsmith_result_to_eval_output_item("item_5", result)

    assert output.id == "item_5"
    assert output.score == 0.8
    assert output.reasoning["key"] == "test_eval"
    assert output.reasoning["comment"] == "Good result"


# --------------------------------------------------------------------------- #
# langsmith_result_to_eval_output_item -- list handling
# --------------------------------------------------------------------------- #


class TestListResultHandling:
    """Tests for bare list[EvaluatorResult] returns."""

    def test_empty_list_returns_zero(self):
        output = langsmith_result_to_eval_output_item("id_1", [])
        assert output.score == 0.0
        assert "Empty list" in output.error

    def test_single_item_list(self):
        result = [{"key": "k1", "score": 0.8, "comment": "OK"}]
        output = langsmith_result_to_eval_output_item("id_2", result)
        assert output.score == 0.8

    def test_multi_item_list_averages(self):
        result = [
            {
                "key": "k1", "score": 1.0, "comment": "Perfect"
            },
            {
                "key": "k2", "score": 0.0, "comment": "Wrong"
            },
        ]
        output = langsmith_result_to_eval_output_item("id_3", result)
        assert output.score == pytest.approx(0.5)
        assert output.reasoning["aggregated_from"] == 2

    def test_bool_scores_in_list_coerced(self):
        result = [
            {
                "key": "k1", "score": True, "comment": "Yes"
            },
            {
                "key": "k2", "score": False, "comment": "No"
            },
        ]
        output = langsmith_result_to_eval_output_item("id_4", result)
        assert output.score == pytest.approx(0.5)

    def test_list_preserves_per_item_details(self):
        result = [
            {
                "key": "k1", "score": 1.0, "comment": "A"
            },
            {
                "key": "k2", "score": 0.5, "comment": "B"
            },
        ]
        output = langsmith_result_to_eval_output_item("id_5", result)
        assert len(output.reasoning["per_item"]) == 2


# --------------------------------------------------------------------------- #
# langsmith_result_to_eval_output_item -- custom output_schema / score_field
# --------------------------------------------------------------------------- #


class TestCustomSchemaResultParsing:
    """Tests for score_field extraction from custom output_schema results."""

    def test_score_field_flat(self):
        result = {"are_equal": True, "justification": "Same values"}
        output = langsmith_result_to_eval_output_item("id_1", result, score_field="are_equal")
        assert output.score is True
        assert output.reasoning["raw_output"] == result

    def test_score_field_nested(self):
        result = {"analysis": {"confidence": 0.95, "score": 0.8}, "metadata": {}}
        output = langsmith_result_to_eval_output_item("id_2", result, score_field="analysis.score")
        assert output.score == 0.8

    def test_score_field_missing_returns_error(self):
        result = {"justification": "Some text"}
        output = langsmith_result_to_eval_output_item("id_3", result, score_field="nonexistent")
        assert output.score == 0.0
        assert "Failed to extract score_field" in output.error

    def test_score_field_takes_precedence_over_standard_key(self):
        """When score_field is set, custom schema handling is always used."""
        result = {"key": "accuracy", "score": 0.95, "comment": "Good"}
        output = langsmith_result_to_eval_output_item("id_4", result, score_field="score")
        assert output.score == 0.95
        assert output.reasoning["raw_output"] == result

    async def test_adapter_uses_score_field(self):
        """Adapter passes score_field through to result converter."""

        def custom_schema_evaluator(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"is_correct": True, "explanation": "Matches reference"}

        evaluator = LangSmithEvaluatorAdapter(
            evaluator=custom_schema_evaluator,
            convention="openevals_function",
            max_concurrency=1,
            score_field="is_correct",
        )

        eval_input = EvalInput(eval_input_items=[
            EvalInputItem(
                id="schema_1",
                input_obj="Q",
                expected_output_obj="A",
                output_obj="A",
                trajectory=[],
                expected_trajectory=[],
                full_dataset_entry={},
            ),
        ])
        output = await evaluator.evaluate(eval_input)
        assert output.eval_output_items[0].score is True
