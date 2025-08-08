# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel
from ragas.evaluation import EvaluationDataset
from ragas.evaluation import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Metric

from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.rag_evaluator.evaluate import RAGEvaluator

# pylint: disable=redefined-outer-name


class ExampleModel(BaseModel):
    content: str
    other: str


@pytest.fixture
def ragas_judge_llm() -> LangchainLLMWrapper:
    """Fixture providing a mocked LangchainLLMWrapper."""
    mock_llm = MagicMock(spec=LangchainLLMWrapper)
    mock_llm.ainvoke = AsyncMock(return_value="Mocked Async LLM Response")
    return mock_llm


@pytest.fixture
def ragas_metrics() -> Sequence[Metric]:
    """Fixture to provide mocked ragas metrics"""
    metric_names = ["AnswerAccuracy", "ContextRelevance", "ResponseGroundedness"]
    # Create mocked Metric objects for each metric name
    mocked_metrics = [MagicMock(spec=Metric, name=name) for name in metric_names]

    return mocked_metrics


@pytest.fixture
def rag_evaluator(ragas_judge_llm, ragas_metrics) -> RAGEvaluator:
    return RAGEvaluator(evaluator_llm=ragas_judge_llm, metrics=ragas_metrics)


@pytest.fixture
def metric_name() -> str:
    return "AnswerAccuracy"


@pytest.fixture
def rag_evaluator_content(ragas_judge_llm, ragas_metrics) -> RAGEvaluator:
    """RAGEvaluator configured to extract a specific field (`content`) from BaseModel or dict input objects."""
    return RAGEvaluator(evaluator_llm=ragas_judge_llm, metrics=ragas_metrics, input_obj_field="content")


def test_eval_input_to_ragas(rag_evaluator, rag_eval_input, intermediate_step_adapter):
    """Test eval_input mapping to ragasas dataset"""

    # call actual function
    dataset = rag_evaluator.eval_input_to_ragas(rag_eval_input)

    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset.samples) == len(rag_eval_input.eval_input_items)

    for sample, item in zip(dataset.samples, rag_eval_input.eval_input_items):
        # check if the contents of the ragas dataset match the original EvalInput
        assert isinstance(sample, SingleTurnSample)
        assert sample.user_input == item.input_obj
        assert sample.reference == item.expected_output_obj
        assert sample.response == item.output_obj
        assert sample.retrieved_contexts == intermediate_step_adapter.get_context(
            item.trajectory, intermediate_step_adapter.DEFAULT_EVENT_FILTER)


def test_ragas_to_eval_output(rag_evaluator, rag_eval_input, rag_user_inputs, metric_name):
    """Test ragas ouput mapping to AIQ Toolkit's EvalOuput"""
    mock_results_dataset = MagicMock()

    # Mock scores
    scores = [{metric_name: 0.8}, {metric_name: 0.9}]
    mock_results_dataset.scores = scores

    # Mock ragas DF converter
    mock_data = pd.DataFrame([{
        "user_input": rag_user_inputs[0], metric_name: scores[0][metric_name]
    }, {
        "user_input": rag_user_inputs[1], metric_name: scores[1][metric_name]
    }])
    mock_results_dataset.to_pandas.return_value = mock_data

    # Call actual function
    eval_output = rag_evaluator.ragas_to_eval_output(rag_eval_input, mock_results_dataset)

    assert isinstance(eval_output, EvalOutput)
    # Check average score
    expected_avg_score = sum(score[metric_name] for score in scores) / len(scores)
    assert eval_output.average_score == expected_avg_score

    # Validate length of eval_output_items
    assert len(eval_output.eval_output_items) == len(scores)

    # Check each output item
    for output_item, input_item, score in zip(eval_output.eval_output_items, rag_eval_input.eval_input_items, scores):
        # Ensure `id` is either `input_item.id` or `input_item.input_obj`
        assert output_item.id in (input_item.id, input_item.input_obj)
        assert output_item.score == score[metric_name]


@pytest.mark.parametrize(
    "scores, expected_avg_score, expected_item_count",
    [
        ([], 0.0, 0),  # Test empty dataset
        ([{
            "AnswerAccuracy": 0.8
        }], 0.8, 1),  # Test fewer entries (single result)
        ([{
            "AnswerAccuracy": 0.8
        }, {
            "AnswerAccuracy": 0.9
        }], 0.85, 2),  # Valid case
    ])
def test_ragas_to_eval_output_unexpected_entries(rag_evaluator,
                                                 rag_eval_input,
                                                 metric_name,
                                                 scores,
                                                 expected_avg_score,
                                                 expected_item_count):
    """Test ragas_to_eval_output with empty, fewer, and more dataset entries"""

    # Mock ragas results
    mock_results_dataset = MagicMock()
    mock_results_dataset.scores = scores

    # Mock ragas results convert
    mock_data = pd.DataFrame([{
        "user_input": f"Question {i+1}", metric_name: score[metric_name]
    } for i, score in enumerate(scores)])
    mock_results_dataset.to_pandas.return_value = mock_data

    # Call the actual function
    eval_output = rag_evaluator.ragas_to_eval_output(rag_eval_input, mock_results_dataset)

    # Assertions
    assert isinstance(eval_output, EvalOutput)
    assert len(eval_output.eval_output_items) == expected_item_count
    assert round(eval_output.average_score, 4) == round(expected_avg_score, 4)


def test_ragas_to_eval_output_nan_handling(rag_evaluator, rag_eval_input, metric_name):
    """
    Ensure that NaN or None scores are coerced to 0.0 for both the
    per‑item scores and the computed average score.
    """
    # fmt: off
    test_cases = [
        # (scores list, expected per‑item scores list, expected average)
        ([{metric_name: float("nan")}],            [0.0],           0.0),
        ([{metric_name: None}],                   [0.0],           0.0),
        ([{metric_name: float("nan")},
          {metric_name: 0.9}],                    [0.0, 0.9],      0.45),
        ([{metric_name: None},
          {metric_name: 0.9},
          {metric_name: float("nan")}],           [0.0, 0.9, 0.0], 0.3),
    ]
    # fmt: on

    for scores, expected_item_scores, expected_avg in test_cases:
        # Mock ragas results
        mock_results_dataset = MagicMock()
        mock_results_dataset.scores = scores

        # Build the mocked pandas DataFrame using the raw (possibly NaN/None) values
        mock_data = pd.DataFrame([{
            "user_input": f"Question {i+1}", metric_name: score[metric_name]
        } for i, score in enumerate(scores)])
        mock_results_dataset.to_pandas.return_value = mock_data

        # Invoke the method under test
        eval_output = rag_evaluator.ragas_to_eval_output(rag_eval_input, mock_results_dataset)

        # --- Assertions ---
        # Average score should match the expected value (with small tolerance for float ops)
        assert round(eval_output.average_score, 4) == round(expected_avg, 4)

        # Each individual item score should match the expected coercion results
        actual_item_scores = [item.score for item in eval_output.eval_output_items]
        assert actual_item_scores == expected_item_scores


async def test_rag_evaluate_success(rag_evaluator, rag_eval_input, ragas_judge_llm, ragas_metrics):
    """
    Test evaluate function to verify the following functions are called
    1. rag_evaluator.eval_input_to_ragas
    2. ragas.evaluate
    3. aiq.eval.evaluator.rag_evaluator.ragas_to_eval_output

    Only limited coverage is possible via unit tests as most of the functionality is
    implemented within the ragas framework. The simple example's end-to-end test covers functional
    testing.
    """
    mock_results_dataset = MagicMock()
    dataset = "mock_dataset"
    mock_output = "mock_output"

    with patch.object(rag_evaluator, "eval_input_to_ragas", return_value=dataset) as mock_eval_input_to_ragas, \
         patch.object(rag_evaluator, "ragas_to_eval_output", return_value=mock_output) as mock_ragas_to_eval_output, \
         patch("ragas.evaluate", new_callable=MagicMock) as mock_ragas_evaluate:

        # Configure mock return values
        mock_ragas_evaluate.return_value = mock_results_dataset

        # Call the actual function
        output = await rag_evaluator.evaluate(rag_eval_input)

        # Assertions to ensure correct function calls
        mock_eval_input_to_ragas.assert_called_once_with(rag_eval_input)
        mock_ragas_evaluate.assert_called_once()
        called_kwargs = mock_ragas_evaluate.call_args.kwargs

        assert called_kwargs["dataset"] == dataset
        assert called_kwargs["metrics"] == ragas_metrics
        assert called_kwargs["show_progress"] is True
        assert called_kwargs["llm"] == ragas_judge_llm
        mock_ragas_to_eval_output.assert_called_once_with(rag_eval_input, mock_results_dataset)

        # Validate final output
        assert output == mock_output


async def test_rag_evaluate_failure(rag_evaluator, rag_eval_input, ragas_judge_llm, ragas_metrics):
    """
    Validate evaluate processing when ragas.evaluate raises an exception. Also
    eval_input_to_ragas and ragas_to_eval_output are run as-is (not mocked) to validate
    their handling of the input and failed-output
    """

    error_message = "Mocked exception in ragas.evaluate"

    with patch("ragas.evaluate", side_effect=Exception(error_message)) as mock_ragas_evaluate:

        # Call function under test and ensure it does not crash
        try:
            output = await rag_evaluator.evaluate(rag_eval_input)
        except Exception:
            pytest.fail("rag_evaluator.evaluate() should handle exceptions gracefully and not crash.")

        ragas_dataset = rag_evaluator.eval_input_to_ragas(eval_input=rag_eval_input)
        # Validate ragas.evaluate was called and failed
        mock_ragas_evaluate.assert_called_once()
        called_kwargs = mock_ragas_evaluate.call_args.kwargs

        assert called_kwargs["dataset"] == ragas_dataset
        assert called_kwargs["metrics"] == ragas_metrics
        assert called_kwargs["show_progress"] is True
        assert called_kwargs["llm"] == ragas_judge_llm

        # Ensure output is valid with an average_score of 0.0
        assert isinstance(output, EvalOutput)
        assert output.average_score == 0.0
        assert output.eval_output_items == []  # No results due to failure


def test_extract_input_obj_base_model_with_field(rag_evaluator_content):
    """Ensure extract_input_obj returns the specified field from a Pydantic BaseModel."""
    model_obj = ExampleModel(content="hello world", other="ignore me")
    dummy_item = SimpleNamespace(input_obj=model_obj)

    extracted = rag_evaluator_content.extract_input_obj(dummy_item)
    assert extracted == "hello world"


def test_extract_input_obj_dict_with_field(rag_evaluator_content):
    """Ensure extract_input_obj returns the specified key when input_obj is a dict."""
    dict_obj = {"content": "dict hello", "other": 123}
    dummy_item = SimpleNamespace(input_obj=dict_obj)

    extracted = rag_evaluator_content.extract_input_obj(dummy_item)
    assert extracted == "dict hello"


def test_extract_input_obj_base_model_without_field(rag_evaluator, rag_evaluator_content):
    """
    When no input_obj_field is supplied, extract_input_obj should default to the model's JSON.
    Compare behaviour between default evaluator and one with a field configured.
    """
    model_obj = ExampleModel(content="json hello", other="data")
    dummy_item = SimpleNamespace(input_obj=model_obj)

    extracted_default = rag_evaluator.extract_input_obj(dummy_item)
    extracted_with_field = rag_evaluator_content.extract_input_obj(dummy_item)

    # Default evaluator returns the full JSON string, evaluator with field returns the field value.
    assert extracted_with_field == "json hello"
    assert extracted_default != extracted_with_field
    assert '"content":"json hello"' in extracted_default  # basic sanity check on JSON output
