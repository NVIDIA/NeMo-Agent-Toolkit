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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest
from ragas.evaluation import EvaluationDataset
from ragas.evaluation import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Metric

from aiq.eval.evaluator.evaluator_model import EvalOutput
from aiq.eval.rag_evaluator.evaluate import RAGEvaluator

# pylint: disable=redefined-outer-name


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
        assert sample.retrieved_contexts == intermediate_step_adapter.get_context(item.trajectory)


def test_ragas_to_eval_output(rag_evaluator, rag_eval_input, rag_user_inputs, metric_name):
    """Test ragas ouput mapping to AgentIQ's EvalOuput"""
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
        mock_ragas_evaluate.assert_called_once_with(dataset=dataset,
                                                    metrics=ragas_metrics,
                                                    show_progress=True,
                                                    llm=ragas_judge_llm)
        mock_ragas_to_eval_output.assert_called_once_with(rag_eval_input, mock_results_dataset)

        # Validate final output
        assert output == mock_output


async def test_rag_evaluate_failure(rag_evaluator, rag_eval_input, ragas_judge_llm, ragas_metrics):
    """
    Validate evalute processing when ragas.evaluate raises an exception. Also
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
        mock_ragas_evaluate.assert_called_once_with(dataset=ragas_dataset,
                                                    metrics=ragas_metrics,
                                                    show_progress=True,
                                                    llm=ragas_judge_llm)

        # Ensure output is valid with an average_score of 0.0
        assert isinstance(output, EvalOutput)
        assert output.average_score == 0.0
        assert output.eval_output_items == []  # No results due to failure
