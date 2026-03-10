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

import typing
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

langchain_exceptions = pytest.importorskip("langchain_core.exceptions")
if not hasattr(langchain_exceptions, "ContextOverflowError"):
    pytest.skip(
        ("Skipping rag_evaluator tests: installed langchain_core lacks "
         "ContextOverflowError required by langchain_openai."),
        allow_module_level=True,
    )

if typing.TYPE_CHECKING:
    # We are lazily importing ragas to avoid import-time side effects such as applying the nest_asyncio patch, which is
    # not compatible with Python 3.12+, we want to ensure that we are able to apply the nest_asyncio2 patch instead.
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Metric

    from nat.plugins.ragas.rag_evaluator.evaluate import RAGEvaluator


class ExampleModel(BaseModel):
    content: str
    other: str


@pytest.fixture(name="atif_samples")
def fixture_atif_samples(rag_user_inputs, rag_expected_outputs, rag_generated_outputs):
    """ATIF-native samples for testing RAG ATIF evaluator path."""
    from nat.data_models.atif import ATIFAgentConfig
    from nat.data_models.atif import ATIFObservation
    from nat.data_models.atif import ATIFObservationResult
    from nat.data_models.atif import ATIFStep
    from nat.data_models.atif import ATIFTrajectory
    from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample

    samples = []
    for index, (user_input, expected_output,
                generated_output) in enumerate(zip(rag_user_inputs, rag_expected_outputs, rag_generated_outputs)):
        trajectory = ATIFTrajectory(
            session_id=str(index + 1),
            agent=ATIFAgentConfig(name="nat-agent", version="0.0.0"),
            steps=[
                ATIFStep(step_id=1, source="user", message=user_input),
                ATIFStep(step_id=2,
                         source="agent",
                         message=str(generated_output),
                         observation=ATIFObservation(results=[ATIFObservationResult(content="retrieved context")])),
            ],
        )
        samples.append(
            AtifEvalSample(
                item_id=index + 1,
                trajectory=trajectory,
                expected_output_obj=expected_output,
                output_obj=generated_output,
                metadata={},
            ))
    return samples


@pytest.fixture
def ragas_judge_llm() -> "LangchainLLMWrapper":
    """Fixture providing a mocked LangchainLLMWrapper."""
    from ragas.llms import LangchainLLMWrapper
    mock_llm = MagicMock(spec=LangchainLLMWrapper)
    mock_llm.ainvoke = AsyncMock(return_value="Mocked Async LLM Response")
    return mock_llm


@pytest.fixture
def ragas_metrics() -> "Sequence[Metric]":
    """Fixture to provide mocked ragas metrics"""
    from ragas.metrics import Metric
    metric_names = ["AnswerAccuracy", "ContextRelevance", "ResponseGroundedness"]
    # Create mocked Metric objects for each metric name
    mocked_metrics = [MagicMock(spec=Metric, name=name) for name in metric_names]

    return mocked_metrics


@pytest.fixture
def rag_evaluator(ragas_judge_llm, ragas_metrics) -> "RAGEvaluator":
    from nat.plugins.ragas.rag_evaluator.evaluate import RAGEvaluator
    return RAGEvaluator(evaluator_llm=ragas_judge_llm, metrics=ragas_metrics)


@pytest.fixture
def rag_evaluator_content(ragas_judge_llm, ragas_metrics) -> "RAGEvaluator":
    """RAGEvaluator configured to extract a specific field (`content`) from BaseModel or dict input objects."""
    from nat.plugins.ragas.rag_evaluator.evaluate import RAGEvaluator
    return RAGEvaluator(evaluator_llm=ragas_judge_llm, metrics=ragas_metrics, input_obj_field="content")


def test_eval_input_to_ragas(rag_evaluator, rag_eval_input, intermediate_step_adapter):
    """Test eval_input mapping to ragasas dataset"""
    from ragas import EvaluationDataset
    from ragas import SingleTurnSample

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


async def test_rag_evaluate_success(rag_evaluator, rag_eval_input):
    """
    Test evaluate function to verify the following functions are called
    1. `score_metric` is invoked once per input item.
    2. Returned `EvalOutput` has expected averaged score and item count.

    Only limited coverage is possible via unit tests as most of the functionality is
    implemented within the ragas framework. The simple example's end-to-end test covers functional
    testing.
    """
    with patch("nat.plugins.ragas.rag_evaluator.evaluate.score_metric",
               new_callable=AsyncMock,
               return_value=0.8) as mock_score_metric:
        output = await rag_evaluator.evaluate(rag_eval_input)

    assert mock_score_metric.await_count == len(rag_eval_input.eval_input_items)
    assert output.average_score == pytest.approx(0.8, abs=1e-9)
    assert len(output.eval_output_items) == len(rag_eval_input.eval_input_items)


async def test_rag_evaluate_failure(rag_evaluator, rag_eval_input):
    """
    Validate evaluate processing when metric scoring raises an exception.
    """

    from nat.data_models.evaluator import EvalOutput

    error_message = "Mocked exception in metric.ascore"

    with patch("nat.plugins.ragas.rag_evaluator.evaluate.score_metric",
               new_callable=AsyncMock,
               side_effect=Exception(error_message)) as mock_score_metric:

        # Call function under test and ensure it does not crash
        try:
            output = await rag_evaluator.evaluate(rag_eval_input)
        except Exception:
            pytest.fail("rag_evaluator.evaluate() should handle exceptions gracefully and not crash.")

        assert mock_score_metric.await_count >= 1

        # Ensure output is valid with an average_score of 0.0
        assert isinstance(output, EvalOutput)
        assert output.average_score == 0.0
        assert len(output.eval_output_items) == len(rag_eval_input.eval_input_items)
        assert all(item.score == 0.0 for item in output.eval_output_items)


def test_atif_samples_to_ragas(ragas_judge_llm, ragas_metrics, atif_samples):
    """Test ATIF sample mapping to ragas dataset."""
    from ragas import EvaluationDataset
    from ragas import SingleTurnSample

    from nat.plugins.ragas.rag_evaluator.atif_evaluate import RAGAtifEvaluator

    atif_evaluator = RAGAtifEvaluator(evaluator_llm=ragas_judge_llm, metrics=ragas_metrics)
    dataset = atif_evaluator.atif_samples_to_ragas(atif_samples)

    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset.samples) == len(atif_samples)
    for sample in dataset.samples:
        assert isinstance(sample, SingleTurnSample)
        assert sample.retrieved_contexts == ["retrieved context"]


async def test_rag_atif_evaluate_success(ragas_judge_llm, ragas_metrics, atif_samples):
    """Test ATIF-native evaluate path for RAGAS evaluator."""
    from nat.plugins.ragas.rag_evaluator.atif_evaluate import RAGAtifEvaluator

    dataset = MagicMock()
    dataset.samples = [MagicMock(), MagicMock()]
    dataset.__len__.return_value = len(dataset.samples)
    atif_evaluator = RAGAtifEvaluator(evaluator_llm=ragas_judge_llm, metrics=ragas_metrics)

    with patch.object(atif_evaluator, "atif_samples_to_ragas", return_value=dataset) as mock_to_ragas, \
         patch("nat.plugins.ragas.rag_evaluator.atif_evaluate.score_metric",
               new_callable=AsyncMock,
               return_value=0.6) as mock_score_metric:
        output = await atif_evaluator.evaluate(atif_samples)

        mock_to_ragas.assert_called_once_with(atif_samples)
        assert mock_score_metric.await_count == len(dataset.samples)
        assert output.average_score == pytest.approx(0.6, abs=1e-9)
        assert len(output.eval_output_items) == len(dataset.samples)


def test_rag_legacy_and_atif_dataset_parity(rag_evaluator,
                                            ragas_judge_llm,
                                            ragas_metrics,
                                            rag_eval_input,
                                            intermediate_step_adapter):
    """Ensure legacy and ATIF lanes produce equivalent ragas input samples."""
    from nat.data_models.atif import ATIFAgentConfig
    from nat.data_models.atif import ATIFObservation
    from nat.data_models.atif import ATIFObservationResult
    from nat.data_models.atif import ATIFStep
    from nat.data_models.atif import ATIFTrajectory
    from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
    from nat.plugins.ragas.rag_evaluator.atif_evaluate import RAGAtifEvaluator

    atif_samples = []
    for item in rag_eval_input.eval_input_items:
        contexts = intermediate_step_adapter.get_context(item.trajectory,
                                                         intermediate_step_adapter.DEFAULT_EVENT_FILTER)
        trajectory = ATIFTrajectory(
            session_id=str(item.id),
            agent=ATIFAgentConfig(name="nat-agent", version="0.0.0"),
            steps=[
                ATIFStep(step_id=1, source="user", message=str(item.input_obj)),
                ATIFStep(step_id=2,
                         source="agent",
                         message=str(item.output_obj),
                         observation=ATIFObservation(
                             results=[ATIFObservationResult(content=context) for context in contexts])),
            ],
        )
        atif_samples.append(
            AtifEvalSample(item_id=item.id,
                           trajectory=trajectory,
                           expected_output_obj=item.expected_output_obj,
                           output_obj=item.output_obj,
                           metadata={}))

    atif_evaluator = RAGAtifEvaluator(evaluator_llm=ragas_judge_llm, metrics=ragas_metrics)
    legacy_dataset = rag_evaluator.eval_input_to_ragas(rag_eval_input)
    atif_dataset = atif_evaluator.atif_samples_to_ragas(atif_samples)

    assert len(legacy_dataset.samples) == len(atif_dataset.samples)
    for legacy_sample, atif_sample in zip(legacy_dataset.samples, atif_dataset.samples):
        assert legacy_sample.user_input == atif_sample.user_input
        assert legacy_sample.reference == atif_sample.reference
        assert legacy_sample.response == atif_sample.response
        assert legacy_sample.retrieved_contexts == atif_sample.retrieved_contexts


@pytest.mark.parametrize(
    "atif_trajectory_steps, expected_user_input, expected_contexts",
    [
        ([], "", []),
        ([{
            "step_id": 1, "source": "user", "message": "question only"
        }], "question only", []),
    ],
)
def test_atif_samples_to_ragas_edge_cases(ragas_judge_llm,
                                          ragas_metrics,
                                          atif_trajectory_steps,
                                          expected_user_input,
                                          expected_contexts):
    """Ensure ATIF lane handles missing/partial trajectory content gracefully."""
    from nat.data_models.atif import ATIFAgentConfig
    from nat.data_models.atif import ATIFTrajectory
    from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
    from nat.plugins.ragas.rag_evaluator.atif_evaluate import RAGAtifEvaluator

    trajectory = ATIFTrajectory(session_id="edge-case-1",
                                agent=ATIFAgentConfig(name="nat-agent", version="0.0.0"),
                                steps=atif_trajectory_steps)
    atif_samples = [
        AtifEvalSample(item_id=1, trajectory=trajectory, expected_output_obj="ref", output_obj="resp", metadata={})
    ]

    atif_evaluator = RAGAtifEvaluator(evaluator_llm=ragas_judge_llm, metrics=ragas_metrics)
    dataset = atif_evaluator.atif_samples_to_ragas(atif_samples)

    assert len(dataset.samples) == 1
    assert dataset.samples[0].user_input == expected_user_input
    assert dataset.samples[0].retrieved_contexts == expected_contexts


async def test_rag_legacy_and_atif_score_parity(rag_evaluator,
                                                ragas_judge_llm,
                                                ragas_metrics,
                                                rag_eval_input,
                                                intermediate_step_adapter):
    """Ensure legacy and ATIF evaluator lanes produce parity scores on the same dataset."""
    from nat.data_models.atif import ATIFAgentConfig
    from nat.data_models.atif import ATIFObservation
    from nat.data_models.atif import ATIFObservationResult
    from nat.data_models.atif import ATIFStep
    from nat.data_models.atif import ATIFTrajectory
    from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
    from nat.plugins.ragas.rag_evaluator.atif_evaluate import RAGAtifEvaluator

    async def _mock_score_metric(_metric, sample):
        return 0.5 + (0.5 if sample.retrieved_contexts else 0.0)

    atif_samples = []
    for item in rag_eval_input.eval_input_items:
        contexts = intermediate_step_adapter.get_context(item.trajectory,
                                                         intermediate_step_adapter.DEFAULT_EVENT_FILTER)
        trajectory = ATIFTrajectory(
            session_id=str(item.id),
            agent=ATIFAgentConfig(name="nat-agent", version="0.0.0"),
            steps=[
                ATIFStep(step_id=1, source="user", message=str(item.input_obj)),
                ATIFStep(step_id=2,
                         source="agent",
                         message=str(item.output_obj),
                         observation=ATIFObservation(
                             results=[ATIFObservationResult(content=context) for context in contexts])),
            ],
        )
        atif_samples.append(
            AtifEvalSample(item_id=item.id,
                           trajectory=trajectory,
                           expected_output_obj=item.expected_output_obj,
                           output_obj=item.output_obj,
                           metadata={}))

    atif_evaluator = RAGAtifEvaluator(evaluator_llm=ragas_judge_llm, metrics=ragas_metrics)
    with patch("nat.plugins.ragas.rag_evaluator.evaluate.score_metric",
               new_callable=AsyncMock,
               side_effect=_mock_score_metric), \
         patch("nat.plugins.ragas.rag_evaluator.atif_evaluate.score_metric",
               new_callable=AsyncMock,
               side_effect=_mock_score_metric):
        legacy_output = await rag_evaluator.evaluate(rag_eval_input)
        atif_output = await atif_evaluator.evaluate(atif_samples)

    assert legacy_output.average_score == pytest.approx(atif_output.average_score, abs=1e-9)
    assert len(legacy_output.eval_output_items) == len(atif_output.eval_output_items)
    for legacy_item, atif_item in zip(legacy_output.eval_output_items, atif_output.eval_output_items):
        assert legacy_item.id == atif_item.id
        assert legacy_item.score == pytest.approx(atif_item.score, abs=1e-9)


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


async def test_register_ragas_evaluator_atif_lane_disabled_by_default():
    """Ensure RAGAS ATIF lane is opt-in while stabilizing."""
    from nat.plugins.ragas.rag_evaluator.register import RagasEvaluatorConfig
    from nat.plugins.ragas.rag_evaluator.register import register_ragas_evaluator

    builder = MagicMock()
    builder.get_llm = AsyncMock(return_value=MagicMock())
    builder.get_max_concurrency = MagicMock(return_value=1)

    config = RagasEvaluatorConfig(llm_name="judge", metric={"AnswerAccuracy": {"skip": True}})
    async with register_ragas_evaluator(config=config, builder=builder) as evaluator_info:
        assert hasattr(evaluator_info, "evaluate_fn")
        assert not hasattr(evaluator_info, "evaluate_atif_fn")

    builder.get_llm.assert_awaited_once()


async def test_register_ragas_evaluator_atif_lane_enabled():
    """Ensure RAGAS ATIF lane can be explicitly enabled by config."""
    from nat.plugins.ragas.rag_evaluator.register import RagasEvaluatorConfig
    from nat.plugins.ragas.rag_evaluator.register import register_ragas_evaluator

    builder = MagicMock()
    builder.get_llm = AsyncMock(return_value=MagicMock())
    builder.get_max_concurrency = MagicMock(return_value=1)

    config = RagasEvaluatorConfig(llm_name="judge",
                                  metric={"AnswerAccuracy": {
                                      "skip": True
                                  }},
                                  enable_atif_evaluator=True)
    async with register_ragas_evaluator(config=config, builder=builder) as evaluator_info:
        assert hasattr(evaluator_info, "evaluate_fn")
        assert callable(getattr(evaluator_info, "evaluate_atif_fn", None))

    builder.get_llm.assert_awaited_once()
