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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel

from nat.data_models.atif import ATIFAgentConfig
from nat.data_models.atif import ATIFStep
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutput
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.langchain.eval.tunable_rag_evaluator import TunableRagEvaluator
from nat.plugins.langchain.eval.tunable_rag_evaluator import TunableRagEvaluatorConfig
from nat.plugins.langchain.eval.tunable_rag_evaluator import register_tunable_rag_evaluator


@pytest.fixture
def mock_llm():
    return MagicMock(spec=BaseChatModel)


@pytest.fixture
def default_score_weights():
    return {"coverage": 1, "correctness": 1, "relevance": 1}


@pytest.fixture
def rag_eval_input():
    items = [
        EvalInputItem(id="1",
                      input_obj="What is AI?",
                      expected_output_obj="AI is artificial intelligence.",
                      output_obj="AI is the simulation of human intelligence.",
                      expected_trajectory=[],
                      trajectory=[],
                      full_dataset_entry={
                          "id": "1",
                          "question": "What is AI?",
                          "answer": "AI is artificial intelligence.",
                          "generated_answer": "AI is the simulation of human intelligence."
                      }),
        EvalInputItem(id="2",
                      input_obj="Define ML",
                      expected_output_obj="Machine Learning is a subset of AI.",
                      output_obj="ML helps machines learn.",
                      expected_trajectory=[],
                      trajectory=[],
                      full_dataset_entry={
                          "id": "2",
                          "question": "Define ML",
                          "answer": "Machine Learning is a subset of AI.",
                          "generated_answer": "ML helps machines learn."
                      })
    ]
    return EvalInput(eval_input_items=items)


@pytest.fixture
def evaluator(mock_llm, default_score_weights):
    return TunableRagEvaluator(llm=mock_llm,
                               judge_llm_prompt="Please evaluate the answer.",
                               max_concurrency=2,
                               default_scoring=True,
                               default_score_weights=default_score_weights,
                               llm_retry_control_params=None)


async def test_evaluate_success(evaluator, rag_eval_input):
    """Test successful evaluation using TunableRagEvaluator with mocked LLM."""

    # Mock LLM response content
    evaluator.llm.ainvoke = AsyncMock(side_effect=[
        MagicMock(content='{"coverage_score": 0.9, "correctness_score": 0.8,\
                "relevance_score": 0.7, "reasoning": "Solid answer."}'),
        MagicMock(content='{"coverage_score": 0.6, "correctness_score": 0.7,\
                "relevance_score": 0.8, "reasoning": "Good effort."}')
    ])

    eval_output: EvalOutput = await evaluator.evaluate(rag_eval_input)

    assert isinstance(eval_output, EvalOutput)
    assert len(eval_output.eval_output_items) == 2

    for item in eval_output.eval_output_items:
        assert item.score > 0
        assert isinstance(item.reasoning, dict)
        assert "reasoning" in item.reasoning

    assert round(eval_output.average_score, 2) > 0.0


async def test_evaluate_partial_failure(evaluator, rag_eval_input):
    """Test partial failure where one LLM response is invalid."""

    # One successful, one broken response
    evaluator.llm.ainvoke = AsyncMock(side_effect=[
        MagicMock(
            content='{"coverage_score": 0.9, "correctness_score": 0.9, "relevance_score": 0.9, "reasoning": "Perfect."}'
        ),
        MagicMock(content='INVALID JSON RESPONSE')
    ])

    eval_output: EvalOutput = await evaluator.evaluate(rag_eval_input)

    assert len(eval_output.eval_output_items) == 2

    successful_item = next(item for item in eval_output.eval_output_items if item.score > 0)
    failed_item = next(item for item in eval_output.eval_output_items if item.score == 0)

    assert "Perfect" in successful_item.reasoning["reasoning"]
    assert "parsing judge LLM response" in failed_item.reasoning["reasoning"]

    assert eval_output.average_score > 0
    assert eval_output.average_score < 1


async def test_evaluate_custom_scoring():
    """Test custom scoring mode (not default)"""

    llm = MagicMock(spec=BaseChatModel)
    evaluator = TunableRagEvaluator(llm=llm,
                                    judge_llm_prompt="Score this answer.",
                                    max_concurrency=1,
                                    default_scoring=False,
                                    default_score_weights={},
                                    llm_retry_control_params=None)

    input_data = EvalInput(eval_input_items=[
        EvalInputItem(id="1",
                      input_obj="What is NLP?",
                      expected_output_obj="Study of language processing",
                      output_obj="It's about language.",
                      expected_trajectory=[],
                      trajectory=[],
                      full_dataset_entry={
                          "id": "1",
                          "question": "What is NLP?",
                          "answer": "Study of language processing",
                          "generated_answer": "It's about language."
                      })
    ])

    llm.ainvoke = AsyncMock(return_value=MagicMock(content='{"score": 0.75, "reasoning": "Fair explanation."}'))

    output = await evaluator.evaluate(input_data)
    assert len(output.eval_output_items) == 1
    assert output.eval_output_items[0].score == 0.75
    assert output.eval_output_items[0].reasoning["reasoning"] == "Fair explanation."


@pytest.fixture(name="atif_samples")
def fixture_atif_samples():
    return [
        AtifEvalSample(
            item_id="1",
            trajectory=ATIFTrajectory(
                session_id="atif-1",
                agent=ATIFAgentConfig(name="test-agent", version="0.0.0"),
                steps=[
                    ATIFStep(step_id=1, source="user", message="What is AI?"),
                    ATIFStep(step_id=2, source="agent", message="AI is the simulation of human intelligence."),
                ],
            ),
            expected_output_obj="AI is artificial intelligence.",
            output_obj="AI is the simulation of human intelligence.",
            metadata={},
        ),
        AtifEvalSample(
            item_id="2",
            trajectory=ATIFTrajectory(
                session_id="atif-2",
                agent=ATIFAgentConfig(name="test-agent", version="0.0.0"),
                steps=[
                    ATIFStep(step_id=1, source="user", message="Define ML"),
                    ATIFStep(step_id=2, source="agent", message="ML helps machines learn."),
                ],
            ),
            expected_output_obj="Machine Learning is a subset of AI.",
            output_obj="ML helps machines learn.",
            metadata={},
        ),
    ]


async def test_evaluate_atif_success(evaluator, atif_samples):
    evaluator.llm.ainvoke = AsyncMock(side_effect=[
        MagicMock(content='{"coverage_score": 0.9, "correctness_score": 0.8,'
                  '"relevance_score": 0.7, "reasoning": "ATIF sample 1"}'),
        MagicMock(content='{"coverage_score": 0.6, "correctness_score": 0.7,'
                  '"relevance_score": 0.8, "reasoning": "ATIF sample 2"}')
    ])

    eval_output: EvalOutput = await evaluator.evaluate_atif_fn(atif_samples)
    assert isinstance(eval_output, EvalOutput)
    assert len(eval_output.eval_output_items) == 2
    assert eval_output.eval_output_items[0].score > 0
    assert eval_output.eval_output_items[1].score > 0
    assert eval_output.average_score > 0


async def test_legacy_and_atif_lane_parity_with_tolerance(evaluator, rag_eval_input, atif_samples):
    # Two legacy evaluations then two ATIF evaluations with identical per-item judge outputs.
    evaluator.llm.ainvoke = AsyncMock(side_effect=[
        MagicMock(content='{"coverage_score": 0.9, "correctness_score": 0.8,'
                  '"relevance_score": 0.7, "reasoning": "shared-1"}'),
        MagicMock(content='{"coverage_score": 0.6, "correctness_score": 0.7,'
                  '"relevance_score": 0.8, "reasoning": "shared-2"}'),
        MagicMock(content='{"coverage_score": 0.9, "correctness_score": 0.8,'
                  '"relevance_score": 0.7, "reasoning": "shared-1"}'),
        MagicMock(content='{"coverage_score": 0.6, "correctness_score": 0.7,'
                  '"relevance_score": 0.8, "reasoning": "shared-2"}'),
    ])

    legacy_output = await evaluator.evaluate(rag_eval_input)
    atif_output = await evaluator.evaluate_atif_fn(atif_samples)

    assert legacy_output.average_score == pytest.approx(atif_output.average_score, abs=0.01)
    assert legacy_output.eval_output_items[0].score == pytest.approx(atif_output.eval_output_items[0].score, abs=0.01)
    assert legacy_output.eval_output_items[1].score == pytest.approx(atif_output.eval_output_items[1].score, abs=0.01)


async def test_register_tunable_rag_evaluator_exposes_legacy_lane_by_default(mock_llm):
    config = TunableRagEvaluatorConfig(llm_name="judge_llm", judge_llm_prompt="Score this answer.")
    builder = MagicMock(spec=["get_llm", "get_max_concurrency"])
    builder.get_llm = AsyncMock(return_value=mock_llm)
    builder.get_max_concurrency.return_value = 2

    async with register_tunable_rag_evaluator(config, builder) as info:
        assert callable(info.evaluate_fn)
        assert not callable(getattr(info, "evaluate_atif_fn", None))


async def test_register_tunable_rag_evaluator_exposes_atif_lane_when_enabled(mock_llm):
    config = TunableRagEvaluatorConfig(llm_name="judge_llm",
                                       judge_llm_prompt="Score this answer.",
                                       enable_atif_evaluator=True)
    builder = MagicMock(spec=["get_llm", "get_max_concurrency"])
    builder.get_llm = AsyncMock(return_value=mock_llm)
    builder.get_max_concurrency.return_value = 2

    async with register_tunable_rag_evaluator(config, builder) as info:
        assert callable(info.evaluate_fn)
        assert callable(getattr(info, "evaluate_atif_fn", None))
