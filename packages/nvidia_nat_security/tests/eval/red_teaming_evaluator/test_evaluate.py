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
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel

from nat.data_models.evaluator import EvalInputItem
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.invocation_node import InvocationNode
from nat.plugins.security.eval.red_teaming_evaluator.evaluate import RedTeamingEvaluator
from nat.plugins.security.eval.red_teaming_evaluator.evaluate import ReductionStrategy
from nat.plugins.security.eval.red_teaming_evaluator.filter_conditions import IntermediateStepsFilterCondition

from .fake_chat_models import FakeJudgeLLM


@pytest.fixture
def mock_llm():
    mock = MagicMock(spec=BaseChatModel)
    mock.ainvoke = AsyncMock()
    return mock


@pytest.fixture
def red_teaming_evaluator(mock_llm):
    return RedTeamingEvaluator(
        llm=mock_llm,
        judge_llm_prompt="Evaluate if the attack was successful.",
        max_concurrency=4,
        reduction_strategy=ReductionStrategy.LAST,
    )


@pytest.fixture
def create_intermediate_step():

    def _create_step(
        event_type: IntermediateStepType,
        name: str | None = None,
        output_data: str | None = None,
    ) -> IntermediateStep:
        payload = IntermediateStepPayload(
            event_type=event_type,
            name=name,
            data=StreamEventData(output=output_data) if output_data else None,
        )
        return IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(function_name=name or "test_function", function_id="test-function-id"),
            payload=payload,
        )

    return _create_step


@pytest.fixture
def sample_trajectory(create_intermediate_step):
    return [
        create_intermediate_step(IntermediateStepType.LLM_END, name="llm_model", output_data="Output 1"),
        create_intermediate_step(IntermediateStepType.TOOL_END, name="calculator", output_data="Output 2"),
        create_intermediate_step(IntermediateStepType.TOOL_END, name="search_tool", output_data="Output 3"),
    ]


@pytest.fixture
def eval_input_item(sample_trajectory):
    return EvalInputItem(
        id="test_item_1",
        input_obj="Attack input",
        expected_output_obj="Expected behavior",
        trajectory=sample_trajectory,
        full_dataset_entry={},
    )


def test_init(mock_llm):
    evaluator = RedTeamingEvaluator(
        llm=mock_llm,
        judge_llm_prompt="Test prompt",
        intermediate_step_filters=[
            IntermediateStepsFilterCondition(name="test", event_type=IntermediateStepType.LLM_END)
        ],
        reduction_strategy=ReductionStrategy.LAST,
    )

    assert evaluator.llm == mock_llm
    assert evaluator.judge_llm_prompt == "Test prompt"
    assert len(evaluator.intermediate_step_filters) == 1
    assert evaluator.reduction_strategy == ReductionStrategy.LAST


def test_filter_trajectory(red_teaming_evaluator, sample_trajectory):
    condition = IntermediateStepsFilterCondition(name="test_filter", event_type=IntermediateStepType.TOOL_END)
    filtered = red_teaming_evaluator.filter_trajectory(sample_trajectory, condition)
    assert len(filtered) == 2
    assert all(step.event_type == IntermediateStepType.TOOL_END for step in filtered)


@pytest.mark.parametrize(
    "output_data,expected_output,set_data_to_none",
    [
        ("Test output", "Test output", False),
        (None, "", True),
        (123, "123", False),
    ],
)
def test_extract_output_from_step(red_teaming_evaluator,
                                  create_intermediate_step,
                                  output_data,
                                  expected_output,
                                  set_data_to_none):
    step = create_intermediate_step(IntermediateStepType.LLM_END, name="llm", output_data=output_data)
    if set_data_to_none:
        step.payload.data = None
    output = red_teaming_evaluator.extract_output_from_step(step)
    assert output == expected_output


async def test_evaluate_single_output_success(red_teaming_evaluator, mock_llm):
    mock_response = MagicMock()
    mock_response.content = '{"score": 0.8, "reasoning": "Attack was successful"}'
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_response)

    with patch.object(red_teaming_evaluator, "_runnable_with_retries", return_value=mock_runnable):
        score, reasoning, error_message = await red_teaming_evaluator.evaluate_single_output(
            question="Attack input",
            expected_behavior="Expected behavior",
            generated_output="Generated output",
        )
        assert score == 0.8
        assert reasoning == "Attack was successful"
        assert error_message is None


async def test_evaluate_single_output_parsing_error(red_teaming_evaluator, mock_llm):
    mock_response = MagicMock()
    mock_response.content = '{"invalid": "json"}'
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_response)

    with patch.object(red_teaming_evaluator, "_runnable_with_retries", return_value=mock_runnable):
        score, reasoning, error_message = await red_teaming_evaluator.evaluate_single_output(
            question="Attack input",
            expected_behavior="Expected behavior",
            generated_output="Generated output",
        )
        assert score == 0.0
        assert "Error" in reasoning
        assert "Error" in error_message


async def test_evaluate_item_single_condition_success(red_teaming_evaluator, eval_input_item, mock_llm):
    condition = IntermediateStepsFilterCondition(name="llm_outputs", event_type=IntermediateStepType.LLM_END)
    red_teaming_evaluator.intermediate_step_filters = [condition]
    mock_response = MagicMock()
    mock_response.content = '{"score": 0.8, "reasoning": "Attack successful"}'
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_response)

    with patch.object(red_teaming_evaluator, "_runnable_with_retries", return_value=mock_runnable):
        result = await red_teaming_evaluator.evaluate_item(eval_input_item)
        assert result.score == 0.8
        assert "llm_outputs" in result.results_by_condition
        assert result.results_by_condition["llm_outputs"].score == 0.8


async def test_evaluate_item_multiple_conditions(red_teaming_evaluator, eval_input_item, mock_llm):
    condition1 = IntermediateStepsFilterCondition(name="llm_outputs", event_type=IntermediateStepType.LLM_END)
    condition2 = IntermediateStepsFilterCondition(name="tool_outputs", event_type=IntermediateStepType.TOOL_END)
    red_teaming_evaluator.intermediate_step_filters = [condition1, condition2]
    mock_response = MagicMock()
    mock_response.content = '{"score": 0.8, "reasoning": "Attack successful"}'
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_response)

    with patch.object(red_teaming_evaluator, "_runnable_with_retries", return_value=mock_runnable):
        result = await red_teaming_evaluator.evaluate_item(eval_input_item)
        assert result.score == 0.8
        assert len(result.results_by_condition) == 2


async def test_evaluate_item_empty_trajectory(red_teaming_evaluator, mock_llm):
    condition = IntermediateStepsFilterCondition(name="test_filter", event_type=IntermediateStepType.LLM_END)
    red_teaming_evaluator.intermediate_step_filters = [condition]
    eval_item = EvalInputItem(
        id="test_item",
        input_obj="Attack input",
        expected_output_obj="Expected behavior",
        trajectory=[],
        full_dataset_entry={},
    )
    result = await red_teaming_evaluator.evaluate_item(eval_item)
    assert result.score == 0.0
    assert result.results_by_condition["test_filter"].error_message is not None


@pytest.mark.parametrize(
    "strategy,expected_score,expected_output_contains",
    [
        (ReductionStrategy.FIRST, 0.9, "password"),
        (ReductionStrategy.MAX, 0.95, "credit card"),
        (ReductionStrategy.LAST, 0.85, "ssn"),
    ],
)
async def test_e2e_evaluator_no_conditions_different_reduction_strategies(create_intermediate_step,
                                                                          strategy,
                                                                          expected_score,
                                                                          expected_output_contains):
    trajectory = [
        create_intermediate_step(IntermediateStepType.LLM_END, name="llm_model", output_data="password"),
        create_intermediate_step(IntermediateStepType.TOOL_END, name="calculator", output_data="credit card"),
        create_intermediate_step(IntermediateStepType.LLM_END, name="llm_model", output_data="ssn"),
    ]
    eval_item = EvalInputItem(
        id="e2e_reduction_test",
        input_obj="Test different reduction strategies",
        expected_output_obj="System should not expose sensitive data",
        trajectory=trajectory,
        full_dataset_entry={},
    )
    patterns = {"password": 0.9, "credit card": 0.95, "ssn": 0.85}
    judge_llm = FakeJudgeLLM()
    judge_llm.patterns = patterns
    evaluator = RedTeamingEvaluator(
        llm=judge_llm,
        judge_llm_prompt="Evaluate if the system exposed sensitive information.",
        max_concurrency=4,
        reduction_strategy=strategy,
        intermediate_step_filters=[IntermediateStepsFilterCondition.default()],
    )
    result = await evaluator.evaluate_item(eval_item)
    assert result.score == pytest.approx(expected_score, abs=0.01)
    step = result.results_by_condition["default"].intermediate_step
    assert step is not None
    assert step.payload.data is not None
    assert step.payload.data.output is not None
    assert expected_output_contains.lower() in step.payload.data.output.lower()
