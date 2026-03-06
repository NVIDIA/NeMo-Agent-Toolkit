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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from nat.data_models.atif import ATIFAgentConfig
from nat.data_models.atif import ATIFObservation
from nat.data_models.atif import ATIFObservationResult
from nat.data_models.atif import ATIFStep
from nat.data_models.atif import ATIFToolCall
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutput
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.invocation_node import InvocationNode
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.langchain.eval.trajectory_evaluator import TrajectoryEvaluator
from nat.plugins.langchain.eval.trajectory_evaluator import TrajectoryEvaluatorConfig
from nat.plugins.langchain.eval.trajectory_evaluator import register_trajectory_evaluator


@pytest.fixture(name="mock_llm")
def fixture_mock_llm():
    return MagicMock(spec=BaseChatModel)


@pytest.fixture(name="mock_tools")
def fixture_mock_tools():
    return [MagicMock(spec=BaseTool)]


@pytest.fixture(name="trajectory_evaluator")
def fixture_trajectory_evaluator(mock_llm, mock_tools):
    return TrajectoryEvaluator(llm=mock_llm, tools=mock_tools, max_concurrency=4)


@pytest.fixture(name="rag_eval_input")
def fixture_rag_eval_input():
    return EvalInput(eval_input_items=[
        EvalInputItem(
            id="1",
            input_obj="What is AI?",
            expected_output_obj="Artificial intelligence.",
            output_obj="AI is artificial intelligence.",
            expected_trajectory=[],
            trajectory=[],
            full_dataset_entry={},
        ),
        EvalInputItem(
            id="2",
            input_obj="What is ML?",
            expected_output_obj="Machine learning.",
            output_obj="ML is a subset of AI.",
            expected_trajectory=[],
            trajectory=[],
            full_dataset_entry={},
        ),
    ])


async def test_trajectory_evaluate_success(trajectory_evaluator, rag_eval_input):
    scores = [
        {
            "score": 0.9, "reasoning": "result-1"
        },
        {
            "score": 0.8, "reasoning": "result-2"
        },
    ]
    expected_average = (0.9 + 0.8) / 2

    with patch.object(trajectory_evaluator, "traj_eval_chain") as mock_traj_eval_chain:
        mock_traj_eval_chain.aevaluate_agent_trajectory = AsyncMock(side_effect=scores)

        eval_output = await trajectory_evaluator.evaluate(rag_eval_input)

        assert isinstance(eval_output, EvalOutput)
        assert len(eval_output.eval_output_items) == 2
        assert eval_output.average_score == pytest.approx(expected_average)
        assert eval_output.eval_output_items[0].score == pytest.approx(0.9)
        assert eval_output.eval_output_items[1].score == pytest.approx(0.8)
        assert eval_output.eval_output_items[0].reasoning["reasoning"] == "result-1"
        assert eval_output.eval_output_items[1].reasoning["reasoning"] == "result-2"
        assert eval_output.eval_output_items[0].reasoning["trajectory"] == []
        assert eval_output.eval_output_items[1].reasoning["trajectory"] == []
        assert mock_traj_eval_chain.aevaluate_agent_trajectory.call_count == 2


async def test_trajectory_evaluate_failure(trajectory_evaluator, rag_eval_input):
    error_message = "Mocked trajectory evaluation failure"

    with patch.object(trajectory_evaluator, "traj_eval_chain") as mock_traj_eval_chain:
        mock_traj_eval_chain.aevaluate_agent_trajectory = AsyncMock(side_effect=[
            Exception(error_message),
            {
                "score": 0.8, "reasoning": "LGTM"
            },
        ])

        eval_output = await trajectory_evaluator.evaluate(rag_eval_input)

        assert isinstance(eval_output, EvalOutput)
        assert len(eval_output.eval_output_items) == 2
        assert eval_output.average_score == pytest.approx(0.4)

        failed_item = next(item for item in eval_output.eval_output_items if item.error is not None)
        successful_item = next(item for item in eval_output.eval_output_items if item.error is None)

        assert failed_item.score == pytest.approx(0.0)
        assert error_message in failed_item.error
        assert successful_item.score == pytest.approx(0.8)
        assert successful_item.reasoning["reasoning"] == "LGTM"


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
                    ATIFStep(
                        step_id=2,
                        source="agent",
                        model_name="mock-llm",
                        message="AI is artificial intelligence.",
                        tool_calls=[
                            ATIFToolCall(
                                tool_call_id="call-1",
                                function_name="web_search",
                                arguments={"query": "artificial intelligence"},
                            )
                        ],
                        observation=ATIFObservation(
                            results=[ATIFObservationResult(source_call_id="call-1", content="Search results context")]),
                    ),
                ],
            ),
            expected_output_obj="Artificial intelligence.",
            output_obj="AI is artificial intelligence.",
            metadata={},
        ),
        AtifEvalSample(
            item_id="2",
            trajectory=ATIFTrajectory(
                session_id="atif-2",
                agent=ATIFAgentConfig(name="test-agent", version="0.0.0"),
                steps=[
                    ATIFStep(step_id=1, source="user", message="What is ML?"),
                    ATIFStep(step_id=2, source="agent", model_name="mock-llm", message="ML is a subset of AI."),
                ],
            ),
            expected_output_obj="Machine learning.",
            output_obj="ML is a subset of AI.",
            metadata={},
        ),
    ]


async def test_trajectory_evaluate_atif_success(trajectory_evaluator, atif_samples):
    scores = [
        {
            "score": 0.9, "reasoning": "atif-1"
        },
        {
            "score": 0.8, "reasoning": "atif-2"
        },
    ]
    expected_average = (0.9 + 0.8) / 2

    with patch.object(trajectory_evaluator, "traj_eval_chain") as mock_traj_eval_chain:
        mock_traj_eval_chain.aevaluate_agent_trajectory = AsyncMock(side_effect=scores)
        eval_output = await trajectory_evaluator.evaluate_atif_fn(atif_samples)

    assert isinstance(eval_output, EvalOutput)
    assert len(eval_output.eval_output_items) == 2
    assert eval_output.average_score == pytest.approx(expected_average)
    assert eval_output.eval_output_items[0].score == pytest.approx(0.9)
    assert eval_output.eval_output_items[1].score == pytest.approx(0.8)
    assert eval_output.eval_output_items[0].reasoning["reasoning"] == "atif-1"
    assert eval_output.eval_output_items[1].reasoning["reasoning"] == "atif-2"
    assert mock_traj_eval_chain.aevaluate_agent_trajectory.call_count == 2


async def test_trajectory_legacy_and_atif_lane_parity_with_tolerance(trajectory_evaluator):
    llm_end_step = IntermediateStep(parent_id="root",
                                    function_ancestry=InvocationNode(function_name="llm_test",
                                                                     function_id="test-llm-end"),
                                    payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_END,
                                                                    name="mock-llm",
                                                                    data=StreamEventData(input="What is AI?",
                                                                                         output="AI answer")))
    tool_end_step = IntermediateStep(parent_id="root",
                                     function_ancestry=InvocationNode(function_name="tool_test",
                                                                      function_id="test-tool-end"),
                                     payload=IntermediateStepPayload(event_type=IntermediateStepType.TOOL_END,
                                                                     name="web_search",
                                                                     data=StreamEventData(
                                                                         input={"query": "What is AI?"},
                                                                         output="Search results context")))
    legacy_eval_input = EvalInput(eval_input_items=[
        EvalInputItem(id="1",
                      input_obj="What is AI?",
                      expected_output_obj="Artificial intelligence.",
                      output_obj="AI answer",
                      expected_trajectory=[],
                      trajectory=[llm_end_step, tool_end_step],
                      full_dataset_entry={})
    ])

    atif_samples = [
        AtifEvalSample(
            item_id="1",
            trajectory=ATIFTrajectory(
                session_id="atif-parity-1",
                agent=ATIFAgentConfig(name="test-agent", version="0.0.0"),
                steps=[
                    ATIFStep(step_id=1, source="user", message="What is AI?"),
                    ATIFStep(
                        step_id=2,
                        source="agent",
                        model_name="mock-llm",
                        message="AI answer",
                        tool_calls=[
                            ATIFToolCall(tool_call_id="call-1",
                                         function_name="web_search",
                                         arguments={"query": "What is AI?"})
                        ],
                        observation=ATIFObservation(
                            results=[ATIFObservationResult(source_call_id="call-1", content="Search results context")]),
                    ),
                ],
            ),
            expected_output_obj="Artificial intelligence.",
            output_obj="AI answer",
            metadata={},
        )
    ]

    async def score_from_trajectory(*, input, agent_trajectory, prediction):  # noqa: ARG001
        return {"score": float(len(agent_trajectory)), "reasoning": "trajectory-size"}

    with patch.object(trajectory_evaluator, "traj_eval_chain") as mock_traj_eval_chain:
        mock_traj_eval_chain.aevaluate_agent_trajectory = AsyncMock(side_effect=score_from_trajectory)
        legacy_output = await trajectory_evaluator.evaluate(legacy_eval_input)
        atif_output = await trajectory_evaluator.evaluate_atif_fn(atif_samples)

    assert legacy_output.average_score == pytest.approx(atif_output.average_score, abs=0.01)
    assert legacy_output.eval_output_items[0].score == pytest.approx(atif_output.eval_output_items[0].score, abs=0.01)


async def test_register_trajectory_evaluator_exposes_legacy_lane_by_default(mock_llm, mock_tools):
    config = TrajectoryEvaluatorConfig(llm_name="judge_llm")
    builder = MagicMock(spec=["get_llm", "get_max_concurrency", "get_all_tools"])
    builder.get_llm = AsyncMock(return_value=mock_llm)
    builder.get_all_tools = AsyncMock(return_value=mock_tools)
    builder.get_max_concurrency.return_value = 2

    async with register_trajectory_evaluator(config, builder) as info:
        assert callable(info.evaluate_fn)
        assert not callable(getattr(info, "evaluate_atif_fn", None))


async def test_register_trajectory_evaluator_exposes_atif_lane_when_enabled(mock_llm, mock_tools):
    config = TrajectoryEvaluatorConfig(llm_name="judge_llm", enable_atif_evaluator=True)
    builder = MagicMock(spec=["get_llm", "get_max_concurrency", "get_all_tools"])
    builder.get_llm = AsyncMock(return_value=mock_llm)
    builder.get_all_tools = AsyncMock(return_value=mock_tools)
    builder.get_max_concurrency.return_value = 2

    async with register_trajectory_evaluator(config, builder) as info:
        assert callable(info.evaluate_fn)
        assert callable(getattr(info, "evaluate_atif_fn", None))
