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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.trajectory_evaluator.evaluate import TrajectoryEvaluator

# pylint: disable=redefined-outer-name


@pytest.fixture
def mock_llm():
    """Fixture to provide a mocked LLM."""
    return MagicMock(spec=BaseChatModel)


@pytest.fixture
def mock_tools():
    """Fixture to provide a list of mocked tools."""
    return [MagicMock(spec=BaseTool)]


@pytest.fixture
def trajectory_evaluator(mock_llm, mock_tools):
    """Fixture to provide a TrajectoryEvaluator instance."""
    return TrajectoryEvaluator(llm=mock_llm, tools=mock_tools, max_concurrency=4)


@pytest.fixture
def trajectory_evaluator_results() -> tuple[float, list[dict]]:
    """Mock results for trajectory evaluation."""
    score_1 = 0.9
    score_2 = 0.8
    average_score = (score_1 + score_2) / 2
    return average_score, [{"score": score_1, "reasoning": "result-1"}, {"score": score_2, "reasoning": "result-2"}]


@pytest.fixture
def agent_trajectories(intermediate_step_adapter, rag_intermediate_steps):
    """Get the agent trajectory from the mock intermediate steps."""
    steps_1, steps_2 = rag_intermediate_steps
    return [
        intermediate_step_adapter.get_agent_actions(steps_1, intermediate_step_adapter.DEFAULT_EVENT_FILTER),
        intermediate_step_adapter.get_agent_actions(steps_2, intermediate_step_adapter.DEFAULT_EVENT_FILTER)
    ]


async def test_trajectory_evaluate_success(trajectory_evaluator,
                                           rag_eval_input,
                                           trajectory_evaluator_results,
                                           agent_trajectories):
    """Test successful evaluation of agent trajectories"""

    average_score, scores = trajectory_evaluator_results

    # Mock trajectory evaluation chain and provide mock results
    with patch.object(trajectory_evaluator, "traj_eval_chain") as mock_traj_eval_chain:
        mock_traj_eval_chain.aevaluate_agent_trajectory = AsyncMock(side_effect=(iter(scores)))

        # Call function under test
        eval_output = await trajectory_evaluator.evaluate(rag_eval_input)

        # Validate average score
        assert isinstance(eval_output, EvalOutput)
        assert round(eval_output.average_score, 5) == round(average_score, 5)
        assert len(eval_output.eval_output_items) == len(scores)

        # Validate the score and reasoning for each individual item
        for output_item, agent_trajectory in zip(eval_output.eval_output_items, agent_trajectories):
            reasoning_text = output_item.reasoning["reasoning"]

            # Find the correct expected score based on reasoning substring
            expected_score_entry = next((score for score in scores if score["reasoning"] in reasoning_text), None)

            assert expected_score_entry is not None, f"Unexpected reasoning: {reasoning_text}"
            assert output_item.score == expected_score_entry["score"], f"Score mismatch for reasoning: {reasoning_text}"

            # Validate trajectory structure
            assert "trajectory" in output_item.reasoning
            expected_trajectory = [(action.model_dump(), output) for (action, output) in agent_trajectory]
            assert output_item.reasoning["trajectory"] == expected_trajectory, \
                f"Trajectory mismatch for reasoning: {reasoning_text}"

        # Ensure the evaluation method was called correctly
        assert mock_traj_eval_chain.aevaluate_agent_trajectory.call_count == len(rag_eval_input.eval_input_items)


async def test_trajectory_evaluate_failure(trajectory_evaluator, rag_eval_input, agent_trajectories):
    """
    Test evaluation handling when aevaluate_agent_trajectory raises an exception for one of the eval_input_items
    """

    # Simulate an exception in one of the evaluations
    error_message = "Mocked trajectory evaluation failure"
    failing_score_value = 0.0
    failing_score = {"score": failing_score_value, "reasoning": f"Error evaluating trajectory: {error_message}"}
    successful_score_value = 0.8
    successful_score = {"score": successful_score_value, "reasoning": "LGTM"}
    expected_average_score = (failing_score_value + successful_score_value) / 2

    with patch.object(trajectory_evaluator, "traj_eval_chain") as mock_traj_eval_chain:
        mock_traj_eval_chain.aevaluate_agent_trajectory = AsyncMock(side_effect=[
            Exception(error_message),  # Simulate failure
            successful_score  # Normal execution for second evaluation
        ])

        # Call function under test
        eval_output = await trajectory_evaluator.evaluate(rag_eval_input)

        # Validate returned type
        assert isinstance(eval_output, EvalOutput)

        # Ensure 2 results are returned, even with one failure
        assert len(eval_output.eval_output_items) == len(rag_eval_input.eval_input_items)

        # Extract the failed item and the successful item
        failed_item = None
        successful_item = None
        for output_item in eval_output.eval_output_items:
            if not output_item.score:
                failed_item = output_item
            else:
                successful_item = output_item

        # Validate failed evaluation
        assert failed_item, "Failed item missing in the output"
        assert failed_item.score == failing_score.get("score")
        assert "Error evaluating trajectory" in failed_item.reasoning, \
            f"Expected error message in reasoning, got: {failed_item.reasoning}"

        # Validate successful evaluation
        assert successful_item, "Successful item missing in the output"
        assert successful_item.score == successful_score.get("score")
        assert successful_item.reasoning.get("reasoning") == successful_score.get("reasoning")

        # Validate average score calculation (will be lower because of the failure)
        assert eval_output.average_score == expected_average_score

        # Ensure the evaluation method was called the correct number of times
        assert mock_traj_eval_chain.aevaluate_agent_trajectory.call_count == len(rag_eval_input.eval_input_items)
