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

from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutput
from nat.plugins.langchain.eval.trajectory_evaluator import TrajectoryEvaluator


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
