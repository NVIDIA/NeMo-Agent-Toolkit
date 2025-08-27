# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from langchain.evaluation import TrajectoryEvalChain
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from nat.eval.evaluator.base_evaluator import BaseEvaluator
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem

logger = logging.getLogger(__name__)


class TrajectoryEvaluator(BaseEvaluator):

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
        max_concurrency: int = 8,
    ):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating Trajectory")
        self.llm = llm
        self.tools = tools
        # Initialize trajectory evaluation chain
        self.traj_eval_chain = TrajectoryEvalChain.from_llm(llm=self.llm,
                                                            tools=self.tools,
                                                            return_reasoning=True,
                                                            requires_reference=True)
        logger.debug("Trajectory evaluation chain initialized.")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """
        Evaluate a single EvalInputItem and return an EvalOutputItem.
        """
        from nat.data_models.intermediate_step import IntermediateStepType
        from nat.eval.intermediate_step_adapter import IntermediateStepAdapter

        intermediate_step_adapter = IntermediateStepAdapter()
        event_filter = [IntermediateStepType.LLM_END, IntermediateStepType.TOOL_END]

        question = item.input_obj
        generated_answer = item.output_obj
        agent_trajectory = intermediate_step_adapter.get_agent_actions(item.trajectory, event_filter)

        try:
            eval_result = await self.traj_eval_chain.aevaluate_agent_trajectory(
                input=question,
                agent_trajectory=agent_trajectory,
                prediction=generated_answer,
            )
        except Exception as e:
            logger.exception("Error evaluating trajectory for question: %s, Error: %s", question, e)
            return EvalOutputItem(id=item.id, score=0.0, reasoning=f"Error evaluating trajectory: {e}")

        reasoning = {
            "reasoning": eval_result["reasoning"],
            "trajectory": [(action.model_dump(), output) for (action, output) in agent_trajectory]
        }
        return EvalOutputItem(id=item.id, score=eval_result["score"], reasoning=reasoning)
