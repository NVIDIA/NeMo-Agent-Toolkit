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

import asyncio
import logging

from langchain.evaluation import TrajectoryEvalChain
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from tqdm import tqdm

from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalInputItem
from aiq.eval.evaluator.evaluator_model import EvalOutput
from aiq.eval.evaluator.evaluator_model import EvalOutputItem
from aiq.eval.utils.tqdm_position_registry import TqdmPositionRegistry

logger = logging.getLogger(__name__)


class TrajectoryEvaluator:

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
        max_concurrency: int = 8,
    ):

        self.llm = llm
        self.tools = tools
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        # Initialize trajectory evaluation chain
        self.traj_eval_chain = TrajectoryEvalChain.from_llm(llm=self.llm,
                                                            tools=self.tools,
                                                            return_reasoning=True,
                                                            requires_reference=True)
        logger.debug("Trajectory evaluation chain initialized.")

    async def evaluate(self, eval_input: EvalInput) -> EvalOutput:
        """
        Evaluates the agent trajectories using trajectory evaluation chain.
        """

        num_records = len(eval_input.eval_input_items)
        logger.info("Running trajectory evaluation with %d records", num_records)
        from aiq.data_models.intermediate_step import IntermediateStepType
        from aiq.eval.intermediate_step_adapter import IntermediateStepAdapter

        intermediate_step_adapter = IntermediateStepAdapter()
        event_filter = [IntermediateStepType.LLM_END, IntermediateStepType.TOOL_END]

        async def process_item(item: EvalInputItem) -> tuple[float, dict]:
            """
            Evaluate a single EvalInputItem asynchronously and return a tuple of-
            1. score
            2. reasoning for the score
            """
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
                logger.exception("Error evaluating trajectory for question: %s, Error: %s", question, e, exc_info=True)
                return 0.0, f"Error evaluating trajectory: {e}"

            reasoning = {
                "reasoning": eval_result["reasoning"],
                "trajectory": [(action.model_dump(), output) for (action, output) in agent_trajectory]
            }
            return eval_result["score"], reasoning

        async def wrapped_process(item: EvalInputItem) -> tuple[float, dict]:
            async with self.semaphore:
                result = await process_item(item)
                pbar.update(1)
                return result

        # Execute all evaluations asynchronously
        try:
            tqdm_position = TqdmPositionRegistry.claim()
            pbar = tqdm(total=len(eval_input.eval_input_items), desc="Evaluating Trajectory", position=tqdm_position)
            results = await asyncio.gather(*[wrapped_process(item) for item in eval_input.eval_input_items])
        finally:
            pbar.close()
            TqdmPositionRegistry.release(tqdm_position)

        # Extract scores and reasonings
        sample_scores, sample_reasonings = zip(*results) if results else ([], [])

        # Compute average score
        avg_score = round(sum(sample_scores) / len(sample_scores), 2) if sample_scores else 0.0

        # Construct EvalOutputItems
        eval_output_items = [
            EvalOutputItem(id=item.id, score=score, reasoning=reasoning)
            for item, score, reasoning in zip(eval_input.eval_input_items, sample_scores, sample_reasonings)
        ]

        return EvalOutput(average_score=avg_score, eval_output_items=eval_output_items)
