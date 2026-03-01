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

import logging

from langchain_classic.evaluation import TrajectoryEvalChain
from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutputItem
from nat.data_models.evaluator import EvaluatorLLMConfig
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator
from nat.utils.exception_handlers.automatic_retries import patch_with_retry

logger = logging.getLogger(__name__)

_DEFAULT_EVENT_FILTER = [IntermediateStepType.LLM_END, IntermediateStepType.TOOL_END]


class TrajectoryEvaluatorConfig(EvaluatorLLMConfig, name="trajectory"):
    """Agent trajectory evaluator configuration."""

    pass


def _to_agent_actions(intermediate_steps: list[IntermediateStep]) -> list[tuple[AgentAction, str]]:
    """Convert intermediate steps to LangChain `agent_trajectory` tuples."""
    filtered_steps = [step for step in intermediate_steps if step.event_type in _DEFAULT_EVENT_FILTER]
    last_llm_end_step: IntermediateStep | None = None
    agent_actions: list[tuple[AgentAction, str]] = []

    for step in filtered_steps:
        log = getattr(last_llm_end_step.data, "output", "") if last_llm_end_step else ""
        if step.event_type == IntermediateStepType.LLM_END:
            last_llm_end_step = step
            log = ""

        tool_name = step.name or ""
        tool_input = getattr(step.data, "input", "") if step.data else ""
        tool_output = getattr(step.data, "output", "") if step.data else ""
        action = AgentAction(tool=tool_name, tool_input=tool_input, log=log)
        agent_actions.append((action, tool_output))

    return agent_actions


class TrajectoryEvaluator(BaseEvaluator):

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool] | None = None, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency)
        self.traj_eval_chain = TrajectoryEvalChain.from_llm(llm=llm,
                                                            tools=tools,
                                                            return_reasoning=True,
                                                            requires_reference=True)

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        question = item.input_obj
        generated_answer = item.output_obj
        agent_trajectory = _to_agent_actions(item.trajectory)

        try:
            eval_result = await self.traj_eval_chain.aevaluate_agent_trajectory(input=question,
                                                                                agent_trajectory=agent_trajectory,
                                                                                prediction=generated_answer)
        except Exception as e:
            logger.exception("Error evaluating trajectory for question: %s, Error: %s", question, e)
            return EvalOutputItem(id=item.id, score=0.0, reasoning={}, error=str(e))

        reasoning = {
            "reasoning": eval_result["reasoning"],
            "trajectory": [(action.model_dump(), output) for (action, output) in agent_trajectory],
        }
        return EvalOutputItem(id=item.id, score=eval_result["score"], reasoning=reasoning)


@register_evaluator(config_type=TrajectoryEvaluatorConfig)
async def register_trajectory_evaluator(config: TrajectoryEvaluatorConfig, builder: EvalBuilder):
    from nat.builder.framework_enum import LLMFrameworkEnum

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    if config.do_auto_retry:
        llm = patch_with_retry(
            llm,
            retries=config.num_retries,
            retry_codes=config.retry_on_status_codes,
            retry_on_messages=config.retry_on_errors,
        )

    tools = await builder.get_all_tools(wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    evaluator = TrajectoryEvaluator(llm=llm, tools=tools, max_concurrency=builder.get_max_concurrency())
    yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description="Trajectory Evaluator")
