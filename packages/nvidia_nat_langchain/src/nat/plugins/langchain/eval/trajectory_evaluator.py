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

import asyncio
import logging
from collections.abc import Mapping

from langchain_classic.evaluation import TrajectoryEvalChain
from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import Field

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutput
from nat.data_models.evaluator import EvalOutputItem
from nat.data_models.evaluator import EvaluatorLLMConfig
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSampleList
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator
from nat.utils.exception_handlers.automatic_retries import patch_with_retry

logger = logging.getLogger(__name__)

_DEFAULT_EVENT_FILTER = [IntermediateStepType.LLM_END, IntermediateStepType.TOOL_END]


def _coerce_text(value) -> str:
    """Best-effort coercion to text for judge-chain inputs."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


class TrajectoryEvaluatorConfig(EvaluatorLLMConfig, name="trajectory"):
    """Agent trajectory evaluator configuration."""

    enable_atif_evaluator: bool = Field(
        default=False,
        description="Enable ATIF-native trajectory evaluator lane. Disabled by default during migration.",
    )


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


def _message_to_text(message) -> str:
    """Convert ATIF message payloads into text for LangChain trajectory scoring."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message

    if isinstance(message, dict):
        parts_iterable = message.get("parts")
        if parts_iterable is None:
            parts_iterable = [message]
    else:
        parts_iterable = message

    text_parts: list[str] = []
    for part in parts_iterable:
        part_type = getattr(part, "type", None)
        part_text = getattr(part, "text", None)
        part_source = getattr(part, "source", None)

        if isinstance(part, dict):
            part_type = part.get("type", part_type)
            part_text = part.get("text", part_text)
            part_source = part.get("source", part_source)

        if part_type == "text" and isinstance(part_text, str) and part_text:
            text_parts.append(part_text)
            continue

        if part_type == "image":
            source_path = getattr(part_source, "path", None)
            if isinstance(part_source, dict):
                source_path = part_source.get("path", source_path)
            if isinstance(source_path, str) and source_path:
                text_parts.append(source_path)
    return "\n".join(text_parts)


def _atif_to_agent_actions(trajectory) -> list[tuple[AgentAction, str]]:
    """Convert an ATIF trajectory into LangChain `agent_trajectory` tuples."""
    agent_actions: list[tuple[AgentAction, str]] = []
    for step in trajectory.steps:
        if step.source != "agent":
            continue

        agent_message = _message_to_text(step.message)
        if step.model_name or agent_message:
            llm_action = AgentAction(tool=step.model_name or "", tool_input="", log="")
            agent_actions.append((llm_action, agent_message))

        if not step.tool_calls:
            continue

        observation_by_call_id: dict[str, str] = {}
        if step.observation:
            for result in step.observation.results:
                if result.source_call_id:
                    observation_by_call_id[result.source_call_id] = _message_to_text(result.content)

        for tool_call in step.tool_calls:
            if isinstance(tool_call.arguments, dict):
                tool_input = tool_call.arguments
            elif isinstance(tool_call.arguments, Mapping):
                tool_input = dict(tool_call.arguments)
            else:
                tool_input = str(tool_call.arguments)
            action = AgentAction(tool=tool_call.function_name, tool_input=tool_input, log=agent_message)
            tool_output = observation_by_call_id.get(tool_call.tool_call_id, "")
            agent_actions.append((action, tool_output))

    return agent_actions


def _atif_to_user_input(trajectory) -> str:
    """Extract first user message from ATIF trajectory."""
    for step in trajectory.steps:
        if step.source == "user":
            text = _message_to_text(step.message)
            if text:
                return text
    return ""


class TrajectoryEvaluator(BaseEvaluator):

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool] | None = None, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency)
        self.traj_eval_chain = TrajectoryEvalChain.from_llm(llm=llm,
                                                            tools=tools,
                                                            return_reasoning=True,
                                                            requires_reference=True)

    async def _evaluate_with_trajectory(self,
                                        item_id,
                                        lane: str,
                                        question: str,
                                        generated_answer: str,
                                        agent_trajectory: list[tuple[AgentAction, str]]) -> EvalOutputItem:
        """Run trajectory scoring for one item regardless of input lane."""
        question_text = _coerce_text(question)
        generated_answer_text = _coerce_text(generated_answer)
        try:
            eval_result = await self.traj_eval_chain.aevaluate_agent_trajectory(input=question_text,
                                                                                agent_trajectory=agent_trajectory,
                                                                                prediction=generated_answer_text)
        except Exception as e:
            # Some judge models occasionally miss the strict "Score: " suffix
            # expected by LangChain's legacy trajectory parser.
            if isinstance(e, ValueError) and "not enough values to unpack" in str(e):
                logger.warning("Trajectory judge output parsing failed [lane=%s item_id=%s]: %s", lane, item_id, e)
            else:
                logger.exception("Error evaluating trajectory [lane=%s item_id=%s]", lane, item_id)
            return EvalOutputItem(id=item_id, score=0.0, reasoning={}, error=str(e))

        reasoning = {
            "reasoning": eval_result["reasoning"],
            "trajectory": [(action.model_dump(), output) for (action, output) in agent_trajectory],
        }
        return EvalOutputItem(id=item_id, score=eval_result["score"], reasoning=reasoning)

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        question = item.input_obj
        generated_answer = item.output_obj
        agent_trajectory = _to_agent_actions(item.trajectory)
        return await self._evaluate_with_trajectory(item.id, "legacy", question, generated_answer, agent_trajectory)

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        """Evaluate a single ATIF-native sample."""
        question = _atif_to_user_input(sample.trajectory)
        generated_answer = sample.output_obj if sample.output_obj is not None else ""
        agent_trajectory = _atif_to_agent_actions(sample.trajectory)
        return await self._evaluate_with_trajectory(sample.item_id,
                                                    "atif",
                                                    question,
                                                    generated_answer,
                                                    agent_trajectory)

    async def evaluate_atif_fn(self, atif_samples: AtifEvalSampleList) -> EvalOutput:
        """ATIF-native evaluation lane for trajectory scoring."""

        async def wrapped(sample: AtifEvalSample) -> EvalOutputItem:
            async with self.semaphore:
                return await self.evaluate_atif_item(sample)

        output_items = await asyncio.gather(*[wrapped(sample) for sample in atif_samples])
        numeric_scores = [item.score for item in output_items if isinstance(item.score, int | float)]
        avg_score = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None
        return EvalOutput(average_score=avg_score, eval_output_items=output_items)


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
    evaluator_info = EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description="Trajectory Evaluator")
    if config.enable_atif_evaluator:
        evaluator_info.evaluate_atif_fn = evaluator.evaluate_atif_fn
    yield evaluator_info
