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
from collections.abc import Callable

from langchain_classic.output_parsers import ResponseSchema
from langchain_classic.output_parsers import StructuredOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from nat.atif import ATIFContentPart
from nat.atif import ATIFTrajectory
from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_evaluator
from nat.data_models.component_ref import LLMRef
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.plugins.eval.data_models.evaluator_io import EvalOutput
from nat.plugins.eval.data_models.evaluator_io import EvalOutputItem
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSampleList
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator
from nat.utils.atif_message_utils import content_part_to_text
from nat.utils.atif_message_utils import message_to_text
from nat.utils.atif_message_utils import trajectory_to_user_input

logger = logging.getLogger(__name__)


class TunableRagEvaluatorConfig(EvaluatorBaseConfig, name="tunable_rag_evaluator"):
    """Configuration for tunable RAG evaluator."""

    llm_name: LLMRef = Field(description="Name of the judge LLM")
    llm_retry_control_params: dict | None = Field(description="Parameters to control LLM retry behavior", default=None)
    judge_llm_prompt: str = Field(description="LLM prompt for the judge LLM")
    default_scoring: bool = Field(description="Whether to use default scoring", default=False)
    default_score_weights: dict = Field(
        default={
            "coverage": 0.5,
            "correctness": 0.3,
            "relevance": 0.2,
        },
        description="Weights for different scoring components when using default scoring",
    )
    enable_atif_evaluator: bool = Field(
        default=False,
        description="Enable ATIF-native tunable RAG evaluator lane. Disabled by default during migration.",
    )


def evaluation_prompt(judge_llm_prompt: str,
                      question: str,
                      answer_description: str,
                      generated_answer: str,
                      format_instructions: str,
                      default_scoring: bool) -> str:
    """Generate a prompt for the judge LLM."""
    default_scoring_instructions = (
        "The coverage score is a measure of how well the generated answer covers the critical aspects mentioned in the "
        "expected answer. A low coverage score indicates that the generated answer misses critical aspects of the "
        "expected answer. A middle coverage score indicates that the generated answer covers some of the must-haves "
        "of the expected answer but lacks other details. A high coverage score indicates that all of the expected "
        "aspects are present in the generated answer. The correctness score is a measure of how well the generated "
        "answer matches the expected answer. A low correctness score indicates that the generated answer is incorrect "
        "or does not match the expected answer. A middle correctness score indicates that the generated answer is "
        "correct but lacks some details. A high correctness score indicates that the generated answer is exactly the "
        "same as the expected answer. The relevance score is a measure of how well the generated answer is relevant "
        "to the question. A low relevance score indicates that the generated answer is not relevant to the question. "
        "A middle relevance score indicates that the generated answer is somewhat relevant to the question. A high "
        "relevance score indicates that the generated answer is exactly relevant to the question. The reasoning is a "
        "1-2 sentence explanation for the scoring.")

    default_eval_prompt = ("You are an intelligent assistant that responds strictly in JSON format. "
                           f"Judge based on the following scoring rubric: {default_scoring_instructions}"
                           f"{judge_llm_prompt}\n"
                           f"{format_instructions}\n"
                           f"Here is the user's query: {question}"
                           f"Here is the description of the expected answer: {answer_description}"
                           f"Here is the generated answer: {generated_answer}")

    eval_prompt = (f"You are an intelligent assistant that responds strictly in JSON format. {judge_llm_prompt}\n"
                   f"{format_instructions}\n"
                   f"Here is the user's query: {question}"
                   f"Here is the description of the expected answer: {answer_description}"
                   f"Here is the generated answer: {generated_answer}")
    return eval_prompt if not default_scoring else default_eval_prompt


def runnable_with_retries(original_fn: Callable, llm_retry_control_params: dict | None = None):
    """Wrap a runnable with retry controls."""
    runnable = RunnableLambda(original_fn)

    if llm_retry_control_params is None:
        llm_retry_control_params = {
            "stop_after_attempt": 3,
            "initial_backoff_delay_seconds": 1,
            "has_exponential_jitter": True,
        }

    if llm_retry_control_params["has_exponential_jitter"] is None:
        llm_retry_control_params["has_exponential_jitter"] = True
    if llm_retry_control_params["stop_after_attempt"] is None:
        llm_retry_control_params["stop_after_attempt"] = 3
    if llm_retry_control_params["initial_backoff_delay_seconds"] is None:
        llm_retry_control_params["initial_backoff_delay_seconds"] = 1

    return runnable.with_retry(
        retry_if_exception_type=(Exception, ),
        wait_exponential_jitter=llm_retry_control_params["has_exponential_jitter"],
        stop_after_attempt=llm_retry_control_params["stop_after_attempt"],
        exponential_jitter_params={"initial": llm_retry_control_params["initial_backoff_delay_seconds"]},
    )


class TunableRagEvaluator(BaseEvaluator):
    """Tunable RAG evaluator with customizable judge prompt."""

    def __init__(self,
                 llm: BaseChatModel,
                 judge_llm_prompt: str,
                 llm_retry_control_params: dict | None,
                 max_concurrency: int,
                 default_scoring: bool,
                 default_score_weights: dict):
        super().__init__(max_concurrency=max_concurrency)
        self.llm = llm
        self.judge_llm_prompt = judge_llm_prompt
        self.llm_retry_control_params = llm_retry_control_params
        self.default_scoring = default_scoring
        self.default_score_weights = default_score_weights if default_score_weights else {
            "coverage": 1 / 3,
            "correctness": 1 / 3,
            "relevance": 1 / 3,
        }

    async def _evaluate_item_core(self, item_id, question: str, answer_description: str,
                                  generated_answer: str) -> EvalOutputItem:
        score = 0.0

        default_evaluation_schema = [
            ResponseSchema(name="coverage_score",
                           description="Score for coverage of critical aspects in the expected answer.",
                           type="float"),
            ResponseSchema(name="correctness_score",
                           description="Score for generated answer correctness compared to expected answer.",
                           type="float"),
            ResponseSchema(name="relevance_score", description="Score for relevance to the question.", type="float"),
            ResponseSchema(name="reasoning", description="1-2 summarized sentences for the scoring.", type="string"),
        ]
        custom_evaluation_schema = [
            ResponseSchema(name="score", description="Score for the generated answer.", type="float"),
            ResponseSchema(name="reasoning", description="1-2 sentence reasoning for the score.", type="string"),
        ]

        evaluation_schema = default_evaluation_schema if self.default_scoring else custom_evaluation_schema
        response_parser = StructuredOutputParser.from_response_schemas(evaluation_schema)
        format_instructions = response_parser.get_format_instructions()

        eval_prompt = evaluation_prompt(judge_llm_prompt=self.judge_llm_prompt,
                                        question=question,
                                        answer_description=answer_description,
                                        generated_answer=generated_answer,
                                        format_instructions=format_instructions,
                                        default_scoring=self.default_scoring)

        messages = [SystemMessage(content="You must respond only in JSON format."), HumanMessage(content=eval_prompt)]
        response = await runnable_with_retries(self.llm.ainvoke, self.llm_retry_control_params).ainvoke(messages)

        coverage_score = 0.0
        correctness_score = 0.0
        relevance_score = 0.0
        reasoning = "Error in evaluator from parsing judge LLM response."

        try:
            parsed_response = response_parser.parse(response.content)
            if self.default_scoring:
                try:
                    coverage_score = parsed_response["coverage_score"]
                    correctness_score = parsed_response["correctness_score"]
                    relevance_score = parsed_response["relevance_score"]
                    reasoning = parsed_response["reasoning"]
                except KeyError as e:
                    logger.exception("Missing required keys in default scoring response: %s",
                                     ", ".join(str(arg) for arg in e.args))
                    reasoning = ("Error in evaluator from parsing judge LLM response. "
                                 f"Missing required key(s): {', '.join(str(arg) for arg in e.args)}")

                coverage_weight = self.default_score_weights.get("coverage", 1 / 3)
                correctness_weight = self.default_score_weights.get("correctness", 1 / 3)
                relevance_weight = self.default_score_weights.get("relevance", 1 / 3)
                total_weight = coverage_weight + correctness_weight + relevance_weight

                coverage_weight = coverage_weight / total_weight
                correctness_weight = correctness_weight / total_weight
                relevance_weight = relevance_weight / total_weight

                if round(coverage_weight + correctness_weight + relevance_weight, 2) != 1:
                    logger.warning("The sum of default score weights is not 1. The weights will be normalized.")
                    renorm = coverage_weight + correctness_weight + relevance_weight
                    coverage_weight = coverage_weight / renorm
                    correctness_weight = correctness_weight / renorm
                    relevance_weight = relevance_weight / renorm

                score = (coverage_weight * coverage_score + correctness_weight * correctness_score +
                         relevance_weight * relevance_score)
            else:
                try:
                    score = parsed_response["score"]
                    reasoning = parsed_response["reasoning"]
                except KeyError as e:
                    logger.error("Missing required keys in custom scoring response: %s",
                                 ", ".join(str(arg) for arg in e.args))
                    reasoning = ("Error in evaluator from parsing judge LLM response. "
                                 f"Missing required key(s): {', '.join(str(arg) for arg in e.args)}")
                    raise
        except (KeyError, ValueError) as e:
            logger.exception("Error parsing judge LLM response: %s", e)
            score = 0.0
            reasoning = "Error in evaluator from parsing judge LLM response."

        if self.default_scoring:
            reasoning_obj = {
                "question": question,
                "answer_description": answer_description,
                "generated_answer": generated_answer,
                "score_breakdown": {
                    "coverage_score": coverage_score,
                    "correctness_score": correctness_score,
                    "relevance_score": relevance_score,
                },
                "reasoning": reasoning,
            }
        else:
            reasoning_obj = {
                "question": question,
                "answer_description": answer_description,
                "generated_answer": generated_answer,
                "reasoning": reasoning,
            }

        return EvalOutputItem(id=item_id, score=score, reasoning=reasoning_obj)

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        question = str(item.input_obj) if item.input_obj is not None else ""
        answer_description = str(item.expected_output_obj) if item.expected_output_obj is not None else ""
        generated_answer = str(item.output_obj) if item.output_obj is not None else ""
        return await self._evaluate_item_core(item.id, question, answer_description, generated_answer)

    @staticmethod
    def _content_part_to_text(part: ATIFContentPart) -> str:
        return content_part_to_text(part)

    @classmethod
    def _message_to_text(cls, message: str | list[ATIFContentPart] | None) -> str:
        return message_to_text(message)

    @classmethod
    def _trajectory_to_user_input(cls, trajectory: ATIFTrajectory) -> str:
        return trajectory_to_user_input(trajectory)

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        question = self._trajectory_to_user_input(sample.trajectory)
        answer_description = str(sample.expected_output_obj) if sample.expected_output_obj is not None else ""
        generated_answer = str(sample.output_obj) if sample.output_obj is not None else ""
        return await self._evaluate_item_core(sample.item_id, question, answer_description, generated_answer)

    async def evaluate_atif_fn(self, atif_samples: AtifEvalSampleList) -> EvalOutput:

        async def wrapped(sample: AtifEvalSample) -> EvalOutputItem:
            async with self.semaphore:
                return await self.evaluate_atif_item(sample)

        output_items = await asyncio.gather(*[wrapped(sample) for sample in atif_samples])
        numeric_scores = [item.score for item in output_items if isinstance(item.score, int | float)]
        avg_score = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None
        return EvalOutput(average_score=avg_score, eval_output_items=output_items)


@register_evaluator(config_type=TunableRagEvaluatorConfig)
async def register_tunable_rag_evaluator(config: TunableRagEvaluatorConfig, builder: EvalBuilder):
    """Register tunable RAG evaluator."""
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    evaluator = TunableRagEvaluator(llm=llm,
                                    judge_llm_prompt=config.judge_llm_prompt,
                                    llm_retry_control_params=config.llm_retry_control_params,
                                    max_concurrency=builder.get_max_concurrency(),
                                    default_scoring=config.default_scoring,
                                    default_score_weights=config.default_score_weights)
    evaluator_info = EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description="Tunable RAG Evaluator")
    if config.enable_atif_evaluator:
        evaluator_info.evaluate_atif_fn = evaluator.evaluate_atif_fn
    yield evaluator_info
