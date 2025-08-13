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

import asyncio
import logging
from typing import Callable

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from tqdm import tqdm

from nat.eval.evaluator.base_evaluator import BaseEvaluator
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem

logger = logging.getLogger(__name__)

# pylint: disable=line-too-long
# flake8: noqa: E501


def evaluation_prompt(judge_llm_prompt: str,
                      question: str,
                      answer_description: str,
                      generated_answer: str,
                      format_instructions: str,
                      default_scoring: bool):
    """
    This function generates a prompt for the judge LLM to evaluate the generated answer.
    """

    DEFAULT_SCORING_INSTRUCTIONS = """
    The coverage score is a measure of how well the generated answer covers the critical aspects mentioned in the expected answer. A low coverage score indicates that the generated answer misses critical aspects of the expected answer. A middle coverage score indicates that the generated answer covers some of the must-haves of the expected answer but lacks other details. A high coverage score indicates that all of the expected aspects are present in the generated answer.
    The correctness score is a measure of how well the generated answer matches the expected answer. A low correctness score indicates that the generated answer is incorrect or does not match the expected answer. A middle correctness score indicates that the generated answer is correct but lacks some details. A high correctness score indicates that the generated answer is exactly the same as the expected answer.
    The relevance score is a measure of how well the generated answer is relevant to the question. A low relevance score indicates that the generated answer is not relevant to the question. A middle relevance score indicates that the generated answer is somewhat relevant to the question. A high relevance score indicates that the generated answer is exactly relevant to the question.
    The reasoning is a 1-2 sentence explanation for the scoring.
    """

    DEFAULT_EVAL_PROMPT = (f"You are an intelligent assistant that responds strictly in JSON format."
                           f"Judge based on the following scoring rubric: {DEFAULT_SCORING_INSTRUCTIONS}"
                           f"{judge_llm_prompt}\n"
                           f"{format_instructions}\n"
                           f"Here is the user's query: {question}"
                           f"Here is the description of the expected answer: {answer_description}"
                           f"Here is the generated answer: {generated_answer}")

    EVAL_PROMPT = (f"You are an intelligent assistant that responds strictly in JSON format. {judge_llm_prompt}\n"
                   f"{format_instructions}\n"
                   f"Here is the user's query: {question}"
                   f"Here is the description of the expected answer: {answer_description}"
                   f"Here is the generated answer: {generated_answer}")

    return EVAL_PROMPT if not default_scoring else DEFAULT_EVAL_PROMPT


def runnable_with_retries(original_fn: Callable, llm_retry_control_params: dict | None = None):
    runnable = RunnableLambda(original_fn)

    if llm_retry_control_params is None:
        llm_retry_control_params = {
            "stop_after_attempt": 3, "initial_backoff_delay_seconds": 1, "has_exponential_jitter": True
        }

    if llm_retry_control_params["has_exponential_jitter"] is None:
        llm_retry_control_params["has_exponential_jitter"] = True
    if llm_retry_control_params["stop_after_attempt"] is None:
        llm_retry_control_params["stop_after_attempt"] = 3
    if llm_retry_control_params["initial_backoff_delay_seconds"] is None:
        llm_retry_control_params["initial_backoff_delay_seconds"] = 1

    # Add retry logic with exponential backoff and jitter
    return runnable.with_retry(
        retry_if_exception_type=(Exception, ),  # Retry on any error
        wait_exponential_jitter=llm_retry_control_params["has_exponential_jitter"],  # Add jitter to exponential backoff
        stop_after_attempt=llm_retry_control_params["stop_after_attempt"],
        exponential_jitter_params={"initial": llm_retry_control_params["initial_backoff_delay_seconds"]
                                   }  # Optional: set initial backoff (seconds)
    )


class TunableRagEvaluator(BaseEvaluator):
    '''Tunable RAG evaluator class with customizable LLM prompt for scoring.'''

    def __init__(self,
                 llm: BaseChatModel,
                 judge_llm_prompt: str,
                 llm_retry_control_params: dict | None,
                 max_concurrency: int,
                 default_scoring: bool,
                 default_score_weights: dict):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating RAG")
        self.llm = llm
        self.judge_llm_prompt = judge_llm_prompt
        self.llm_retry_control_params = llm_retry_control_params
        self.default_scoring = default_scoring
        # Use user-provided weights if available; otherwise, set equal weights for each score
        self.default_score_weights = default_score_weights if default_score_weights else {
            "coverage": 1 / 3, "correctness": 1 / 3, "relevance": 1 / 3
        }

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Compute RAG evaluation for an individual item and return EvalOutputItem"""
        question = item.input_obj
        answer_description = item.expected_output_obj
        generated_answer = item.output_obj

        # Call judge LLM to generate score
        score = 0.0

        default_evaluation_schema = [
            ResponseSchema(
                name="coverage_score",
                description="Score for the coverage of all critical aspects mentioned in the expected answer. Ex. 0.5",
                type="float"),
            ResponseSchema(
                name="correctness_score",
                description="Score for the accuracy of the generated answer compared to the expected answer. Ex. 0.5",
                type="float"),
            ResponseSchema(name="relevance_score",
                           description="Score for the relevance of the generated answer to the question. Ex. 0.5",
                           type="float"),
            ResponseSchema(
                name="reasoning",
                description=
                "1-2 summarized sentences of reasoning for the scores. Ex. 'The generated answer covers all critical aspects mentioned in the expected answer, is correct, and is relevant to the question.'",
                type="string"),
        ]

        custom_evaluation_schema = [
            ResponseSchema(name="score", description="Score for the generated answer. Ex. 0.5", type="float"),
            ResponseSchema(
                name="reasoning",
                description=
                "1-2 sentence reasoning for the score. Ex. 'The generated answer is exactly the same as the description of the expected answer.'",
                type="string"),
        ]

        if self.default_scoring:
            evaluation_schema = default_evaluation_schema
        else:
            evaluation_schema = custom_evaluation_schema

        llm_input_response_parser = StructuredOutputParser.from_response_schemas(evaluation_schema)
        format_instructions = llm_input_response_parser.get_format_instructions()

        eval_prompt = evaluation_prompt(judge_llm_prompt=self.judge_llm_prompt,
                                        question=question,
                                        answer_description=answer_description,
                                        generated_answer=generated_answer,
                                        format_instructions=format_instructions,
                                        default_scoring=self.default_scoring)

        messages = [SystemMessage(content="You must respond only in JSON format."), HumanMessage(content=eval_prompt)]

        response = await runnable_with_retries(self.llm.ainvoke, self.llm_retry_control_params).ainvoke(messages)

        # Initialize default values to handle service errors
        coverage_score = 0.0
        correctness_score = 0.0
        relevance_score = 0.0
        reasoning = "Error in evaluator from parsing judge LLM response."

        try:
            parsed_response = llm_input_response_parser.parse(response.content)
            if self.default_scoring:
                try:
                    coverage_score = parsed_response["coverage_score"]
                    correctness_score = parsed_response["correctness_score"]
                    relevance_score = parsed_response["relevance_score"]
                    reasoning = parsed_response["reasoning"]
                except KeyError as e:
                    logger.error("Missing required keys in default scoring response: %s",
                                 ", ".join(str(arg) for arg in e.args))
                    reasoning = f"Error in evaluator from parsing judge LLM response. Missing required key(s): {', '.join(str(arg) for arg in e.args)}"

                coverage_weight = self.default_score_weights.get("coverage", 1 / 3)
                correctness_weight = self.default_score_weights.get("correctness", 1 / 3)
                relevance_weight = self.default_score_weights.get("relevance", 1 / 3)

                # Calculate score
                total_weight = coverage_weight + correctness_weight + relevance_weight
                coverage_weight = coverage_weight / total_weight
                correctness_weight = correctness_weight / total_weight
                relevance_weight = relevance_weight / total_weight

                if round(coverage_weight + correctness_weight + relevance_weight, 2) != 1:
                    logger.warning("The sum of the default score weights is not 1. The weights will be normalized.")
                    coverage_weight = coverage_weight / (coverage_weight + correctness_weight + relevance_weight)
                    correctness_weight = correctness_weight / (coverage_weight + correctness_weight + relevance_weight)
                    relevance_weight = relevance_weight / (coverage_weight + correctness_weight + relevance_weight)

                score = (coverage_weight * coverage_score + correctness_weight * correctness_score +
                         relevance_weight * relevance_score)

            else:
                try:
                    score = parsed_response["score"]
                    reasoning = parsed_response["reasoning"]
                except KeyError as e:
                    logger.error("Missing required keys in custom scoring response: %s",
                                 ", ".join(str(arg) for arg in e.args))
                    reasoning = f"Error in evaluator from parsing judge LLM response. Missing required key(s): {', '.join(str(arg) for arg in e.args)}"
                    raise
        except (KeyError, ValueError) as e:
            logger.error("Error parsing judge LLM response: %s", e)
            score = 0.0
            reasoning = "Error in evaluator from parsing judge LLM response."

        if self.default_scoring:
            reasoning = {
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
            reasoning = {
                "question": question,
                "answer_description": answer_description,
                "generated_answer": generated_answer,
                "reasoning": reasoning
            }

        return EvalOutputItem(id=item.id, score=score, reasoning=reasoning)
