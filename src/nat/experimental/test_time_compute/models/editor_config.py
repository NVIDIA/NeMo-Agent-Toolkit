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

import typing

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import LLMRef
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig


class LLMAsAJudgeEditorConfig(TTCStrategyBaseConfig, name="llm_as_a_judge_editor"):
    """
    Configuration for the LLMAsAJudgeEditor.
    """
    num_feedback: int = Field(default=10,
                              description="Number of feedback items to generate for each plan during editing. "
                              "This can help in refining the plans based on feedback.")

    # If strategy is provided, LLM must be
    editing_llm: LLMRef | typing.Any | None = Field(
        default=None,
        description="The LLM to use for editing the plans. This can be a callable or an instance of an LLM client.")

    # If strategy is LLM_AS_A_JUDGE, ensure that the feedback_llm is provided.
    feedback_llm: LLMRef | typing.Any | None = Field(default=None,
                                                     description="The LLM to use for generating feedback on the plans."
                                                     " This can be a callable or an instance of an LLM client.")

    editor_template: str = Field(default=(
        "You are an expert at improving execution plans. You will be given a plan and feedback on that plan."
        " Your task is to create an improved version of the plan that addresses the feedback "
        "while maintaining its strengths.\n\n"
        "Here is the context:\n\n"
        "{context}\n\n"
        "**Input:** \n{original_prompt}\n\n"
        "**Original Plan:**\n{plan}\n\n"
        "**Feedback on the Plan:**\n{feedback}\n\n"
        "Please provide an improved version of the plan that addresses"
        " the feedback points. Maintain the same structure and "
        "step-by-step format, but enhance the content. Do not include explanations of your changes, just provide the "
        "improved plan directly:\n\n"
        "Begin the final improve plan with 'EDITED PLAN:'"),
                                 description="The template to use for editing the planning items based on feedback.")

    feedback_template: str = Field(
        default=("You are an expert at evaluating execution plans. You will be given a plan and "
                 "need to provide {num_feedback} "
                 "specific points of feedback about its strengths and weaknesses.\n\n"
                 "Your feedback should cover aspects like:\n"
                 "- Comprehensiveness of the plan\n"
                 "- Logical flow and sequencing\n"
                 "- Appropriate use of available tools\n"
                 "- Potential edge cases or failure points\n"
                 "- Efficiency and optimization opportunities\n\n"
                 "Here is the context and plan to evaluate:\n\n"
                 "{context}\n\n"
                 "**Objective:** \n{original_prompt}\n\n"
                 "**Plan to Evaluate:**\n{plan}\n\n"
                 "Please provide exactly {num_feedback} numbered points of feedback, including "
                 "both strengths and areas for improvement. Begin the feedback with 'FEEDBACK:' and provide"
                 "{num_feedback} specific feedback points."),
        description="The template to use for generating feedback for each planning item.")

    @model_validator(mode="before")
    def validate_strategies(cls, values: dict[str, typing.Any]) -> dict[str, typing.Any]:

        if values.get('editing_llm') is None:
            raise ValueError('editing_llm must be provided when editing_strategy is set.')
        # If editing strategy is LLM_AS_A_JUDGE, feedback_llm must also be provided
        if (values.get('feedback_llm') is None):
            raise ValueError('feedback_llm must be provided when editing_strategy is LLM_AS_A_JUDGE.')

        return values


class IterativePlanRefinementConfig(TTCStrategyBaseConfig, name="iterative_plan_refinement"):
    """Configuration for an 'iterative plan refinement' strategy."""
    editor_llm: LLMRef | typing.Any | None = Field(
        default=None, description="The LLM to use for generating and refining the plan across multiple iterations.")
    num_iterations: int = Field(default=3, description="How many refinement steps to perform.")
    refinement_template: str = Field(
        default=("You have the current plan:\n{current_plan}\n\n"
                 "The plan was generated to achieve the following objective:\n{original_prompt}\n\n"
                 "Using an agent system with the following description:\n{context}\n\n"
                 "Refine or improve it to achieve the objective better."
                 "Output the updated plan, beginning with:\nEDITED PLAN:\n"),
        description="Prompt used in each iteration to refine the plan.")

    @model_validator(mode="before")
    def validate_iterative_strategies(cls, values: dict) -> dict:
        if not values.get('editor_llm'):
            raise ValueError('planning_llm must be provided for iterative plan refinement.')
        if values.get('num_iterations', 0) < 1:
            raise ValueError('num_iterations must be >= 1 for iterative plan refinement.')
        return values


class MotivationAwareSummarizationConfig(TTCStrategyBaseConfig, name="motivation_aware_editing"):
    """
    Configuration for the MotivationAwareSummarization strategy.
    """
    editor_llm: LLMRef | typing.Any | None = Field(
        default=None,
        description="The LLM to use for editing the plans. This can be a callable or an instance of an LLM client.")

    editor_template: str = Field(
        default=("You are an expert at summarizing key information from relevant documents based on an input task"
                 "and motivation. Given a task and motivation, and documents, your task is to create a concise "
                 "a summarized response to the task and motivation grounded in the documents .\n\n"
                 "Here is the task:\n\n"
                 "{task}\n\n"
                 "Here is the motivation:\n\n"
                 "{motivation}\n\n"
                 "and here are the documents:\n\n"
                 "{output}\n\n"
                 "Please respond with a concise summary that addresses the task and motivation, in at most one"
                 "or two sentences. Do not include any other output except the summary. "),
        description="The template to use for summarizing documents.")
