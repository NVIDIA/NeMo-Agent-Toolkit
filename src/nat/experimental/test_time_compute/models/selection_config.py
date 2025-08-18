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
# limitations under the License..

import typing

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import LLMRef
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig


class LLMBasedPlanSelectionConfig(TTCStrategyBaseConfig, name="llm_based_plan_selection"):
    """
    Configuration for LLMBasedSelection.
    """
    selection_llm: LLMRef | typing.Any | None = Field(
        default=None,
        description="The LLM to use for selecting the best plan. This can be an instance of an LLM client.")

    selection_template: str = Field(
        default=("You are tasked with selecting the best plan from several alternative plans."
                 " Review the following plans and their feedback carefully to select the most "
                 "comprehensive, efficient, and effective one."
                 "The plan is for an agent system with the following objective and context:\n\n"
                 "{context}\n\n"
                 "The system is asked to achieve the following goal:\n\n"
                 "{original_prompt}\n\n"
                 "The generated plans are as follows."
                 "\n\n{plans}"
                 "\n\nBased on your analysis, which plan (numbered 1 and onwards) is the best? "
                 "Provide a thorough explanation of your choice,"
                 " referencing specific strengths from the feedback and how they outweigh any weaknesses."
                 "Make sure you begin your choice of selected plan with the words 'SELECTED PLAN:' "
                 "followed by the plan number."),
        description="The template to use for selecting the best plan. This should guide the LLM on how to evaluate "
        "the plans and select the best one. Ensure it is clear and concise.")

    @model_validator(mode="before")
    def validate_strategies(cls, values: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Ensure that the selection_llm is provided when using LLMBasedSelection.
        """
        if values.get('selection_llm') is None:
            raise ValueError('selection_llm must be provided when'
                             ' selection_strategy is set to LLM_BASED_PLAN_SELECTION.')

        return values


class LLMBasedAgentOutputSelectionConfig(TTCStrategyBaseConfig, name="llm_based_agent_output_selection"):
    """
    Configuration for LLMBasedSelection.
    """
    selection_llm: LLMRef | typing.Any | None = Field(
        default=None,
        description="The LLM to use for selecting the best plan. This can be an instance of an LLM client.")

    selection_template: str = Field(
        default=("You are tasked with selecting the best output from several output."
                 "The outputs are from an agent system whose object and input will be provided below.\n "
                 "Review all the outputs and select one that fits the best. You will do this by "
                 "looking at how many outputs have the same classification. Chose the one that has the most. "
                 "Of the ones that have the same classification, choose the one that is the most complete, "
                 "clear, and comprehensive. The objective of the agent is: \n"
                 "{objective}\n\n"
                 "\n\nThe agent is asked to achieve the following goal:\n\n"
                 "{input}\n\n"
                 "The generated outputs are as follows."
                 "\n\n{results}"
                 "\n\nBased on your analysis, which plan (numbered 1 and onwards) is the best? "
                 "Provide a thorough explanation of your choice,"
                 " referencing specific strengths from the feedback and how they outweigh any weaknesses."
                 "You must ALWAYS select an option, even if the options are identical or similar. "
                 "Make sure you begin your choice of selected plan with the words 'SELECTED ITEM:' "
                 "followed by the plan number."),
        description="The template to use for selecting the best output. This should guide the LLM on how to evaluate "
        "the outputs and select the best one. Ensure it is clear and concise. Must contain {objective}, "
        "{input}, and {results} ")

    @model_validator(mode="before")
    def validate_strategies(cls, values: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Ensure that the selection_llm is provided when using LLMBasedSelection.
        """
        if values.get('selection_llm') is None:
            raise ValueError('selection_llm must be provided when '
                             'selection_strategy is set to LLM_BASED_AGENT_OUTPUT_SELECTION.')

        return values


class LLMBasedOutputMergingConfig(TTCStrategyBaseConfig, name="llm_based_agent_output_merging"):
    """
    Configuration for LLMBasedSelection.
    """
    selection_llm: LLMRef | typing.Any | None = Field(
        default=None,
        description="The LLM to use for selecting the best plan. This can be an instance of an LLM client.")

    selection_template: str = Field(
        default=("You are tasked with merging the output of an agent systems that produces {pipeline_type}."
                 "The outputs are from an agent system whose objective and input will be provided below.\n "
                 "Review all the outputs, please combine them all into one output, keeping with the intended structure "
                 "generated by the outputs and general tone. Capture the important pieces of each of the outputs "
                 "to create comprehensive output that achieves the input and objective. "
                 "The objective of the agent is: \n"
                 "{objective}\n\n"
                 "\n\nThe agent is asked to achieve the following goal:\n\n"
                 "{input}\n\n"
                 "The generated outputs are as follows."
                 "\n\n{results}"
                 "\n\n Make sure you begin your updated output with the words 'MERGED OUTPUT:' "),
        description="The template to use for selecting the best output. This should guide the LLM on how to evaluate "
        "the outputs and select the best one. Ensure it is clear and concise. Must contain {objective}, "
        "{input}, and {results} ")

    @model_validator(mode="before")
    def validate_strategies(cls, values: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Ensure that the selection_llm is provided when using LLMBasedSelection.
        """
        if values.get('selection_llm') is None:
            raise ValueError('selection_llm must be provided when '
                             'selection_strategy is set to LLM_BASED_AGENT_OUTPUT_SELECTION.')

        return values


class ThresholdSelectionConfig(TTCStrategyBaseConfig, name="threshold_selection"):
    """
    Configuration for a selection strategy that keeps only the items
    whose scores exceed a specified threshold.
    """
    threshold: float = Field(default=5.0, description="Only keep TTCItems with score >= this value.")


class BestOfNSelectionConfig(TTCStrategyBaseConfig, name="best_of_n_selection"):
    """
    Configuration for Best of N Selection
    """
    pass
