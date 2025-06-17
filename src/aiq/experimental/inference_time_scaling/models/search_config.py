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

from aiq.data_models.component_ref import LLMRef
from aiq.data_models.its_strategy import ITSStrategyBaseConfig


class SingleShotMultiPlanConfig(ITSStrategyBaseConfig, name="single_shot_multi_plan"):
    num_plans: int = Field(default=4, description="Number of plans to generate.")
    max_temperature: float = Field(default=1.0,
                                   description="Maximum temperature to use for sampling when generating plans. "
                                   "This can help control the randomness of the generated plans.")
    min_temperature: float = Field(default=0.5,
                                   description="Minimum temperature to use for sampling when generating plans. "
                                   "This can help control the randomness of the generated plans.")
    # If strategy is provided, LLM must be
    planning_llm: LLMRef | typing.Any | None = Field(
        default=None,
        description="The LLM to use for planning. This can be a callable or an "
        "instance of an LLM client.")

    planning_template: str = Field(
        default=("You are an expert reasoning model task with creating a detailed execution plan"
                 " for a system that has the following information to get the result of a given input:\n\n"
                 "**System Information:**\n {context}"
                 "**Input:** \n{prompt}\n\n"
                 "An example plan could look like this:\n\n"
                 "1. Call tool A with input X\n"
                 "2. Call tool B with input Y\n"
                 "3. Interpret the output of tool A and B\n"
                 "4. Return the final result"
                 "\n\nBegin the final plan with PLAN:\n"),
        description="The template to use for generating plans.")

    @model_validator(mode="before")
    def validate_strategies(cls, values: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Ensure that the required LLMs are provided based on the selected strategies.
        """
        # Validate planning strategy: planning_llm must be provided if planning_strategy is set
        if values.get('planning_llm') is None:
            raise ValueError('planning_llm must be provided when planning_strategy is set.')

        return values


class MultiLLMPlanConfig(ITSStrategyBaseConfig, name="multi_llm_plan"):
    """Configuration for a 'multi LLM plan generation' strategy."""
    llms: list[LLMRef] = Field(
        default_factory=list,
        description="list of LLMs to use for plan generation. Each LLM can generate one or more plans.")
    plans_per_llm: int = Field(default=2, description="Number of plans each LLM should generate.")
    max_temperature: float = Field(default=1.0,
                                   description="Maximum temperature to use for sampling when generating plans. "
                                   "This can help control the randomness of the generated plans.")
    min_temperature: float = Field(default=0.5,
                                   description="Minimum temperature to use for sampling when generating plans. "
                                   "This can help control the randomness of the generated plans.")
    planning_template: str = Field(
        default=("You are an expert reasoning model task with creating a detailed execution plan"
                 " for a system that has the following information to get the result of a given input:\n\n"
                 "**System Information:**\n {context}"
                 "**Input:** \n{prompt}\n\n"
                 "An example plan could look like this:\n\n"
                 "1. Call tool A with input X\n"
                 "2. Call tool B with input Y\n"
                 "3. Interpret the output of tool A and B\n"
                 "4. Return the final result"
                 "\n\nBegin the final plan with PLAN:\n"),
        description="The template to use for generating plans.")

    @model_validator(mode="before")
    def validate_multi_llm_strategies(cls, values: dict) -> dict:
        if not values.get('llms'):
            raise ValueError('Must provide at least one LLMRef in `llms` for multi-LLM strategy.')
        return values


class MultiQueryRetrievalSearchConfig(ITSStrategyBaseConfig, name="multi_query_retrieval_search"):
    """
    Configuration for the MultiQueryRetrievalSearch strategy.
    This strategy generates multiple new 'ITSItem's per original item,
    each containing a differently phrased or re-focused version of the original task.
    """
    llms: list[LLMRef] = Field(default_factory=list,
                               description="list of LLM references to use for generating diverse queries.")

    query_generation_template: str = Field(
        default=("You are an expert at re-framing a user's query to encourage new solution paths. "
                 "Given the task description and an optional motivation, produce a short alternative query "
                 "that addresses the same task from a different angle. By generating multiple "
                 "perspectives on the task, your goal is to help "
                 "the user overcome some of the limitations of distance-based similarity search.\n\n"
                 "Task: {task}\n"
                 "Motivation: {motivation}\n\n"
                 "Output a concise new query statement below. Only output the revised query and nothing else.\n"),
        description="Prompt template for rewriting the task from a different perspective.")

    @model_validator(mode="before")
    def validate_llms(cls, values):
        if not values.get('llms'):
            raise ValueError("At least one LLMRef must be provided for multi_query_retrieval_search.")
        return values
