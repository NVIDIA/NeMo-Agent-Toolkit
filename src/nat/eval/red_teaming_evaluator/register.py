# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pydantic import Field

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_evaluator
from nat.data_models.component_ref import LLMRef
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.eval.red_teaming_evaluator.evaluate import ReductionStrategy
from nat.eval.red_teaming_evaluator.filter_conditions import IntermediateStepsFilterCondition


class RedTeamingEvaluatorConfig(EvaluatorBaseConfig, name="red_teaming_evaluator"):
    '''Configuration for red teaming evaluator'''
    llm_name: LLMRef = Field(description="Name of the judge LLM")
    llm_retry_control_params: dict | None = Field(description="Parameters to control LLM retry behavior", default=None)
    judge_llm_prompt: str = Field(description="LLM prompt for the judge LLM")
    filter_conditions: list[IntermediateStepsFilterCondition] = Field(
        description="List of filter conditions for selecting intermediate steps to evaluate")
    reduction_strategy: str = Field(
        description="Strategy to combine scores from multiple steps ('first', 'max', 'last')", default="last")
    scenario_specific_instructions: str | None = Field(
        description="Optional scenario-specific instructions for evaluating the output", default=None)


@register_evaluator(config_type=RedTeamingEvaluatorConfig)
async def register_red_teaming_evaluator(config: RedTeamingEvaluatorConfig, builder: EvalBuilder):
    '''Register red teaming evaluator'''
    from .evaluate import RedTeamingEvaluator

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    evaluator = RedTeamingEvaluator(llm,
                                    config.judge_llm_prompt,
                                    config.filter_conditions,
                                    config.llm_retry_control_params,
                                    builder.get_max_concurrency(),
                                    ReductionStrategy(config.reduction_strategy),
                                    config.scenario_specific_instructions)

    yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description="Red Teaming Evaluator")
