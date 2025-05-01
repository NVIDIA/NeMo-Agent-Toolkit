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

from pydantic import Field

from aiq.builder.builder import EvalBuilder
from aiq.builder.evaluator import EvaluatorInfo
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_evaluator
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.evaluator import EvaluatorBaseConfig


class TunableRagEvaluatorConfig(EvaluatorBaseConfig, name="tunable_rag_evaluator"):
    '''Configuration for tunable RAG evaluator'''
    llm_name: LLMRef = Field(description="Name of the judge LLM")
    judge_llm_prompt: str = Field(description="LLM prompt for the judge LLM")
    default_scoring: bool = Field(description="Whether to use default scoring", default=False)
    default_score_weights: dict = Field(
        default={
            "coverage": 0.5, "correctness": 0.3, "relevance": 0.2
        },
        description="Weights for the different scoring components when using default scoring")


@register_evaluator(config_type=TunableRagEvaluatorConfig)
async def register_tunable_rag_evaluator(config: TunableRagEvaluatorConfig, builder: EvalBuilder):
    '''Register tunable RAG evaluator'''
    from .evaluate import TunableRagEvaluator

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    evaluator = TunableRagEvaluator(llm,
                                    config.judge_llm_prompt,
                                    builder.get_max_concurrency(),
                                    config.default_scoring,
                                    config.default_score_weights)

    yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description="Tunable RAG Evaluator")
