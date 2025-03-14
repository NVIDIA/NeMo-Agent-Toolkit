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
from aiq.cli.register_workflow import register_evaluator
from aiq.data_models.evaluator import EvaluatorBaseConfig


class TrajectoryEvaluatorConfig(EvaluatorBaseConfig, name="trajectory"):
    """Agent Trajectory Evaluation."""

    llm_name: str = Field(description="LLM as a judge.")


@register_evaluator(config_type=TrajectoryEvaluatorConfig)
async def register_trajectory_evaluator(config: TrajectoryEvaluatorConfig, builder: EvalBuilder):
    from aiq.builder.framework_enum import LLMFrameworkEnum

    from .evaluate import TrajectoryEvaluator
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools = builder.get_all_tools(wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    _evaluator = TrajectoryEvaluator(llm, tools, builder.get_max_concurrency())

    yield EvaluatorInfo(config=config, evaluate_fn=_evaluator.evaluate, description="Trajectory Evaluator")
