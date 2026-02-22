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

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorLLMConfig
from nat.utils.exception_handlers.automatic_retries import patch_with_retry


class TrajectoryEvaluatorConfig(EvaluatorLLMConfig, name="trajectory"):
    """Agent Trajectory Evaluation."""

    pass


@register_evaluator(config_type=TrajectoryEvaluatorConfig)
async def register_trajectory_evaluator(config: TrajectoryEvaluatorConfig, builder: EvalBuilder):
    from nat.builder.framework_enum import LLMFrameworkEnum

    from .evaluate import TrajectoryEvaluator

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    if config.do_auto_retry:
        llm = patch_with_retry(
            llm,
            retries=config.num_retries,
            retry_codes=config.retry_on_status_codes,
            retry_on_messages=config.retry_on_errors,
        )

    tools = await builder.get_all_tools(wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    _evaluator = TrajectoryEvaluator(llm, tools, builder.get_max_concurrency())

    yield EvaluatorInfo(config=config, evaluate_fn=_evaluator.evaluate, description="Trajectory Evaluator")
