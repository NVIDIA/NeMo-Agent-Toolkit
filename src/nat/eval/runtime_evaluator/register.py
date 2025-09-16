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

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalOutput


class AverageLLMLatencyConfig(EvaluatorBaseConfig, name="avg_llm_latency"):
    """Mean difference between connected LLM_START and LLM_END events (same UUID)."""

    max_concurrency: int = Field(default=8, description="Max concurrency for evaluation.")


class AverageWorkflowRuntimeConfig(EvaluatorBaseConfig, name="avg_workflow_runtime"):
    """Average workflow runtime per item (max timestamp - min timestamp)."""

    max_concurrency: int = Field(default=8, description="Max concurrency for evaluation.")


class AverageNumberOfLLMCallsConfig(EvaluatorBaseConfig, name="avg_num_llm_calls"):
    """Average number of LLM calls per item (count of LLM_END)."""

    max_concurrency: int = Field(default=8, description="Max concurrency for evaluation.")


class AverageTokensPerLLMEndConfig(EvaluatorBaseConfig, name="avg_tokens_per_llm_end"):
    """Average total tokens per LLM_END event (prompt + completion if available)."""

    max_concurrency: int = Field(default=8, description="Max concurrency for evaluation.")


@register_evaluator(config_type=AverageLLMLatencyConfig)
async def register_avg_llm_latency_evaluator(config: AverageLLMLatencyConfig, builder: EvalBuilder):
    from .evaluate import AverageLLMLatencyEvaluator

    evaluator = AverageLLMLatencyEvaluator(max_concurrency=config.max_concurrency or builder.get_max_concurrency())

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        return await evaluator.evaluate(eval_input)

    yield EvaluatorInfo(config=config,
                        evaluate_fn=evaluate_fn,
                        description="Average LLM latency (s) from LLM_START to LLM_END")


@register_evaluator(config_type=AverageWorkflowRuntimeConfig)
async def register_avg_workflow_runtime_evaluator(config: AverageWorkflowRuntimeConfig, builder: EvalBuilder):
    from .evaluate import AverageWorkflowRuntimeEvaluator

    evaluator = AverageWorkflowRuntimeEvaluator(max_concurrency=config.max_concurrency or builder.get_max_concurrency())

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        return await evaluator.evaluate(eval_input)

    yield EvaluatorInfo(config=config, evaluate_fn=evaluate_fn, description="Average workflow runtime (s)")


@register_evaluator(config_type=AverageNumberOfLLMCallsConfig)
async def register_avg_num_llm_calls_evaluator(config: AverageNumberOfLLMCallsConfig, builder: EvalBuilder):
    from .evaluate import AverageNumberOfLLMCallsEvaluator

    evaluator = AverageNumberOfLLMCallsEvaluator(
        max_concurrency=config.max_concurrency or builder.get_max_concurrency())

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        return await evaluator.evaluate(eval_input)

    yield EvaluatorInfo(config=config, evaluate_fn=evaluate_fn, description="Average number of LLM calls")


@register_evaluator(config_type=AverageTokensPerLLMEndConfig)
async def register_avg_tokens_per_llm_end_evaluator(config: AverageTokensPerLLMEndConfig, builder: EvalBuilder):
    from .evaluate import AverageTokensPerLLMEndEvaluator

    evaluator = AverageTokensPerLLMEndEvaluator(max_concurrency=config.max_concurrency or builder.get_max_concurrency())

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        return await evaluator.evaluate(eval_input)

    yield EvaluatorInfo(config=config,
                        evaluate_fn=evaluate_fn,
                        description="Average total tokens per LLM_END (prompt + completion)")
