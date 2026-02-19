# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS UNDER ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry for optimizer strategies (numeric/parameter and GA prompt)."""

import asyncio
from collections.abc import AsyncIterator

from nat.cli.register_workflow import register_optimizer
from nat.data_models.config import Config
from nat.data_models.optimizable import SearchSpace
from nat.data_models.optimizer import NumericOptimizationConfig
from nat.data_models.optimizer import OptimizerConfig
from nat.data_models.optimizer import OptimizerRunConfig
from nat.data_models.optimizer import PromptGAOptimizationConfig

from .ga_config import GAOptimizerConfig
from .ga_prompt_optimizer import GAPromptOptimizer
from .parameter_optimizer import optimize_parameters


async def _parameter_optimizer_build(
    config: NumericOptimizationConfig,
) -> AsyncIterator["_ParameterOptimizerRunner"]:
    yield _ParameterOptimizerRunner()


class _ParameterOptimizerRunner:
    """Runner that delegates to optimize_parameters (sync) via asyncio.to_thread."""

    async def run(
        self,
        *,
        base_cfg: Config,
        full_space: dict[str, SearchSpace],
        optimizer_config: OptimizerConfig,
        opt_run_config: OptimizerRunConfig,
    ) -> Config:
        return await asyncio.to_thread(
            optimize_parameters,
            base_cfg=base_cfg,
            full_space=full_space,
            optimizer_config=optimizer_config,
            opt_run_config=opt_run_config,
        )


@register_optimizer(config_type=NumericOptimizationConfig)
async def register_numeric_optimizer(config: NumericOptimizationConfig):
    async for runner in _parameter_optimizer_build(config):
        yield runner


async def _ga_prompt_optimizer_build(
    config: PromptGAOptimizationConfig,
) -> AsyncIterator[GAPromptOptimizer]:
    yield GAPromptOptimizer()


class _GAPromptOptimizerRunnerAdapter:
    """Adapter so GAPromptOptimizer (expects GAOptimizerConfig) can be called with OptimizerConfig."""

    def __init__(self, ga_runner: GAPromptOptimizer):
        self._ga_runner = ga_runner

    async def run(
        self,
        *,
        base_cfg: Config,
        full_space: dict[str, SearchSpace],
        optimizer_config: OptimizerConfig,
        opt_run_config: OptimizerRunConfig,
    ) -> None:
        p = optimizer_config.prompt
        extra = getattr(p, "model_extra", None) or {}
        ga_config = GAOptimizerConfig(
            output_path=optimizer_config.output_path,
            eval_metrics=optimizer_config.eval_metrics,
            reps_per_param_set=optimizer_config.reps_per_param_set,
            target=optimizer_config.target,
            multi_objective_combination_mode=optimizer_config.multi_objective_combination_mode,
            prompt=p,
            oracle_feedback_mode=extra.get("oracle_feedback_mode", "never"),
            oracle_feedback_worst_n=extra.get("oracle_feedback_worst_n", 5),
            oracle_feedback_max_chars=extra.get("oracle_feedback_max_chars", 4000),
            oracle_feedback_fitness_threshold=extra.get("oracle_feedback_fitness_threshold", 0.3),
            oracle_feedback_stagnation_generations=extra.get(
                "oracle_feedback_stagnation_generations", 3
            ),
            oracle_feedback_fitness_variance_threshold=extra.get(
                "oracle_feedback_fitness_variance_threshold", 0.01
            ),
            oracle_feedback_diversity_threshold=extra.get(
                "oracle_feedback_diversity_threshold", 0.5
            ),
        )
        await self._ga_runner.run(
            base_cfg=base_cfg,
            full_space=full_space,
            optimizer_config=ga_config,
            opt_run_config=opt_run_config,
        )


@register_optimizer(config_type=PromptGAOptimizationConfig)
async def register_ga_prompt_optimizer(config: PromptGAOptimizationConfig):
    async for ga_runner in _ga_prompt_optimizer_build(config):
        yield _GAPromptOptimizerRunnerAdapter(ga_runner)
