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
"""Registry for optimizer strategies (numeric/parameter and GA prompt)."""

import asyncio
from collections.abc import AsyncIterator

from nat.cli.register_workflow import register_optimizer
from nat.data_models.config import Config
from nat.data_models.optimizable import SearchSpace
from nat.data_models.optimizer import GAPromptOptimizationConfig
from nat.data_models.optimizer import OptimizerConfig
from nat.data_models.optimizer import OptimizerRunConfig
from nat.data_models.optimizer import OptunaParameterOptimizationConfig
from nat.plugins.config_optimizer.parameters.base import BaseParameterOptimizer
from nat.plugins.config_optimizer.parameters.optimizer import optimize_parameters
from nat.plugins.config_optimizer.prompts.ga_prompt_optimizer import GAPromptOptimizer


class _ParameterOptimizerRunner(BaseParameterOptimizer):
    """Runner that delegates to optimize_parameters (sync) via asyncio.to_thread."""

    async def run(
        self,
        *,
        base_cfg: Config,
        full_space: dict[str, SearchSpace],
        optimizer_config: OptimizerConfig,
        opt_run_config: OptimizerRunConfig,
        callback_manager=None,
    ) -> tuple[Config, dict[str, object], int]:
        return await asyncio.to_thread(
            optimize_parameters,
            base_cfg=base_cfg,
            full_space=full_space,
            optimizer_config=optimizer_config,
            opt_run_config=opt_run_config,
            callback_manager=callback_manager,
        )


async def _parameter_optimizer_build(
    _config: OptunaParameterOptimizationConfig, ) -> AsyncIterator[_ParameterOptimizerRunner]:
    yield _ParameterOptimizerRunner()


@register_optimizer(config_type=OptunaParameterOptimizationConfig)
async def register_numeric_optimizer(config: OptunaParameterOptimizationConfig):
    async for runner in _parameter_optimizer_build(config):
        yield runner


async def _ga_prompt_optimizer_build(_config: GAPromptOptimizationConfig, ) -> AsyncIterator[GAPromptOptimizer]:
    yield GAPromptOptimizer()


@register_optimizer(config_type=GAPromptOptimizationConfig)
async def register_ga_prompt_optimizer(config: GAPromptOptimizationConfig):
    async for runner in _ga_prompt_optimizer_build(config):
        yield runner
