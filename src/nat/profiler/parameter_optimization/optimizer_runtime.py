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

import logging

from pydantic import BaseModel

from nat.data_models.optimizer import OptimizerRunConfig
from nat.experimental.decorators.experimental_warning_decorator import experimental
from nat.profiler.parameter_optimization.optimizable_utils import walk_optimizables
from nat.profiler.parameter_optimization.parameter_optimizer import optimize_parameters
from nat.profiler.parameter_optimization.prompt_optimizer import optimize_prompts
from nat.runtime.loader import load_config

logger = logging.getLogger(__name__)


@experimental(feature_name="Optimizer")
async def optimize_config(opt_run_config: OptimizerRunConfig):
    """Entry-point called by the CLI or runtime."""
    # ---------------- 1. load / normalise ---------------- #
    if not isinstance(opt_run_config.config_file, BaseModel):
        from nat.data_models.config import Config  # guarded import
        base_cfg: Config = load_config(config_file=opt_run_config.config_file)
    else:
        base_cfg = opt_run_config.config_file  # already validated

    # ---------------- 2. discover search space ----------- #
    full_space = walk_optimizables(base_cfg)
    if not full_space:
        logger.warning("No optimizable parameters found in the configuration. "
                       "Skipping optimization.")
        return base_cfg

    # ---------------- 3. numeric / enum tuning ----------- #
    tuned_cfg = base_cfg
    if base_cfg.optimizer.numeric.enabled:
        tuned_cfg = optimize_parameters(
            base_cfg=base_cfg,
            full_space=full_space,
            optimizer_config=base_cfg.optimizer,
            opt_run_config=opt_run_config,
        )

    # ---------------- 4. prompt optimization ------------- #
    if base_cfg.optimizer.prompt.enabled:
        await optimize_prompts(
            base_cfg=tuned_cfg,
            full_space=full_space,
            optimizer_config=base_cfg.optimizer,
            opt_run_config=opt_run_config,
        )

    logger.info("All optimization phases complete.")
    return tuned_cfg
