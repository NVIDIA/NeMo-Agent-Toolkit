# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract base class for evolutionary prompt optimizers (GA and related strategies)."""

import asyncio
from abc import ABC
from abc import abstractmethod
from typing import Any

from nat.data_models.config import Config
from nat.data_models.optimizable import SearchSpace
from nat.data_models.optimizer import OptimizerConfig
from nat.data_models.optimizer import OptimizerRunConfig
from nat.eval.evaluate import EvaluationRun
from nat.eval.evaluate import EvaluationRunConfig
from nat.optimizer.ga_individual import Individual
from nat.optimizer.update_helpers import apply_suggestions


class BaseEvolutionaryPromptOptimizer(ABC):
    """
    Base class for evolutionary prompt optimizers.

    Provides evaluation infrastructure: apply_suggestions + EvaluationRun,
    concurrent population evaluation, and a _post_evaluate_single hook for
    subclasses. Fitness computation and persistence are implementation-specific.
    """

    @abstractmethod
    async def run(
        self,
        *,
        base_cfg: Config,
        full_space: dict[str, SearchSpace],
        optimizer_config: OptimizerConfig,
        opt_run_config: OptimizerRunConfig,
    ) -> None:
        """Run the evolutionary optimization loop."""
        ...

    # ---------- evaluation ---------- #

    async def _evaluate_single_given_trial(
        self,
        ind: Individual,
        cfg_trial: Config,
        optimizer_config: OptimizerConfig,
        opt_run_config: OptimizerRunConfig,
    ) -> None:
        """Run EvaluationRun for an already-built trial config; fill ind.metrics. Subclasses may extend via _post_evaluate_single."""
        eval_cfg = EvaluationRunConfig(
            config_file=cfg_trial,
            dataset=opt_run_config.dataset,
            result_json_path=opt_run_config.result_json_path,
            endpoint=opt_run_config.endpoint,
            endpoint_timeout=opt_run_config.endpoint_timeout,
            override=opt_run_config.override,
        )
        metric_cfg = optimizer_config.eval_metrics or {}
        eval_metrics = [v.evaluator_name for v in metric_cfg.values()]
        reps = max(1, getattr(optimizer_config, "reps_per_param_set", 1))

        all_results: list[list[tuple[str, Any]]] = []
        for _ in range(reps):
            res = (await EvaluationRun(config=eval_cfg).run_and_evaluate()).evaluation_results
            all_results.append(res)

        metrics: dict[str, float] = {}
        for metric_name in eval_metrics:
            scores: list[float] = []
            for run_results in all_results:
                for name, result in run_results:
                    if name == metric_name:
                        scores.append(result.average_score)
                        break
            metrics[metric_name] = float(sum(scores) / len(scores)) if scores else 0.0
        ind.metrics = metrics
        await self._post_evaluate_single(ind, all_results, optimizer_config, opt_run_config)

    async def _post_evaluate_single(
        self,
        ind: Individual,
        all_results: list[list[tuple[str, Any]]],
        optimizer_config: OptimizerConfig,
        opt_run_config: OptimizerRunConfig,
    ) -> None:
        """Override in subclasses to add post-evaluation logic (e.g. oracle feedback). Default no-op."""
        pass

    async def _evaluate_population(
        self,
        population: list[Individual],
        base_cfg: Config,
        optimizer_config: OptimizerConfig,
        opt_run_config: OptimizerRunConfig,
        max_concurrency: int = 8,
    ) -> None:
        """Evaluate all individuals (concurrently). Semaphore wraps only apply_suggestions."""
        unevaluated = [ind for ind in population if not ind.metrics]
        if unevaluated:
            sem = asyncio.Semaphore(max_concurrency)

            async def _eval_one(ind: Individual) -> None:
                async with sem:
                    cfg_trial = apply_suggestions(base_cfg, ind.prompts)
                await self._evaluate_single_given_trial(
                    ind, cfg_trial, optimizer_config, opt_run_config
                )

            await asyncio.gather(*[_eval_one(ind) for ind in unevaluated])
