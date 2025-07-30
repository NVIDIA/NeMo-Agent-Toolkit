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

import asyncio
import logging
from typing import Dict

import optuna
import yaml

from aiq.data_models.config import AIQConfig
from aiq.data_models.optimizable import SearchSpace
from aiq.data_models.optimizer import OptimizerConfig
from aiq.data_models.optimizer import OptimizerRunConfig
from aiq.eval.evaluate import EvaluationRun
from aiq.eval.evaluate import EvaluationRunConfig
from aiq.experimental.optimizer.parameter_selection import pick_trial
from aiq.experimental.optimizer.pareto_visualizer import create_pareto_visualization
from aiq.experimental.optimizer.update_helpers import apply_suggestions

logger = logging.getLogger(__name__)


def optimize_parameters(
    *,
    base_cfg: AIQConfig,
    full_space: Dict[str, SearchSpace],
    optimizer_config: OptimizerConfig,
    opt_run_config: OptimizerRunConfig,
) -> AIQConfig:
    """Tune all *non-prompt* hyper-parameters and persist the best config."""
    space = {k: v for k, v in full_space.items() if not v.is_prompt}

    # Ensure output_path is not None
    if optimizer_config.output_path is None:
        raise ValueError("optimizer_config.output_path cannot be None")
    out_dir = optimizer_config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure eval_metrics is not None
    if optimizer_config.eval_metrics is None:
        raise ValueError("optimizer_config.eval_metrics cannot be None")

    metric_cfg = optimizer_config.eval_metrics
    directions = [v.direction for v in metric_cfg.values()]
    eval_metrics = [v.evaluator_name for v in metric_cfg.values()]
    weights = [v.weight for v in metric_cfg.values()]

    study = optuna.create_study(directions=directions)

    # Create output directory for intermediate files
    out_dir = optimizer_config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    async def _run_eval(runner: EvaluationRun):
        return await runner.run_and_evaluate()

    def _objective(trial: optuna.Trial):
        reps = max(1, getattr(optimizer_config, "reps_per_param_set", 1))

        # build trial config
        suggestions = {p: spec.suggest(trial, p) for p, spec in space.items()}
        cfg_trial = apply_suggestions(base_cfg, suggestions)

        async def _single_eval(trial_idx: int) -> list[float]:
            eval_cfg = EvaluationRunConfig(
                config_file=cfg_trial,
                dataset=str(opt_run_config.dataset) if opt_run_config.dataset else None,
                result_json_path=opt_run_config.result_json_path,
                endpoint=opt_run_config.endpoint,
                endpoint_timeout=opt_run_config.endpoint_timeout,
            )
            scores = await _run_eval(EvaluationRun(config=eval_cfg))
            values = []
            for metric_name in eval_metrics:
                metric = next(r[1] for r in scores.evaluation_results if r[0] == metric_name)
                values.append(metric.average_score)

            return values

        # Create tasks for all evaluations
        async def _run_all_evals():
            tasks = [_single_eval(i) for i in range(reps)]
            return await asyncio.gather(*tasks)

        all_scores = asyncio.run(_run_all_evals())

        with (out_dir / f"config_numeric_trial_{trial._trial_id}.yml").open("w") as fh:
            yaml.dump(cfg_trial.model_dump(), fh)

        return [sum(run[i] for run in all_scores) / reps for i in range(len(eval_metrics))]

    logger.info("Starting numeric / enum parameter optimization...")
    study.optimize(_objective, n_trials=optimizer_config.n_trials_numeric)
    logger.info("Numeric optimization finished")

    best_params = pick_trial(
        study=study,
        mode=optimizer_config.multi_objective_combination_mode,
        weights=weights,
    ).params
    tuned_cfg = apply_suggestions(base_cfg, best_params)

    # Save final results (out_dir already created and defined above)
    with (out_dir / "optimized_config.yml").open("w") as fh:
        yaml.dump(tuned_cfg.model_dump(), fh)
    with (out_dir / "trials_dataframe_params.csv").open("w") as fh:
        study.trials_dataframe().to_csv(fh, index=False)

    # Generate Pareto front visualizations
    try:
        logger.info("Generating Pareto front visualizations...")
        create_pareto_visualization(
            data_source=study,
            metric_names=eval_metrics,
            directions=directions,
            output_dir=out_dir / "plots",
            title_prefix="Parameter Optimization",
            show_plots=False  # Don't show plots in automated runs
        )
        logger.info("Pareto visualizations saved to: %s", out_dir / "plots")
    except Exception as e:
        logger.warning("Failed to generate visualizations: %s", e)

    return tuned_cfg
