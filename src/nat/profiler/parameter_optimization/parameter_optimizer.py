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
from collections.abc import Mapping as Dict

import optuna
import yaml

from nat.data_models.config import Config
from nat.data_models.optimizable import SearchSpace
from nat.data_models.optimizer import OptimizerConfig
from nat.data_models.optimizer import OptimizerRunConfig
from nat.data_models.optimizer import SamplerType
from nat.eval.evaluate import EvaluationRun
from nat.eval.evaluate import EvaluationRunConfig
from nat.experimental.decorators.experimental_warning_decorator import experimental
from nat.profiler.parameter_optimization.parameter_selection import pick_trial
from nat.profiler.parameter_optimization.update_helpers import apply_suggestions

logger = logging.getLogger(__name__)


@experimental(feature_name="Optimizer")
def optimize_parameters(
    *,
    base_cfg: Config,
    full_space: Dict[str, SearchSpace],
    optimizer_config: OptimizerConfig,
    opt_run_config: OptimizerRunConfig,
) -> Config:
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

    # Create appropriate sampler based on configuration
    sampler_type = optimizer_config.numeric.sampler

    if sampler_type == SamplerType.GRID:
        # For grid search, convert the existing space to value sequences
        grid_search_space = {param_name: search_space.to_grid_values() for param_name, search_space in space.items()}
        sampler = optuna.samplers.GridSampler(grid_search_space)
        logger.info("Using Grid sampler for numeric optimization")
    else:
        # None or BAYESIAN: let Optuna choose defaults
        sampler = None
        logger.info(
            "Using Optuna default sampler types: TPESampler for single-objective, NSGAIISampler for multi-objective")

    study = optuna.create_study(directions=directions, sampler=sampler)

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

        async def _single_eval(trial_idx: int) -> list[float]:  # noqa: ARG001
            eval_cfg = EvaluationRunConfig(
                config_file=cfg_trial,
                dataset=opt_run_config.dataset,
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

        # Calculate padding width based on total number of trials
        trial_id_width = len(str(max(0, optimizer_config.numeric.n_trials - 1)))
        trial_id_padded = f"{trial.number:0{trial_id_width}d}"
        with (out_dir / f"config_numeric_trial_{trial_id_padded}.yml").open("w") as fh:
            yaml.dump(cfg_trial.model_dump(), fh)

        all_scores = asyncio.run(_run_all_evals())
        # Persist raw per‑repetition scores so they appear in `trials_dataframe`.
        trial.set_user_attr("rep_scores", all_scores)
        return [sum(run[i] for run in all_scores) / reps for i in range(len(eval_metrics))]

    logger.info("Starting numeric / enum parameter optimization...")
    study.optimize(_objective, n_trials=optimizer_config.numeric.n_trials)
    logger.info("Numeric optimization finished")

    best_params = pick_trial(
        study=study,
        mode=optimizer_config.multi_objective_combination_mode,
        weights=weights,
    ).params
    tuned_cfg = apply_suggestions(base_cfg, best_params)

    # Save final results (out_dir already created and defined above)
    with (out_dir / "optimized_config.yml").open("w") as fh:
        yaml.dump(tuned_cfg.model_dump(mode='json'), fh)
    with (out_dir / "trials_dataframe_params.csv").open("w") as fh:
        # Export full trials DataFrame (values, params, timings, etc.).
        df = study.trials_dataframe()

        # Rename values_X columns to actual metric names
        metric_names = list(metric_cfg.keys())
        rename_mapping = {}
        for i, metric_name in enumerate(metric_names):
            old_col = f"values_{i}"
            if old_col in df.columns:
                rename_mapping[old_col] = f"values_{metric_name}"
        if rename_mapping:
            df = df.rename(columns=rename_mapping)

        # Normalise rep_scores column naming for convenience.
        if "user_attrs_rep_scores" in df.columns and "rep_scores" not in df.columns:
            df = df.rename(columns={"user_attrs_rep_scores": "rep_scores"})
        elif "user_attrs" in df.columns and "rep_scores" not in df.columns:
            # Some Optuna versions return a dict in a single user_attrs column.
            df["rep_scores"] = df["user_attrs"].apply(lambda d: d.get("rep_scores") if isinstance(d, dict) else None)
            df = df.drop(columns=["user_attrs"])

        # Get Pareto optimal trial numbers from Optuna study
        pareto_trials = study.best_trials
        pareto_trial_numbers = {trial.number for trial in pareto_trials}
        # Add boolean column indicating if trial is Pareto optimal
        df["pareto_optimal"] = df["number"].isin(pareto_trial_numbers)

        df.to_csv(fh, index=False)

    # Generate Pareto front visualizations
    try:
        from nat.profiler.parameter_optimization.pareto_visualizer import create_pareto_visualization
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
    except ImportError as ie:
        logger.warning("Could not import visualization dependencies: %s. "
                       "Have you installed nvidia-nat-profiling?",
                       ie)
    except Exception as e:
        logger.warning("Failed to generate visualizations: %s", e)

    return tuned_cfg
