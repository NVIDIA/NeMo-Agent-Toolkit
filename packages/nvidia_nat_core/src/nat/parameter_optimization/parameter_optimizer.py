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

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping as Dict
from typing import TYPE_CHECKING
from typing import Any

import optuna
import yaml

from nat.data_models.config import Config
from nat.data_models.evaluate_runtime import EvaluationRunConfig
from nat.data_models.optimizable import SearchSpace
from nat.data_models.optimizer import OptimizerConfig
from nat.data_models.optimizer import OptimizerRunConfig
from nat.data_models.optimizer import SamplerType
from nat.experimental.decorators.experimental_warning_decorator import experimental
from nat.parameter_optimization.eval_runtime_loader import load_evaluation_run
from nat.parameter_optimization.parameter_selection import pick_trial
from nat.parameter_optimization.update_helpers import apply_suggestions

if TYPE_CHECKING:
    from nat.profiler.parameter_optimization.optimizer_callbacks import OptimizerCallbackManager

logger = logging.getLogger(__name__)
"""Optional eval runtime class."""


def _on_numeric_trial_end(
    callback_manager: OptimizerCallbackManager | None,
    trial: Any,
    eval_metrics: list[str],
    avg_scores: list[float],
    suggestions: dict[str, Any],
    last_eval_output: Any,
    all_scores: list[list[float]],
) -> None:
    """Build a TrialResult from one numeric-optimisation trial and fire on_trial_end."""
    if callback_manager is None:
        return
    from nat.plugins.eval.eval_callbacks import build_eval_result
    from nat.profiler.parameter_optimization.optimizer_callbacks import TrialResult

    eval_result = None
    try:
        eval_result = build_eval_result(
            eval_input_items=last_eval_output.eval_input.eval_input_items,
            evaluation_results=last_eval_output.evaluation_results,
            metric_scores=dict(zip(eval_metrics, avg_scores)),
            usage_stats=last_eval_output.usage_stats,
        )
    except Exception:
        logger.warning("Failed to build EvalResult for optimizer callback", exc_info=True)

    callback_manager.on_trial_end(
        TrialResult(
            trial_number=trial.number,
            parameters=dict(suggestions),
            metric_scores=dict(zip(eval_metrics, avg_scores)),
            is_best=False,
            rep_scores=all_scores,
            eval_result=eval_result,
        ))


def _on_numeric_study_end(
    callback_manager: OptimizerCallbackManager | None,
    best_trial_obj: Any,
    eval_metrics: list[str],
    n_trials: int,
) -> None:
    """Fire on_study_end for a completed numeric optimisation study."""
    if callback_manager is None:
        return
    from nat.profiler.parameter_optimization.optimizer_callbacks import TrialResult

    callback_manager.on_study_end(
        best_trial=TrialResult(
            trial_number=best_trial_obj.number,
            parameters=dict(best_trial_obj.params),
            metric_scores=dict(zip(eval_metrics, best_trial_obj.values)),
            is_best=True,
        ),
        total_trials=n_trials,
    )


@experimental(feature_name="Optimizer")
def optimize_parameters(
    *,
    base_cfg: Config,
    full_space: Dict[str, SearchSpace],
    optimizer_config: OptimizerConfig,
    opt_run_config: OptimizerRunConfig,
    callback_manager: OptimizerCallbackManager | None = None,
) -> tuple[Config, dict[str, Any], int]:
    """Tune all *non-prompt* hyper-parameters and persist the best config."""
    EvaluationRun = load_evaluation_run()
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

        # Route this trial's OTEL traces to a per-trial experiment project
        if callback_manager:
            trial_project = callback_manager.get_trial_project_name(trial.number)
            if trial_project:
                from nat.observability.utils.tracing_utils import get_tracing_configs
                tracing = get_tracing_configs(cfg_trial)
                for exporter_config in tracing.values():
                    if hasattr(exporter_config, 'project'):
                        exporter_config.project = trial_project

        async def _single_eval(trial_idx: int) -> tuple[list[float], Any]:  # noqa: ARG001
            eval_cfg = EvaluationRunConfig(
                config_file=cfg_trial,
                dataset=opt_run_config.dataset,
                result_json_path=opt_run_config.result_json_path,
                endpoint=opt_run_config.endpoint,
                endpoint_timeout=opt_run_config.endpoint_timeout,
            )
            eval_output = await _run_eval(EvaluationRun(config=eval_cfg))
            values = []
            for metric_name in eval_metrics:
                metric = next(r[1] for r in eval_output.evaluation_results if r[0] == metric_name)
                values.append(metric.average_score)

            return values, eval_output

        # Create tasks for all evaluations
        async def _run_all_evals():
            tasks = [_single_eval(i) for i in range(reps)]
            return await asyncio.gather(*tasks)

        # Calculate padding width based on total number of trials
        trial_id_width = len(str(max(0, optimizer_config.numeric.n_trials - 1)))
        trial_id_padded = f"{trial.number:0{trial_id_width}d}"
        with (out_dir / f"config_numeric_trial_{trial_id_padded}.yml").open("w") as fh:
            yaml.dump(cfg_trial.model_dump(), fh)

        all_results = asyncio.run(_run_all_evals())
        all_scores = [r[0] for r in all_results]
        last_eval_output = all_results[-1][1]  # Use last rep for per-item data
        # Persist raw per-repetition scores so they appear in `trials_dataframe`.
        trial.set_user_attr("rep_scores", all_scores)
        avg_scores = [sum(run[i] for run in all_scores) / reps for i in range(len(eval_metrics))]

        _on_numeric_trial_end(
            callback_manager,
            trial,
            eval_metrics,
            avg_scores,
            suggestions,
            last_eval_output,
            all_scores,
        )

        return avg_scores

    logger.info("Starting numeric / enum parameter optimization...")
    study.optimize(_objective, n_trials=optimizer_config.numeric.n_trials)
    logger.info("Numeric optimization finished")

    best_trial_obj = pick_trial(
        study=study,
        mode=optimizer_config.multi_objective_combination_mode,
        weights=weights,
    )
    best_params = best_trial_obj.params

    _on_numeric_study_end(callback_manager, best_trial_obj, eval_metrics, optimizer_config.numeric.n_trials)

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
        from nat.parameter_optimization.pareto_visualizer import create_pareto_visualization
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

    return tuned_cfg, dict(best_params), optimizer_config.numeric.n_trials
