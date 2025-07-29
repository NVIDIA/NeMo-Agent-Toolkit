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
import random
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import get_args
from typing import get_origin

from pydantic import BaseModel

from aiq.data_models.config import AIQConfig
from aiq.data_models.optimizable import SearchSpace
from aiq.data_models.optimizer import OptimizerConfig
from aiq.data_models.optimizer import OptimizerRunConfig
from aiq.profiler.parameter_optimization.parameter_selection import pick_trial

logger = logging.getLogger(__name__)


class PromptOptimizerInputSchema(BaseModel):
    original_prompt: str
    objective: str
    oracle_feedback: str | None = None


def walk_optimizables(
    obj: BaseModel,  # **instance**, not a class
    path: str = "",
) -> Dict[str, SearchSpace]:
    """
    Return {flattened.path: SearchSpace} for every OptimizableField
    inside `obj`, including   • direct sub‑models
                              • dict[str, BaseModel] maps
                              • any nesting depth.
    """
    spaces: Dict[str, SearchSpace] = {}

    for name, fld in obj.model_fields.items():
        full = f"{path}.{name}" if path else name
        extra = fld.json_schema_extra or {}

        # 1 Plain field marked as optimizable
        if extra.get("optimizable"):
            spaces[full] = extra["search_space"]

        value = getattr(obj, name, None)

        # 2 Nested BaseModel instance
        if isinstance(value, BaseModel):
            spaces.update(walk_optimizables(value, full))

        # 3 Dict[str, BaseModel] container
        elif isinstance(value, dict):
            # runtime check: is the dict's *value* type a BaseModel?
            # works even with untyped dicts
            for key, subval in value.items():
                if isinstance(subval, BaseModel):
                    new_path = f"{full}.{key}"
                    spaces.update(walk_optimizables(subval, new_path))

        # 4 static-type fallback for *class* parsing
        elif isinstance(obj, type):
            ann = fld.annotation
            if get_origin(ann) in (dict, Dict):
                _, val_t = get_args(ann) or (None, None)
                if isinstance(val_t, type) and issubclass(val_t, BaseModel):
                    # We can't know the keys here, so we expose the
                    # container itself; caller can refine later.
                    spaces[f"{full}.*"] = SearchSpace(low=None, high=None)  # sentinel
    return spaces


# ------------------------------------------------------------------ #
# 2.  rebuild nested update‑dict (models + dict keys)                #
# ------------------------------------------------------------------ #
def nest_updates(flat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert {'a.b.c': 1, 'd.x.y': 2} ➜
            {'a': {'b': {'c': 1}}, 'd': {'x': {'y': 2}}}
    Works even when the middle segment is a dict key.
    """
    root: Dict[str, Any] = defaultdict(dict)

    for dotted, value in flat.items():
        head, *rest = dotted.split(".", 1)
        if not rest:  # reached leaf
            root[head] = value
            continue

        tail = rest[0]
        # Recurse and merge
        child_updates = nest_updates({tail: value})
        if isinstance(root[head], dict):
            root[head].update(child_updates)
        else:
            root[head] = child_updates
    return dict(root)


# ------------------------------------------------------------------ #
# 2.b helper to apply flat suggestions *without* clobbering siblings #
# ------------------------------------------------------------------ #
def apply_suggestions(cfg: BaseModel, flat: Dict[str, Any]) -> BaseModel:
    """
    Return a **new** config where only the dotted‑path keys in *flat*
    have been modified.

    Unlike `model_copy(update=...)`, this function merges updates
    *inside* nested BaseModels and dict[str, BaseModel] containers
    so that non‑mentioned siblings survive.
    """
    cfg_dict = cfg.model_dump(mode="python")  # deep, loss‑less copy
    for dotted, value in flat.items():
        keys = dotted.split(".")
        cursor = cfg_dict
        for key in keys[:-1]:
            # Ensure path exists and is a mutable mapping
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value
    # Re‑validate to get proper BaseModel instances back
    return cfg.__class__.model_validate(cfg_dict)


def _optimize_parameters(
    *,
    base_cfg: AIQConfig,
    full_space: dict[str, SearchSpace],
    optimizer_config: OptimizerConfig,
    opt_run_config: OptimizerRunConfig,
) -> AIQConfig:
    """Tune all *non‑prompt* hyper‑parameters and persist the best config."""
    import asyncio

    import optuna
    import yaml

    from aiq.eval.evaluate import EvaluationRun
    from aiq.eval.evaluate import EvaluationRunConfig

    # --------------------------------------------------------------------- #
    # 1.  prepare search space                                              #
    # --------------------------------------------------------------------- #
    space = {k: v for k, v in full_space.items() if not v.is_prompt}
    metric_names = optimizer_config.eval_metrics
    directions = [v.direction for v in metric_names.values()]
    eval_metrics = [v.evaluator_name for v in metric_names.values()]
    weights = [v.weight for v in metric_names.values()]

    study = optuna.create_study(directions=directions)

    async def _run_eval(runner):
        return await runner.run_and_evaluate()

    def _objective(trial):
        # -----------------------------------------------------------------
        # Determine how many repetitions to run for this parameter set
        # -----------------------------------------------------------------
        reps = max(1, getattr(optimizer_config, "reps_per_param_set", 1))

        # -----------------------------------------------------------------
        # 1. Build the trial‑specific configuration
        # -----------------------------------------------------------------
        suggestions = {p: spec.suggest(trial, p) for p, spec in space.items()}
        cfg_trial = apply_suggestions(base_cfg, suggestions)

        # -----------------------------------------------------------------
        # 2. Helper coroutine to execute a single evaluation run
        # -----------------------------------------------------------------
        async def _single_eval() -> list[float]:
            eval_cfg = EvaluationRunConfig(
                config_file=cfg_trial,
                dataset=opt_run_config.dataset,
                result_json_path=opt_run_config.result_json_path,
                endpoint=opt_run_config.endpoint,
                endpoint_timeout=opt_run_config.endpoint_timeout,
            )
            eval_runner = EvaluationRun(config=eval_cfg)
            outcome = await eval_runner.run_and_evaluate()
            eval_results = outcome.evaluation_results

            scores: list[float] = []
            for metric_name in eval_metrics:
                metric_result = next(
                    (r[1] for r in eval_results if r[0] == metric_name),
                    None,
                )
                if metric_result is None:
                    raise ValueError(f"Metric '{metric_name}' not found in eval results")
                scores.append(metric_result.average_score)
            return scores

        # -----------------------------------------------------------------
        # 3. Run the evaluation `reps` times concurrently
        # -----------------------------------------------------------------
        all_scores: list[list[float]] = asyncio.run(asyncio.gather(*[_single_eval() for _ in range(reps)]))

        # -----------------------------------------------------------------
        # 4. Average the metric scores element‑wise across repetitions
        # -----------------------------------------------------------------
        averaged_scores = [sum(run[i] for run in all_scores) / reps for i in range(len(eval_metrics))]
        return averaged_scores

    # --------------------------------------------------------------------- #
    # 2.  main optimization loop                                            #
    # --------------------------------------------------------------------- #
    n_iter = 1
    logger.info("Beginning parameter optimization")
    study.optimize(_objective, n_trials=optimizer_config.n_trials_numeric)
    logger.info("Completed iteration %d", n_iter)

    params = pick_trial(study=study, mode=optimizer_config.multi_objective_combination_mode, weights=weights).params
    tuned_cfg = apply_suggestions(base_cfg, params)

    # --------------------------------------------------------------------- #
    # 3.  persist outputs                                                   #
    # --------------------------------------------------------------------- #
    out_dir = optimizer_config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "optimized_config.yml").open("w+") as fh:
        yaml.dump(tuned_cfg.model_dump(), fh)

    with (out_dir / "trials_dataframe_params.csv").open("w+") as fh:
        study.trials_dataframe().to_csv(fh)

    logger.info("Parameter tuning complete", )
    return tuned_cfg


async def _optimize_prompts(
    *,
    base_cfg: AIQConfig,
    full_space: dict[str, SearchSpace],
    optimizer_config: OptimizerConfig,
    opt_run_config: OptimizerRunConfig,
) -> None:
    """optimize any prompt‑style search spaces and write them to disk."""
    import asyncio
    import json

    import optuna

    from aiq.builder.workflow_builder import WorkflowBuilder
    from aiq.eval.evaluate import EvaluationRun
    from aiq.eval.evaluate import EvaluationRunConfig

    metric_cfg = optimizer_config.eval_metrics
    directions = [v.direction for v in metric_cfg.values()]  # e.g. "maximize", ...
    eval_metrics = [v.evaluator_name for v in metric_cfg.values()]
    weights = [v.weight for v in metric_cfg.values()]

    prompt_space = {k: [(v.prompt, v.prompt_purpose)] for k, v in full_space.items() if v.is_prompt}
    if not prompt_space:
        logger.info("No prompts found to optimize – skipping.")
        return

    study = optuna.create_study(
        directions=directions,
        sampler=optuna.samplers.TPESampler(multivariate=True),
    )  # NSGA-II would be default; we override with MOTPE variant

    out_dir = optimizer_config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    async with WorkflowBuilder(general_config=base_cfg.general, registry=None) as builder:
        await builder.populate_builder(base_cfg)
        prompt_optim_fn = builder.get_function(optimizer_config.prompt_optimization_function)

    logger.info("Prompt optimization workflow loaded successfully")
    feedback_lists: dict[str, list[str]] = {}
    while len(study.trials) < optimizer_config.n_trials_prompt:
        trial = study.ask()
        suggestions: dict[str, str] = {}

        for param, choices in prompt_space.items():
            prompt_text, purpose = random.choice(choices)

            # bundle oracle feedback (if accumulated)
            oracle_fb = "\n".join(f"**Feedback:** {fb}" for fb in feedback_lists.get(param, [])) or None
            fn_input = PromptOptimizerInputSchema(
                original_prompt=prompt_text,
                objective=purpose,
                oracle_feedback=oracle_fb,
            )

            new_prompt = await prompt_optim_fn.acall_invoke(fn_input)

            logger.info("Prompt trial %d: %s", len(study.trials), new_prompt)
            choices.append((new_prompt, purpose))

            idx = trial._suggest(
                param,
                optuna.distributions.IntDistribution(0, len(choices) - 1),
            )

            suggestions[param] = choices[int(idx)][0]
            trial.set_user_attr("prompt_text", prompt_text)

        cfg_trial = apply_suggestions(base_cfg, suggestions)

        # -----------------------------------------------------------------
        # Evaluate the trial `reps_per_param_set` times concurrently
        # -----------------------------------------------------------------
        reps = max(1, getattr(optimizer_config, "reps_per_param_set", 1))

        async def _single_eval() -> list[tuple[str, Any]]:
            local_eval_cfg = EvaluationRunConfig(
                config_file=cfg_trial,
                dataset=opt_run_config.dataset,
                result_json_path=opt_run_config.result_json_path,
                endpoint=opt_run_config.endpoint,
                endpoint_timeout=opt_run_config.endpoint_timeout,
                override=opt_run_config.override,
            )
            runner = EvaluationRun(config=local_eval_cfg, skip_output=True)
            outcome = await runner.run_and_evaluate()
            return outcome.evaluation_results

        # Run the evaluations concurrently
        all_eval_results = await asyncio.gather(*(_single_eval() for _ in range(reps)))

        # Use the first run's results for feedback/oracle processing
        eval_results = all_eval_results[0]

        # -----------------------------------------------------------------
        # 1. Feedback processing based on first run
        # -----------------------------------------------------------------
        for metric_name, direction in zip(eval_metrics, directions):
            output = next((o for n, o in eval_results if n == metric_name), None)
            if output and getattr(output, "eval_output_items", None):
                output.eval_output_items.sort(
                    key=lambda item: item.score,
                    reverse=(direction == "minimize"),
                )

        num_fb = optimizer_config.num_feedback
        for metric_name, output in eval_results:
            if getattr(output, "eval_output_items", None):
                feedback_lists[metric_name] = [it.reasoning for it in output.eval_output_items[:num_fb]]

        # -----------------------------------------------------------------
        # 2. Compute metric values averaged across repetitions
        # -----------------------------------------------------------------
        metric_values: list[float] = []
        for metric_name in eval_metrics:
            scores = []
            for run_results in all_eval_results:
                metric = next(
                    (r[1] for r in run_results if r[0] == metric_name),
                    None,
                )
                if metric is None:
                    raise ValueError(f"Metric '{metric_name}' not found in eval results")
                scores.append(metric.average_score)
            metric_values.append(sum(scores) / reps)

        study.tell(trial, metric_values)
        logger.info("Prompt trial complete – current Pareto size %d", len(study.best_trials))

        # checkpoint best prompts so far
        best_params_flat = pick_trial(
            study=study,
            mode=optimizer_config.multi_objective_combination_mode,
            weights=weights,
        ).params
        best_prompts = {k: prompt_space[k][v] for k, v in best_params_flat.items()}
        ckpt_file = out_dir / f"optimized_prompts_round{len(study.trials)}.json"
        with ckpt_file.open("w+") as fh:
            json.dump(best_prompts, fh, indent=4)

        # ---- 6. final selection & persist -----------------------------------
    final_params = pick_trial(
        study=study,
        mode=optimizer_config.multi_objective_combination_mode,
        weights=weights,
    ).params
    best_prompts = {k: prompt_space[k][v] for k, v in final_params.items()}

    with (out_dir / "optimized_prompts.json").open("w+") as fh:
        json.dump(best_prompts, fh, indent=4)

    with (out_dir / "trials_dataframe_prompts.csv").open("w+") as fh:
        study.trials_dataframe().to_csv(fh)

    logger.info("Prompt optimization finished; results written to %s", out_dir / "optimized_prompts.json")


# --------------------------------------------------------------------
# 3.  optimization loop ----------------------------------------------
# --------------------------------------------------------------------
async def optimize_config(opt_run_config: OptimizerRunConfig):
    """
    Optimize the pipeline in two stages:
        1. numeric / enum parameter search
        2. (optional) prompt string optimization
    The heavy lifting is delegated to the private helpers above.
    """
    from aiq.data_models.config import AIQConfig
    from aiq.runtime.loader import load_config

    # ------------------------------------------------------------------ #
    # 1.  normalise / load the base config                               #
    # ------------------------------------------------------------------ #
    if not isinstance(opt_run_config.config_file, BaseModel):
        base_cfg: AIQConfig = load_config(config_file=opt_run_config.config_file)
    else:
        base_cfg = opt_run_config.config_file  # already a model instance

    # ------------------------------------------------------------------ #
    # 2.  discover optimizable fields once                               #
    # ------------------------------------------------------------------ #
    full_space = walk_optimizables(base_cfg)

    # ------------------------------------------------------------------ #
    # 3.  parameter optimization (if enabled)                            #
    # ------------------------------------------------------------------ #
    tuned_cfg = base_cfg
    if base_cfg.optimizer.do_numeric_optimization:
        tuned_cfg = _optimize_parameters(
            base_cfg=base_cfg,
            full_space=full_space,
            optimizer_config=base_cfg.optimizer,
            opt_run_config=opt_run_config,
        )

    # ------------------------------------------------------------------ #
    # 4.  prompt optimization (if enabled)                               #
    # ------------------------------------------------------------------ #
    if base_cfg.optimizer.do_prompt_optimization:
        await _optimize_prompts(
            base_cfg=tuned_cfg,
            full_space=full_space,
            optimizer_config=base_cfg.optimizer,
            opt_run_config=opt_run_config,
        )
