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
import json
import logging
import random
from typing import Dict
from typing import List

import optuna
from pydantic import BaseModel

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.config import AIQConfig
from aiq.data_models.optimizable import SearchSpace
from aiq.data_models.optimizer import OptimizerConfig
from aiq.data_models.optimizer import OptimizerRunConfig
from aiq.eval.evaluate import EvaluationRun
from aiq.eval.evaluate import EvaluationRunConfig
from aiq.profiler.parameter_optimization.parameter_selection import pick_trial

from aiq.profiler.parameter_optimization.update_helpers import apply_suggestions

logger = logging.getLogger(__name__)


class PromptOptimizerInputSchema(BaseModel):
    original_prompt: str
    objective: str
    oracle_feedback: str | None = None


async def optimize_prompts(
    *,
    base_cfg: AIQConfig,
    full_space: Dict[str, SearchSpace],
    optimizer_config: OptimizerConfig,
    opt_run_config: OptimizerRunConfig,
) -> None:
    """Optimizes prompt-style search spaces and writes results to disk."""
    prompt_space = {k: [(v.prompt, v.prompt_purpose)] for k, v in full_space.items() if v.is_prompt}
    if not prompt_space:
        logger.info("No prompts to optimize – skipping.")
        return

    metric_cfg = optimizer_config.eval_metrics
    directions = [v.direction for v in metric_cfg.values()]
    eval_metrics = [v.evaluator_name for v in metric_cfg.values()]
    weights = [v.weight for v in metric_cfg.values()]

    study = optuna.create_study(
        directions=directions,
        sampler=optuna.samplers.TPESampler(multivariate=True),
    )
    out_dir = optimizer_config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    async with WorkflowBuilder(general_config=base_cfg.general, registry=None) as builder:
        await builder.populate_builder(base_cfg)
        prompt_optim_fn = builder.get_function(optimizer_config.prompt_optimization_function)

    logger.info("Prompt optimization workflow ready")

    feedback_lists: Dict[str, List[str]] = {}
    while len(study.trials) < optimizer_config.n_trials_prompt:
        trial = study.ask()
        suggestions: Dict[str, str] = {}

        for param, choices in prompt_space.items():
            prompt_text, purpose = random.choice(choices)
            oracle_fb = "\n".join(f"**Feedback:** {fb}" for fb in feedback_lists.get(param, [])) or None
            new_prompt = await prompt_optim_fn.acall_invoke(
                PromptOptimizerInputSchema(
                    original_prompt=prompt_text,
                    objective=purpose,
                    oracle_feedback=oracle_fb,
                ))
            choices.append((new_prompt, purpose))
            idx = trial._suggest(
                param,
                optuna.distributions.IntDistribution(0, len(choices) - 1),
            )
            suggestions[param] = choices[int(idx)][0]
            trial.set_user_attr("prompt_text", prompt_text)

        cfg_trial = apply_suggestions(base_cfg, suggestions)
        reps = max(1, getattr(optimizer_config, "reps_per_param_set", 1))

        async def _single_eval() -> list[tuple[str, any]]:
            eval_cfg = EvaluationRunConfig(
                config_file=cfg_trial,
                dataset=opt_run_config.dataset,
                result_json_path=opt_run_config.result_json_path,
                endpoint=opt_run_config.endpoint,
                endpoint_timeout=opt_run_config.endpoint_timeout,
                override=opt_run_config.override,
            )
            return (await EvaluationRun(config=eval_cfg, skip_output=True).run_and_evaluate()).evaluation_results

        all_results = await asyncio.gather(*(_single_eval() for _ in range(reps)))

        # ---------- feedback handling (first run only) ----------
        eval_results = all_results[0]
        for metric_name, direction in zip(eval_metrics, directions):
            output = next((o for n, o in eval_results if n == metric_name), None)
            if output and getattr(output, "eval_output_items", None):
                output.eval_output_items.sort(
                    key=lambda it: it.score,
                    reverse=(direction == "minimize"),
                )
                feedback_lists[metric_name] = [
                    it.reasoning for it in output.eval_output_items[:optimizer_config.num_feedback]
                ]

        # ---------- aggregate metric values ----------
        trial_values = []
        for metric_name in eval_metrics:
            scores = [next(r[1] for r in run if r[0] == metric_name).average_score for run in all_results]
            trial_values.append(sum(scores) / reps)

        study.tell(trial, trial_values)
        logger.info("Prompt trial complete – Pareto set size %d", len(study.best_trials))

        # checkpoint best prompts so far
        ckpt = {
            k: prompt_space[k][v]
            for k, v in pick_trial(
                study=study,
                mode=optimizer_config.multi_objective_combination_mode,
                weights=weights, ).params.items()
        }
        with (out_dir / f"optimized_prompts_round{len(study.trials)}.json").open("w") as fh:
            json.dump(ckpt, fh, indent=4)

    final_params = pick_trial(
        study=study,
        mode=optimizer_config.multi_objective_combination_mode,
        weights=weights,
    ).params
    best_prompts = {k: prompt_space[k][v] for k, v in final_params.items()}

    with (out_dir / "optimized_prompts.json").open("w") as fh:
        json.dump(best_prompts, fh, indent=4)
    with (out_dir / "trials_dataframe_prompts.csv").open("w") as fh:
        study.trials_dataframe().to_csv(fh)

    logger.info("Prompt optimization finished; results written to %s", out_dir / "optimized_prompts.json")
