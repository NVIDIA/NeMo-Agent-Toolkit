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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.data_models.optimizable import SearchSpace
from nat.data_models.optimizer import OptimizerConfig
from nat.data_models.optimizer import OptimizerRunConfig
from nat.eval.evaluate import EvaluationRun
from nat.eval.evaluate import EvaluationRunConfig
from nat.experimental.decorators.experimental_warning_decorator import experimental
from nat.profiler.parameter_optimization.update_helpers import apply_suggestions

logger = logging.getLogger(__name__)


class PromptOptimizerInputSchema(BaseModel):
    original_prompt: str
    objective: str
    oracle_feedback: str | None = None


@experimental(feature_name="Optimizer")
async def optimize_prompts(
    *,
    base_cfg: Config,
    full_space: dict[str, SearchSpace],
    optimizer_config: OptimizerConfig,
    opt_run_config: OptimizerRunConfig,
) -> None:

    # ------------- helpers ------------- #
    @dataclass
    class Individual:
        prompts: dict[str, str]  # param_name -> prompt text
        metrics: dict[str, float] | None = None  # evaluator_name -> average score
        scalar_fitness: float | None = None

    def _normalize_generation(
        individuals: Sequence[Individual],
        metric_names: Sequence[str],
        directions: Sequence[str],
        eps: float = 1e-12,
    ) -> list[dict[str, float]]:
        """Return per-individual dict of normalised scores in [0,1] where higher is better."""
        # Extract arrays per metric
        arrays = {m: [ind.metrics.get(m, 0.0) if ind.metrics else 0.0 for ind in individuals] for m in metric_names}
        normed: list[dict[str, float]] = []
        for i in range(len(individuals)):
            entry: dict[str, float] = {}
            for m, dirn in zip(metric_names, directions):
                vals = arrays[m]
                vmin = min(vals)
                vmax = max(vals)
                v = vals[i]
                # Map to [0,1] with higher=better regardless of direction
                if vmax - vmin < eps:
                    score01 = 0.5
                else:
                    score01 = (v - vmin) / (vmax - vmin)
                if dirn == "minimize":
                    score01 = 1.0 - score01
                entry[m] = float(score01)
            normed.append(entry)
        return normed

    def _scalarize(norm_scores: dict[str, float], *, mode: str, weights: Sequence[float] | None) -> float:
        """Collapse normalised scores to a single scalar (higher is better)."""
        vals = list(norm_scores.values())
        if not vals:
            return 0.0
        if mode == "harmonic":
            inv_sum = sum(1.0 / max(v, 1e-12) for v in vals)
            return len(vals) / max(inv_sum, 1e-12)
        if mode == "sum":
            if weights is None:
                return float(sum(vals))
            if len(weights) != len(vals):
                raise ValueError("weights length must equal number of objectives")
            return float(sum(w * v for w, v in zip(weights, vals)))
        if mode == "chebyshev":
            return float(min(vals))  # maximise the worst-case score
        raise ValueError(f"Unknown combination mode: {mode}")

    def _apply_diversity_penalty(individuals: Sequence[Individual], diversity_lambda: float) -> list[float]:
        if diversity_lambda <= 0.0:
            return [0.0 for _ in individuals]
        seen: dict[str, int] = {}
        keys: list[str] = []
        penalties: list[float] = []
        for ind in individuals:
            key = "\u241f".join(ind.prompts.get(k, "") for k in sorted(ind.prompts.keys()))
            keys.append(key)
            seen[key] = seen.get(key, 0) + 1
        for key in keys:
            duplicates = seen[key] - 1
            penalties.append(diversity_lambda * float(duplicates))
        return penalties

    def _tournament_select(pop: Sequence[Individual], k: int) -> Individual:
        contenders = random.sample(pop, k=min(k, len(pop)))
        return max(contenders, key=lambda i: (i.scalar_fitness or 0.0))

    # ------------- discover space ------------- #
    prompt_space: dict[str, tuple[str, str]] = {
        k: (v.prompt, v.prompt_purpose)
        for k, v in full_space.items() if v.is_prompt
    }

    if not prompt_space:
        logger.info("No prompts to optimize – skipping.")
        return

    metric_cfg = optimizer_config.eval_metrics
    if metric_cfg is None or len(metric_cfg) == 0:
        raise ValueError("optimizer_config.eval_metrics must be provided for GA prompt optimization")

    directions = [v.direction for v in metric_cfg.values()]  # "minimize" or "maximize"
    eval_metrics = [v.evaluator_name for v in metric_cfg.values()]
    weights = [v.weight for v in metric_cfg.values()]

    out_dir = optimizer_config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------- builder & functions ------------- #
    async with WorkflowBuilder(general_config=base_cfg.general, registry=None) as builder:
        await builder.populate_builder(base_cfg)
        init_fn_name = (optimizer_config.prompt.prompt_population_init_function)
        if not init_fn_name:
            raise ValueError(
                "No prompt optimization function configured. Set optimizer.prompt_population_init_function")
        init_fn = await builder.get_function(init_fn_name)

        recombine_fn = None
        if optimizer_config.prompt.prompt_recombination_function:
            recombine_fn = await builder.get_function(optimizer_config.prompt.prompt_recombination_function)

        logger.info(
            "GA Prompt optimization ready: init_fn=%s, recombine_fn=%s",
            init_fn_name,
            optimizer_config.prompt.prompt_recombination_function,
        )

        # ------------- GA parameters ------------- #
        pop_size = max(2, int(optimizer_config.prompt.ga_population_size))
        generations = max(1, int(optimizer_config.prompt.ga_generations))
        offspring_size = (optimizer_config.prompt.ga_offspring_size
                          or max(0, pop_size - optimizer_config.prompt.ga_elitism))
        crossover_rate = float(optimizer_config.prompt.ga_crossover_rate)
        mutation_rate = float(optimizer_config.prompt.ga_mutation_rate)
        elitism = max(0, int(optimizer_config.prompt.ga_elitism))
        selection_method = optimizer_config.prompt.ga_selection_method.lower()
        tournament_size = max(2, int(optimizer_config.prompt.ga_tournament_size))
        max_eval_concurrency = max(1, int(optimizer_config.prompt.ga_parallel_evaluations))
        diversity_lambda = float(optimizer_config.prompt.ga_diversity_lambda)

        # ------------- population init ------------- #
        async def _mutate_prompt(original_prompt: str, purpose: str) -> str:
            # Use LLM-based optimizer with no feedback
            return await init_fn.acall_invoke(
                PromptOptimizerInputSchema(
                    original_prompt=original_prompt,
                    objective=purpose,
                    oracle_feedback=None,
                ))

        async def _recombine_prompts(a: str, b: str, purpose: str) -> str:
            if recombine_fn is None:
                # Fallback: uniform choice per recombination
                return random.choice([a, b])
            payload = {"original_prompt": a, "objective": purpose, "oracle_feedback": None, "parent_b": b}
            return await recombine_fn.acall_invoke(payload)

        def _make_individual_from_prompts(prompts: dict[str, str]) -> Individual:
            return Individual(prompts=dict(prompts))

        async def _initial_population() -> list[Individual]:
            individuals: list[Individual] = []
            # Ensure first individual is the original prompts
            originals = {k: prompt_space[k][0] for k in prompt_space}
            individuals.append(_make_individual_from_prompts(originals))

            init_sem = asyncio.Semaphore(max_eval_concurrency)

            async def _create_random_individual() -> Individual:
                async with init_sem:
                    mutated: dict[str, str] = {}
                    for param, (base_prompt, purpose) in prompt_space.items():
                        try:
                            new_p = await _mutate_prompt(base_prompt, purpose)
                        except Exception as e:
                            logger.warning("Mutation failed for %s: %s; using original.", param, e)
                            new_p = base_prompt
                        mutated[param] = new_p
                    return _make_individual_from_prompts(mutated)

            needed = max(0, pop_size - 1)
            tasks = [_create_random_individual() for _ in range(needed)]
            individuals.extend(await asyncio.gather(*tasks))
            return individuals

        # ------------- evaluation ------------- #
        reps = max(1, getattr(optimizer_config, "reps_per_param_set", 1))

        sem = asyncio.Semaphore(max_eval_concurrency)

        async def _evaluate(ind: Individual) -> Individual:
            async with sem:
                cfg_trial = apply_suggestions(base_cfg, ind.prompts)
            eval_cfg = EvaluationRunConfig(
                config_file=cfg_trial,
                dataset=opt_run_config.dataset,
                result_json_path=opt_run_config.result_json_path,
                endpoint=opt_run_config.endpoint,
                endpoint_timeout=opt_run_config.endpoint_timeout,
                override=opt_run_config.override,
            )
            # Run reps sequentially under the same semaphore to avoid overload
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
            return ind

        async def _evaluate_population(pop: list[Individual]) -> list[Individual]:
            # Evaluate those missing metrics
            unevaluated = [ind for ind in pop if not ind.metrics]
            if unevaluated:
                evaluated = await asyncio.gather(*[_evaluate(ind) for ind in unevaluated])
                # in-place update
                for ind, ev in zip(unevaluated, evaluated):
                    ind.metrics = ev.metrics
            # Scalarize
            norm_per_ind = _normalize_generation(pop, eval_metrics, directions)
            penalties = _apply_diversity_penalty(pop, diversity_lambda)
            for ind, norm_scores, penalty in zip(pop, norm_per_ind, penalties):
                ind.scalar_fitness = _scalarize(
                    norm_scores, mode=optimizer_config.multi_objective_combination_mode, weights=weights) - penalty
            return pop

        # ------------- reproduction ops ------------- #
        async def _make_child(parent_a: Individual, parent_b: Individual) -> Individual:
            child_prompts: dict[str, str] = {}
            for param, (base_prompt, purpose) in prompt_space.items():
                pa = parent_a.prompts.get(param, base_prompt)
                pb = parent_b.prompts.get(param, base_prompt)
                child = pa
                # crossover
                if random.random() < crossover_rate:
                    try:
                        child = await _recombine_prompts(pa, pb, purpose)
                    except Exception as e:
                        logger.warning("Recombination failed for %s: %s; falling back to parent.", param, e)
                        child = random.choice([pa, pb])
                # mutation
                if random.random() < mutation_rate:
                    try:
                        child = await _mutate_prompt(child, purpose)
                    except Exception as e:
                        logger.warning("Mutation failed for %s: %s; keeping child as-is.", param, e)
                child_prompts[param] = child
            return _make_individual_from_prompts(child_prompts)

        # ------------- GA loop ------------- #
        population = await _initial_population()
        history_rows: list[dict[str, Any]] = []

        for gen in range(1, generations + 1):
            logger.info("[GA] Generation %d/%d: evaluating population of %d", gen, generations, len(population))
            population = await _evaluate_population(population)

            # Log and save checkpoint
            best = max(population, key=lambda i: (i.scalar_fitness or 0.0))
            checkpoint = {k: (best.prompts[k], prompt_space[k][1]) for k in prompt_space}
            checkpoint_path = out_dir / f"optimized_prompts_gen{gen}.json"
            with checkpoint_path.open("w") as fh:
                json.dump(checkpoint, fh, indent=2)
            logger.info("[GA] Saved checkpoint: %s (fitness=%.4f)", checkpoint_path, best.scalar_fitness or 0.0)

            # Append history
            for idx, ind in enumerate(population):
                row = {
                    "generation": gen,
                    "index": idx,
                    "scalar_fitness": ind.scalar_fitness,
                }
                if ind.metrics:
                    row.update({f"metric::{m}": ind.metrics[m] for m in eval_metrics})
                history_rows.append(row)

            # Next generation via elitism + reproduction
            next_population: list[Individual] = []
            if elitism > 0:
                elites = sorted(population, key=lambda i: (i.scalar_fitness or 0.0), reverse=True)[:elitism]
                next_population.extend([_make_individual_from_prompts(e.prompts) for e in elites])

            def _select_parent(curr_pop: list[Individual]) -> Individual:
                if selection_method == "tournament":
                    return _tournament_select(curr_pop, tournament_size)
                # roulette wheel
                total = sum(max(ind.scalar_fitness or 0.0, 0.0) for ind in curr_pop) or 1.0
                r = random.random() * total
                acc = 0.0
                for ind in curr_pop:
                    acc += max(ind.scalar_fitness or 0.0, 0.0)
                    if acc >= r:
                        return ind
                return curr_pop[-1]

            # Produce offspring
            needed = pop_size - len(next_population)
            offspring: list[Individual] = []
            for _ in range(max(0, offspring_size), needed):
                pass  # ensure bound correctness
            while len(offspring) < needed:
                p1 = _select_parent(population)
                p2 = _select_parent(population)
                if p2 is p1 and len(population) > 1:
                    p2 = random.choice([ind for ind in population if ind is not p1])
                child = await _make_child(p1, p2)
                offspring.append(child)

            population = next_population + offspring

        # Final evaluation to ensure metrics present
        population = await _evaluate_population(population)
        best = max(population, key=lambda i: (i.scalar_fitness or 0.0))
        best_prompts = {k: (best.prompts[k], prompt_space[k][1]) for k in prompt_space}

        # Save final
        final_prompts_path = out_dir / "optimized_prompts.json"
        with final_prompts_path.open("w") as fh:
            json.dump(best_prompts, fh, indent=2)

        trials_df_path = out_dir / "ga_history_prompts.csv"
        try:
            # Lazy import pandas if available; otherwise write CSV manually
            import csv  # pylint: disable=import-outside-toplevel

            fieldnames: list[str] = sorted({k for row in history_rows for k in row.keys()})
            with trials_df_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in history_rows:
                    writer.writerow(row)
        except Exception as e:  # pragma: no cover - best effort
            logger.warning("Failed to write GA history CSV: %s", e)

        logger.info("Prompt GA optimization finished successfully!")
        logger.info("Final prompts saved to: %s", final_prompts_path)
        logger.info("History saved to: %s", trials_df_path)
