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
"""
Prompt Optimization Module

This module provides functionality for optimizing prompts using Optuna-based hyperparameter
optimization. It uses a feedback-driven approach where prompts are iteratively improved
based on evaluation results and trajectory feedback.

The optimization process:
1. Identifies prompt-type parameters from the search space
2. Creates an Optuna study for multi-objective optimization
3. For each trial:
   - Generates new prompt variants using an optimization function
   - Evaluates the prompts using specified metrics
   - Collects feedback from trajectory evaluations
   - Uses feedback to improve subsequent prompt generations
4. Saves the best performing prompts based on Pareto optimality

Key Features:
- Multi-objective optimization support
- Feedback-driven prompt improvement
- Trajectory-based reasoning collection
- Checkpoint saving for intermediate results
"""

import asyncio
import json
import logging
import random
from typing import Any

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
from aiq.profiler.parameter_optimization.pareto_visualizer import create_pareto_visualization
from aiq.profiler.parameter_optimization.update_helpers import apply_suggestions

logger = logging.getLogger(__name__)


class PromptOptimizerInputSchema(BaseModel):
    """
    Input schema for the prompt optimization function.
    
    Attributes:
        original_prompt: The base prompt text to be optimized
        objective: The purpose or goal of the prompt (used for optimization guidance)
        oracle_feedback: Optional feedback from previous evaluations to guide improvements
    """
    original_prompt: str
    objective: str
    oracle_feedback: str | None = None


async def optimize_prompts(
    *,
    base_cfg: AIQConfig,
    full_space: dict[str, SearchSpace],
    optimizer_config: OptimizerConfig,
    opt_run_config: OptimizerRunConfig,
) -> None:
    """
    Optimizes prompt-style search spaces using multi-objective optimization.
    
    This function performs iterative prompt optimization by:
    1. Filtering the search space to identify prompt parameters
    2. Setting up an Optuna study for multi-objective optimization
    3. Running optimization trials with feedback collection
    4. Saving intermediate and final results
    
    Args:
        base_cfg: Base AIQ configuration containing workflow settings
        full_space: Complete search space with all optimizable parameters
        optimizer_config: Configuration for the optimization process
        opt_run_config: Runtime configuration for evaluation runs
        
    Returns:
        None: Results are written to disk at the specified output path
        
    Raises:
        Exception: Various exceptions may be raised during evaluation or optimization
    """
    # Filter search space to only include prompt-type parameters
    # Each prompt parameter contains (prompt_text, purpose) tuples
    prompt_space: dict[str, list[tuple[str, str]]] = {
        k: [(v.prompt, v.prompt_purpose)]
        for k, v in full_space.items() if v.is_prompt
    }

    if not prompt_space:
        logger.info("No prompts to optimize – skipping.")
        return

    # Extract optimization configuration
    metric_cfg = optimizer_config.eval_metrics
    directions = [v.direction for v in metric_cfg.values()]  # "minimize" or "maximize"
    eval_metrics = [v.evaluator_name for v in metric_cfg.values()]
    trajectory_eval_metric_name = optimizer_config.trajectory_eval_metric_name
    weights = [v.weight for v in metric_cfg.values()]  # For weighted optimization

    logger.info("Starting prompt optimization with %d prompt parameters", len(prompt_space))
    logger.info("Optimization metrics: %s", eval_metrics)
    logger.info("Optimization directions: %s", directions)

    # Create Optuna study for multi-objective optimization
    study = optuna.create_study(
        directions=directions,
        sampler=optuna.samplers.TPESampler(multivariate=True),
    )

    # Ensure output directory exists
    out_dir = optimizer_config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the workflow builder and get the prompt optimization function
    try:
        async with WorkflowBuilder(general_config=base_cfg.general, registry=None) as builder:
            await builder.populate_builder(base_cfg)
            prompt_optim_fn = builder.get_function(optimizer_config.prompt_optimization_function)
    except Exception as e:
        logger.error("Failed to initialize workflow builder: %s", e)
        raise

    logger.info("Prompt optimization workflow ready")

    # Dictionary to store accumulated feedback for each parameter
    # Key: parameter name, Value: list of feedback strings
    feedback_lists: dict[str, list[str]] = {}

    # Main optimization loop
    while len(study.trials) < optimizer_config.n_trials_prompt:
        trial = study.ask()
        suggestions: dict[str, str] = {}

        logger.info("Starting trial %d/%d", len(study.trials) + 1, optimizer_config.n_trials_prompt)

        # Generate new prompt suggestions for each parameter
        for param, choices in prompt_space.items():
            try:
                # Randomly select a base prompt from existing choices
                prompt_text, purpose = random.choice(choices)

                # Prepare accumulated feedback for this parameter
                oracle_fb = "\n".join(f"**Feedback:** {fb}"
                                      for fb in feedback_lists.get(param, [])) if feedback_lists.get(param) else None

                # Generate optimized prompt using the optimization function
                new_prompt = await prompt_optim_fn.acall_invoke(
                    PromptOptimizerInputSchema(
                        original_prompt=prompt_text,
                        objective=purpose,
                        oracle_feedback=oracle_fb,
                    ))

                # Add the new prompt to available choices
                choices.append((new_prompt, purpose))

                # Let Optuna suggest which prompt to use (by index)
                idx = trial._suggest(
                    param,
                    optuna.distributions.IntDistribution(0, len(choices) - 1),
                )

                # Store the selected prompt for this trial
                suggestions[param] = choices[int(idx)][0]
                trial.set_user_attr("prompt_text", prompt_text)

            except Exception as e:
                logger.error("Error generating prompt for parameter %s: %s", param, e)
                raise

        # Apply the suggested prompts to create a trial configuration
        cfg_trial = apply_suggestions(base_cfg, suggestions)

        # Number of repetitions for this parameter set (for statistical robustness)
        reps = max(1, getattr(optimizer_config, "reps_per_param_set", 1))

        # Define evaluation function for a single run
        async def _single_eval() -> list[tuple[str, Any]]:
            """Runs a single evaluation and returns the results."""
            eval_cfg = EvaluationRunConfig(
                config_file=cfg_trial,
                dataset=opt_run_config.dataset,
                result_json_path=opt_run_config.result_json_path,
                endpoint=opt_run_config.endpoint,
                endpoint_timeout=opt_run_config.endpoint_timeout,
                override=opt_run_config.override,
            )
            return (await EvaluationRun(config=eval_cfg).run_and_evaluate()).evaluation_results

        # Run multiple evaluations in parallel for statistical robustness
        try:

            async def _run_all_evals():
                tasks = [_single_eval() for _ in range(reps)]
                return await asyncio.gather(*tasks)

            all_results = asyncio.run(_run_all_evals())
        except Exception as e:
            logger.error("Error during evaluation runs: %s", e)
            raise

        # Process trajectory evaluation reasoning for feedback collection
        # Map eval output item ID to list of trajectory evaluation reasonings
        eval_output_item_id_to_trajectory_eval_reasoning: dict[str, list[str]] = {}

        # Process each evaluation run's results
        for run_results in all_results:  # iterate over runs
            for name, eval_result in run_results:  # Each run returns list of (name, result) tuples
                if name == trajectory_eval_metric_name:
                    try:
                        # Extract reasoning from trajectory evaluation items
                        for eval_output_item in eval_result.eval_output_items:
                            item_id = eval_output_item.id

                            if item_id not in eval_output_item_id_to_trajectory_eval_reasoning:
                                eval_output_item_id_to_trajectory_eval_reasoning[item_id] = []

                            # Safely extract reasoning - handle different reasoning structures
                            reasoning = eval_output_item.reasoning
                            if isinstance(reasoning, dict) and "reasoning" in reasoning:
                                reasoning_text = reasoning["reasoning"]
                            elif isinstance(reasoning, str):
                                reasoning_text = reasoning
                            else:
                                reasoning_text = str(reasoning)

                            eval_output_item_id_to_trajectory_eval_reasoning[item_id].append(reasoning_text)

                    except Exception as e:
                        logger.warning("Error processing trajectory evaluation reasoning: %s", e)
                    break  # Only process the first matching metric

        # Collect feedback from evaluation results
        logger.info("Processing evaluation results for feedback collection")

        # Process each evaluation run to collect feedback
        for run_results in all_results:  # Fixed: iterate over runs correctly
            for metric_name, direction in zip(eval_metrics, directions):
                # Find the output for this metric in this run
                output = next((result for name, result in run_results if name == metric_name), None)

                if output and hasattr(output, "eval_output_items") and output.eval_output_items:
                    # Sort evaluation items by score (best first based on direction)
                    output.eval_output_items.sort(
                        key=lambda it: it.score,
                        reverse=(direction == "minimize"),  # For minimize: worst first (reverse=True gives us best)
                    )

                    # Collect trajectory feedback from top performers
                    try:
                        trajectory_feedback_parts = []
                        for eval_output_item in output.eval_output_items[:optimizer_config.num_feedback]:
                            if eval_output_item.id in eval_output_item_id_to_trajectory_eval_reasoning:
                                reasoning_list = eval_output_item_id_to_trajectory_eval_reasoning[eval_output_item.id]
                                trajectory_feedback_parts.extend(reasoning_list)

                        trajectory_feedback = "\n".join(trajectory_feedback_parts)

                        # Accumulate feedback
                        if metric_name not in feedback_lists:
                            feedback_lists[metric_name] = []
                        if trajectory_feedback:  # Only add non-empty feedback
                            feedback_lists[metric_name].append(trajectory_feedback)

                    except Exception as e:
                        logger.warning("Error collecting feedback for metric %s: %s", metric_name, e)

        # Aggregate metric values across all runs
        trial_values = []
        for metric_name in eval_metrics:
            try:
                # properly extract scores from the nested structure
                scores = []
                for run_results in all_results:  # Each run is a list of (name, result) tuples
                    for name, result in run_results:
                        if name == metric_name:
                            scores.append(result.average_score)
                            break

                if scores:
                    avg_score = sum(scores) / len(scores)
                    trial_values.append(avg_score)
                else:
                    logger.warning("No scores found for metric %s", metric_name)
                    trial_values.append(0.0)  # Default value

            except Exception as e:
                logger.error("Error aggregating scores for metric %s: %s", metric_name, e)
                trial_values.append(0.0)  # Default value to continue optimization

        # Report trial results to Optuna
        study.tell(trial, trial_values)
        logger.info("Trial %d complete – Pareto set size: %d", len(study.trials), len(study.best_trials))
        logger.info("Trial values: %s", dict(zip(eval_metrics, trial_values)))

        # Save checkpoint of best prompts so far
        try:
            best_trial = pick_trial(
                study=study,
                mode=optimizer_config.multi_objective_combination_mode,
                weights=weights,
            )

            checkpoint = {k: prompt_space[k][v] for k, v in best_trial.params.items()}

            checkpoint_path = out_dir / f"optimized_prompts_round{len(study.trials)}.json"
            with checkpoint_path.open("w") as fh:
                json.dump(checkpoint, fh, indent=4)
            logger.info("Checkpoint saved: %s", checkpoint_path)

        except Exception as e:
            logger.error("Error saving checkpoint: %s", e)

    # Final optimization results
    logger.info("Optimization complete, processing final results")

    try:
        final_trial = pick_trial(
            study=study,
            mode=optimizer_config.multi_objective_combination_mode,
            weights=weights,
        )
        best_prompts = {k: prompt_space[k][v] for k, v in final_trial.params.items()}

        # Save final results
        final_prompts_path = out_dir / "optimized_prompts.json"
        with final_prompts_path.open("w") as fh:
            json.dump(best_prompts, fh, indent=4)

        trials_df_path = out_dir / "trials_dataframe_prompts.csv"
        with trials_df_path.open("w") as fh:
            study.trials_dataframe().to_csv(fh, index=False)

        # Generate Pareto front visualizations
        try:
            logger.info("Generating Pareto front visualizations...")
            create_pareto_visualization(
                data_source=study,
                metric_names=eval_metrics,
                directions=directions,
                output_dir=out_dir / "plots",
                title_prefix="Prompt Optimization",
                show_plots=False  # Don't show plots in automated runs
            )
            logger.info("Pareto visualizations saved to: %s", out_dir / "plots")
        except Exception as e:
            logger.warning("Failed to generate visualizations: %s", e)

        logger.info("Prompt optimization finished successfully!")
        logger.info("Final prompts saved to: %s", final_prompts_path)
        logger.info("Trial data saved to: %s", trials_df_path)
        logger.info("Total trials completed: %d", len(study.trials))
        logger.info("Pareto optimal trials: %d", len(study.best_trials))

    except Exception as e:
        logger.error("Error saving final results: %s", e)
        raise
