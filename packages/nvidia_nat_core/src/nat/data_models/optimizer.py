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

import typing
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .common import BaseModelRegistryTag
from .common import TypedBaseModel


class OptimizerStrategyBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    """Base for optimizer strategy configs (numeric, prompt) registered in the optimizer registry."""

    enabled: bool = Field(default=False, description="Enable this optimizer strategy.")


class OptimizerMetric(BaseModel):
    """
    Parameters used by the workflow optimizer to define a metric to optimize.
    """
    evaluator_name: str = Field(description="Name of the metric to optimize.")
    direction: str = Field(description="Direction of the optimization. Can be 'maximize' or 'minimize'.")
    weight: float = Field(description="Weight of the metric in the optimization process.", default=1.0)


class SamplerType(StrEnum):
    BAYESIAN = "bayesian"
    GRID = "grid"


class NumericOptimizationConfig(OptimizerStrategyBaseConfig, name="numeric"):
    """
    Configuration for numeric/enum optimization (Optuna).
    """

    enabled: bool = Field(default=True, description="Enable numeric optimization")
    n_trials: int = Field(description="Number of trials for numeric optimization.", default=20)
    sampler: SamplerType | None = Field(
        default=None,
        description="Sampling strategy for numeric optimization. Options: None or 'bayesian' uses \
            the Optuna default (TPE for single-objective, NSGA-II for multi-objective) or 'grid' performs \
            exhaustive grid search over parameter combinations. Defaults to None.",
    )


class PromptGAOptimizationConfig(OptimizerStrategyBaseConfig, name="ga"):
    """
    Configuration for prompt optimization using a Genetic Algorithm.
    Oracle feedback and other implementation-specific options are not part of this
    shared contract; they may be passed as extra fields and stored in model_extra.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(default=False, description="Enable GA-based prompt optimization")

    # Prompt optimization function hooks
    prompt_population_init_function: str | None = Field(
        default=None,
        description="Optional function name to initialize/mutate candidate prompts.",
    )
    prompt_recombination_function: str | None = Field(
        default=None,
        description="Optional function name to recombine two parent prompts into a child.",
    )

    # Genetic algorithm configuration
    ga_population_size: int = Field(
        description="Population size for genetic algorithm prompt optimization.",
        default=24,
    )
    ga_generations: int = Field(
        description="Number of generations to evolve in GA prompt optimization.",
        default=15,
    )
    ga_offspring_size: int | None = Field(
        description="Number of offspring to produce per generation. Defaults to population_size - elitism.",
        default=None,
    )
    ga_crossover_rate: float = Field(
        description="Probability of applying crossover during reproduction.",
        default=0.8,
        ge=0.0,
        le=1.0,
    )
    ga_mutation_rate: float = Field(
        description="Probability of mutating a child after crossover.",
        default=0.3,
        ge=0.0,
        le=1.0,
    )
    ga_elitism: int = Field(
        description="Number of top individuals carried over unchanged each generation.",
        default=2,
    )
    ga_selection_method: str = Field(
        description="Parent selection strategy: 'tournament' or 'roulette'.",
        default="tournament",
    )
    ga_tournament_size: int = Field(
        description="Tournament size when using tournament selection.",
        default=3,
    )
    ga_parallel_evaluations: int = Field(
        description="Max number of individuals to evaluate concurrently per generation.",
        default=8,
    )
    ga_diversity_lambda: float = Field(
        description="Strength of diversity penalty (0 disables). Penalizes identical/near-identical prompts.",
        default=0.0,
        ge=0.0,
    )


class BaseOptimizerConfig(BaseModel):
    """
    Shared optimizer parameters: only fields that any optimizer strategy
    (or any future GA implementation) could reuse. Strategy-specific config
    lives on subtypes (e.g. GAOptimizerConfig adds .prompt; parameter optimizer
    uses .numeric on OptimizerConfig).
    """
    output_path: Path | None = Field(
        default=None,
        description="Path to the output directory where the results will be saved.",
    )

    eval_metrics: dict[str, OptimizerMetric] | None = Field(
        description="List of evaluation metrics to optimize.",
        default=None,
    )

    reps_per_param_set: int = Field(
        default=3,
        description="Number of repetitions per parameter set for the optimization.",
    )

    target: float | None = Field(
        description=(
            "Target value for the optimization. If set, the optimization will stop when this value is reached."),
        default=None,
    )

    multi_objective_combination_mode: str = Field(
        description="Method to combine multiple objectives into a single score.",
        default="harmonic",
    )


class OptimizerConfig(BaseOptimizerConfig):
    """
    Full optimizer config used in the app Config and for parsing YAML.
    Extends the shared base with strategy-specific nests: .numeric for
    parameter/Optuna optimization, .prompt for GA prompt optimization.
    Runners use subtypes (e.g. GAOptimizerConfig) to declare the shape they need.
    """
    numeric: NumericOptimizationConfig = NumericOptimizationConfig()
    prompt: PromptGAOptimizationConfig = PromptGAOptimizationConfig()


OptimizerStrategyBaseConfigT = typing.TypeVar("OptimizerStrategyBaseConfigT", bound=OptimizerStrategyBaseConfig)


class OptimizerRunConfig(BaseModel):
    """
    Parameters used for an Optimizer R=run
    """
    # Eval parameters

    config_file: Path | BaseModel  # allow for instantiated configs to be passed in
    dataset: str | Path | None  # dataset file path can be specified in the config file
    result_json_path: str = "$"
    endpoint: str | None = None  # only used when running the workflow remotely
    endpoint_timeout: int = 300
    override: tuple[tuple[str, str], ...] = ()
