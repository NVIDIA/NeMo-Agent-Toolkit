<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# NVIDIA NeMo Agent toolkit Optimizer Guide

Welcome to the NeMo Agent toolkit Optimizer guide. This document provides a comprehensive overview of how to use the NeMo Agent toolkit Optimizer to tune your NeMo Agent toolkit workflows.

## Introduction

### What is Parameter Optimization?

Parameter optimization is the process of automatically finding the best combination of settings (parameters) for your NeMo Agent toolkit workflows. Think of it like tuning a musical instrument – you adjust different knobs and strings until you achieve the perfect sound. Similarly, AI workflows have various "knobs" you can adjust:

- **Hyperparameters**: Numerical settings that control model behavior (such as `temperature`, `top_p`, `max_tokens`)
- **Prompts**: The instructions and context you provide to language models
- **Model choices**: Which specific AI models to use for different tasks
- **Processing parameters**: Settings that affect how data flows through your workflow

### Why Use Parameter Optimization?

Manual parameter tuning has several challenges:

1. **Time-consuming**: Testing different combinations manually can take days or weeks
2. **Suboptimal results**: Humans often miss the best combinations due to the vast search space
3. **Lack of reproducibility**: Manual tuning is hard to document and reproduce
4. **Complex interactions**: Parameters often interact in non-obvious ways

The NeMo Agent toolkit Optimizer solves these problems by:

- **Automating the search process**: Tests hundreds of parameter combinations automatically
- **Using intelligent algorithms**: Employs proven optimization techniques (Optuna for numerical parameters, genetic algorithms for prompts)
- **Balancing multiple objectives**: Optimizes for multiple goals simultaneously (such as accuracy vs. speed)
- **Providing insights**: Generates visualizations and reports to help you understand parameter impacts

### Real-World Example

Imagine you're building a customer service chatbot. You need to optimize:
- The system prompt to get the right tone and behavior
- Model parameters like temperature (creativity vs. consistency)
- Which LLM to use (balancing cost vs. quality)
- Response length limits

Instead of manually testing hundreds of combinations, the optimizer can find the best settings that maximize customer satisfaction while minimizing response time and cost.

### What This Guide Covers

This guide will walk you through:
1. Understanding the core concepts (`OptimizableField` and `SearchSpace`)
2. Configuring which parameters to optimize
3. Setting up the optimization process
4. Running the optimizer
5. Interpreting the results and applying them

## How it Works

The NeMo Agent toolkit Optimizer uses a combination of techniques to find the best parameters for your workflow:

- Numerical Values
  - [Optuna](https://optuna.org/) is used to optimize numerical values.
- Prompts
  - A custom genetic algorithm (GA) is used to optimize prompts. It evolves a population of prompt candidates over multiple generations using LLM-powered mutation and optional recombination.

![Optimizer Flow Chart](../_static/optimizer_flow_chart.png)

The optimization process follows the steps outlined in the diagram above:

1.  **Configuration Loading**: The optimizer starts by reading the `optimizer` section of your workflow configuration file. It uses this to understand your optimization objectives, which parameters are tunable, and the overall optimization strategy.

2.  **Study Initialization**: An [Optuna study](https://optuna.readthedocs.io/en/stable/reference/study.html) is created to manage the optimization process. This study keeps track of all the trials, their parameters, and their resulting scores.

3.  **Optimization Loops**:
    - Numerical parameters: loop for `n_trials_numeric` trials (Optuna).
    - Prompt parameters: loop for `ga_generations` generations (Genetic Algorithm).

4.  **Parameter Suggestion**: In each numeric trial, Optuna's sampler suggests a new set of hyperparameters from the `SearchSpace` you defined with `OptimizableField`. For prompt optimization, a population of prompts is evolved each generation using LLM-powered mutation and optional recombination guided by the `prompt_purpose`. No trajectory feedback is used.

5.  **Workflow Execution**: The NeMo Agent toolkit workflow is executed using the suggested parameters for that trial. This is repeated `reps_per_param_set` times to ensure the results are statistically stable.

6.  **Evaluation**: The output of each workflow run is passed to the evaluators defined in the `eval_metrics` configuration. Each evaluator calculates a score for a specific objective (such as correctness, latency, or creativity).

7.  **Recording Results**:
    - Numeric trials: scores are combined per `multi_objective_combination_mode` and recorded in the Optuna study.
    - Prompt GA: each individual's metrics are normalized per generation and `scalarized` per `multi_objective_combination_mode`; the best individuals are checkpointed each generation.

8.  **Analysis and Output**: Once all trials are complete, the optimizer analyzes the study to find the best-performing trial. It then generates the output files, including `best_params.json` and the various plots, to help you understand the results.

Before diving into configuration, let's understand the fundamental concepts that make parameters optimizable.

## Core Concepts: `OptimizableField` and `SearchSpace`

The optimizer needs to know two things about each parameter:
1. **Which parameters can be optimized** (`OptimizableField`)
2. **What values to try** (`SearchSpace`)

### Understanding `OptimizableField`

An `OptimizableField` is a special type of field in your workflow configuration that tells the optimizer "this parameter can be tuned." It's like putting a label on certain knobs saying "you can adjust this."

For example, in a language model configuration:
- `temperature` might be an OptimizableField (can be tuned)
- `api_key` would be a regular field (should not be tuned)

### Understanding SearchSpaces

A `SearchSpace` defines the range or set of possible values for an optimizable parameter. It answers the question: "What values should the optimizer try?"

There are three main types of search spaces:

1. **Continuous Numerical**: A range of numbers (e.g., temperature from 0.1 to 0.9)
2. **Discrete/Categorical**: A list of specific choices (e.g., model names)
3. **Prompt**: Special search space for optimizing text prompts using AI-powered mutations

### How They Work Together

When you mark a field as optimizable and define its search space, you're telling the optimizer:
- "This parameter affects my workflow's performance"
- "Here are the reasonable values to try"
- "Find the best value within these constraints"

The optimizer will then systematically explore these search spaces to find the optimal combination.

## Implementing `OptimizableField`

To make a parameter in your workflow optimizable, you need to use the `OptimizableField` function instead of Pydantic's standard `Field`. This allows you to attach search space metadata to the field. You may omit the `space` argument to mark a field as optimizable and supply its search space later in the configuration file.

### SearchSpace Model

The `SearchSpace` Pydantic model is used to define the range or set of possible values for a hyperparameter.

-   `values: Sequence[T] | None`: Categorical values for a discrete search space. You can either set `values`. Mutually exclusive with `low` and `high`.
-   `low: T | None`: The lower bound for a numerical parameter.
-   `high: T | None`: The upper bound for a numerical parameter.
-   `log: bool`: Whether to use a logarithmic scale for numerical parameters. Defaults to `False`.
-   `step: float`: The step size for numerical parameters.
-   `is_prompt: bool`: Indicates that this field is a prompt to be optimized. Defaults to `False`.
-   `prompt: str`: The base prompt to be optimized.
-   `prompt_purpose: str`: A description of what the prompt is for, used to guide the LLM-based prompt optimizer.

### `OptimizableField` Function

This function is a drop-in replacement for `pydantic.Field` that optionally takes a `space` argument.

Here's how you can define optimizable fields in your workflow's data models:

```python
from pydantic import BaseModel

from nat.data_models.function import FunctionBaseConfig
from nat.data_models.optimizable import OptimizableField, SearchSpace, OptimizableMixin

class SomeImageAgentConfig(FunctionBaseConfig, OptimizableMixin, name="some_image_agent_config"):
    quality: int = OptimizableField(
        default=90,
        space=SearchSpace(low=75, high=100)
    )
    sharpening: float = OptimizableField(
        default=0.5,
        space=SearchSpace(low=0.0, high=1.0)
    )
    model_name: str = OptimizableField(
        default="gpt-3.5-turbo",
        space=SearchSpace(values=["gpt-3.5-turbo", "gpt-4", "claude-2"]),
        description="The name of the model to use."
    )
    # Option A: Start from a prompt different from the default (set prompt in space)
    system_prompt_a: str = OptimizableField(
        default="You are a helpful assistant.",
        space=SearchSpace(
            is_prompt=True,
            prompt="You are a concise and safety-aware assistant.",
            prompt_purpose="To guide the behavior of the chatbot."
        ),
        description="The system prompt for the LLM."
    )

    # Option B: Start from the field's default prompt (omit prompt in space)
    system_prompt_b: str = OptimizableField(
        default="You are a helpful assistant.",
        space=SearchSpace(
            is_prompt=True,
            # prompt is intentionally omitted; defaults to the field's default
            prompt_purpose="To guide the behavior of the chatbot."
        ),
        description="The system prompt for the LLM."
    )

    # Option C: Mark as optimizable but provide search space in config
    temperature: float = OptimizableField(0.0)
```

In this example:
- `quality` (int) and `sharpening` (float) are continuous parameters.
- `model_name` is a categorical parameter, and the optimizer will choose from the provided list of models.
- `system_prompt_a` demonstrates setting a different starting prompt in the `SearchSpace`.
- `system_prompt_b` demonstrates omitting `SearchSpace.prompt`, which uses the field's default as the base prompt.
- `temperature` shows how to mark a field as optimizable without specifying a search space in code; the search space must then be provided in the workflow configuration.

Behavior for prompt-optimized fields:
- If `space.is_prompt` is `true` and `space.prompt` is `None`, the optimizer will use the `OptimizableField`'s `default` as the base prompt.
- If both `space.prompt` and the field `default` are `None`, an error is raised. Provide at least one.
- If `space` is omitted entirely, a corresponding search space **must** be supplied in the configuration's `search_space` mapping; otherwise a runtime error is raised when walking optimizable fields.

## Enabling Optimization in Configuration Files

Once `OptimizableField`s have been created in your workflow's data models, you need to enable optimization for these fields in your workflow configuration file.
This can be enabled using the `optimizable_params` field of your configuration file.

For example:
```yaml

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    optimizable_params:
      - temperature
      - top_p
      - max_tokens
```

**NOTE:** Ensure your configuration object inherits from `OptimizableMixin` to enable the `optimizable_params` field.

### Overriding Search Spaces in Configuration Files

You can override the search space for any optimizable parameter directly in your workflow configuration by adding a `search_space` mapping alongside `optimizable_params`:

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    optimizable_params: [temperature, top_p]
    search_space:
      temperature:
        low: 0.2
        high: 0.8
        step: 0.2
      top_p:
        low: 0.5
        high: 1.0
        step: 0.1
```

The `search_space` entries are parsed into `SearchSpace` objects and override any defaults defined in the data models.
If a field is marked as optimizable but lacks a `search_space` in both the data model and this mapping, the optimizer will raise an error when collecting optimizable fields.

## Default Optimizable LLM Parameters

Many of the LLM providers in the NeMo Agent Toolkit come with pre-configured optimizable parameters. This means you can start tuning common hyperparameters like `temperature` and `top_p` without any extra configuration.

Here is a matrix of the default optimizable parameters for some of the built-in LLM providers:

| Parameter     | Provider | Default Value | Search Space                       |
|:--------------|:---------|:--------------|:-----------------------------------|
| `temperature` | `openai` | `0.0`         | `low=0.1`, `high=0.8`, `step=0.2`  |
|               | `nim`    | `0.0`         | `low=0.1`, `high=0.8`, `step=0.2`  |
| `top_p`       | `openai` | `1.0`         | `low=0.5`, `high=1.0`, `step=0.1`  |
|               | `nim`    | `1.0`         | `low=0.5`, `high=1.0`, `step=0.1`  |
| `max_tokens`  | `nim`    | `300`         | `low=128`, `high=2176`, `step=512` |

To use these defaults, you just need to enable numeric optimization in your `config.yml`. The optimizer will automatically find these `OptimizableField`s in the LLM configuration and start tuning them. You can always override these defaults by defining your own `OptimizableField` on the LLM configuration in your workflow.

## Optimizer Configuration

Now that you understand how to make fields optimizable, let's look at how to configure the optimization process itself.

The optimizer is configured through an `optimizer` section in your workflow's YAML configuration file. This configuration is mapped to the `OptimizerConfig` and `OptimizerMetric` Pydantic models.

Here is an example of an `optimizer` section in a YAML configuration file:

```yaml
optimizer:
  output_path: "optimizer_results"

  # Numeric (Optuna)
  numeric:
    enabled: true
    n_trials: 50

  # Prompt (Genetic Algorithm)
  prompt:
    enabled: true
    prompt_population_init_function: "prompt_optimizer"
    prompt_recombination_function: "prompt_recombiner"  # optional
    ga_population_size: 16
    ga_generations: 8
    ga_offspring_size: 12        # optional; defaults to pop_size - elitism
    ga_crossover_rate: 0.7
    ga_mutation_rate: 0.2
    ga_elitism: 2
    ga_selection_method: "tournament"  # or "roulette"
    ga_tournament_size: 3
    ga_parallel_evaluations: 8
    ga_diversity_lambda: 0.0

  # Evaluation
  reps_per_param_set: 5
  eval_metrics:
    latency:
      evaluator_name: "latency"
      direction: "minimize"
      weight: 0.2
    correctness:
      evaluator_name: "correctness"
      direction: "maximize"
      weight: 0.8
```

### `OptimizerConfig`

This is the main configuration object for the optimizer.

-   `output_path: Path | None`: The directory where optimization results will be saved, for example, `optimizer_results/`. Defaults to `None`.
-   `eval_metrics: dict[str, OptimizerMetric] | None`: A dictionary of evaluation metrics to optimize. The keys are custom names for the metrics, and the values are `OptimizerMetric` objects.
-   `numeric.enabled: bool`: Enable numeric optimization (Optuna). Defaults to `true`.
-   `numeric.n_trials: int`: Number of numeric trials. Defaults to `20`.
-   `prompt.enabled: bool`: Enable GA-based prompt optimization. Defaults to `false`.
-   `prompt.ga_population_size: int`: Population size for GA prompt optimization. Larger populations increase diversity but cost more per generation. Defaults to `10`.
-   `prompt.ga_generations: int`: Number of generations for GA prompt optimization. Replaces `n_trials_prompt`. Defaults to `5`.
-   `prompt.ga_offspring_size: int | null`: Number of offspring produced per generation. If `null`, defaults to `ga_population_size - ga_elitism`.
-   `prompt.ga_crossover_rate: float`: Probability of recombination between two parents for each prompt parameter. Defaults to `0.7`.
-   `prompt.ga_mutation_rate: float`: Probability of mutating a child's prompt parameter using the LLM optimizer. Defaults to `0.1`.
-   `prompt.ga_elitism: int`: Number of elite individuals copied unchanged to the next generation. Defaults to `1`.
-   `prompt.ga_selection_method: str`: Parent selection scheme. `tournament` (default) or `roulette`.
-   `prompt.ga_tournament_size: int`: Tournament size when `ga_selection_method` is `tournament`. Defaults to `3`.
-   `prompt.ga_parallel_evaluations: int`: Maximum number of concurrent evaluations. Controls async concurrency. Defaults to `8`.
-   `prompt.ga_diversity_lambda: float`: Diversity penalty strength to discourage duplicate prompt sets. `0.0` disables it. Defaults to `0.0`.
-   `prompt.prompt_population_init_function: str | null`: Function name used to mutate base prompts to seed the initial population and perform mutations. The NeMo Agent Toolkit includes a built-in `prompt_init` Function located in the {py:mod}`~nat.agent.prompt_optimizer.register` file you can use in your configurations. 
-   `prompt.prompt_recombination_function: str | null`: Optional function name used to recombine two parent prompts into a child prompt. The NeMo Agent Toolkit includes a built-in `prompt_recombiner` Function located in the {py:mod}`~nat.agent.prompt_optimizer.register` file you can use in your configurations. 
-   `reps_per_param_set: int`: The number of times to run the workflow for each set of parameters to get a more stable evaluation. This is important for noisy evaluations where the result might vary even with the same parameters. Defaults to `3`.
-   `target: float | None`: If set, the optimization will stop when the combined score for a trial reaches this value. This is useful if you have a specific performance target and want to save time. The score is normalized between 0 and 1. Defaults to `None`.
-   `multi_objective_combination_mode: str`: How to combine multiple objective scores into a single scalar. Supported: `harmonic`, `sum`, `chebyshev`. Defaults to `harmonic`.

### `OptimizerMetric`

This model defines a single metric to be used in the optimization.

-   `evaluator_name: str`: The name of the evaluator to use for this metric. This should correspond to a registered evaluator in the system.
-   `direction: str`: The direction of optimization. Must be either `maximize` or `minimize`.
-   `weight: float`: The weight of this metric in the multi-objective optimization. The weights will be normalized. Defaults to `1.0`.


### How Genetic Prompt Optimization Works in Practice

1. Start with an initial population of prompt variations
2. Evaluate each prompt's performance using your metrics
3. Select the best performers as parents
4. Create new prompts through mutation and crossover
5. Replace the old population with the new one
6. Repeat until you find optimal prompts

This evolutionary approach is particularly effective for prompt optimization because it can explore creative combinations while gradually improving performance.

Before diving into prompt optimization, let's clarify the genetic algorithm (GA) terminology used throughout this guide. Genetic algorithms are inspired by natural evolution and use biological metaphors:

### Key GA Concepts

**Population**: A collection of candidate solutions (in our case, different prompt variations). Think of it as a group of individuals, each representing a different approach to solving your problem.

**Individual**: A single candidate solution - one specific set of prompts being evaluated.

**Generation**: One iteration of the evolutionary process. Each generation produces a new population based on the performance of the previous one.

**Fitness**: A score indicating how well an individual performs according to your evaluation metrics. Higher fitness means better performance.

**Parents**: Individuals selected from the current generation to create new individuals for the next generation. Better-performing individuals are more likely to be selected as parents.

**Offspring/Children**: New individuals created by combining aspects of parent individuals or by mutating existing ones.

**Mutation**: Random changes applied to an individual to introduce variety. In prompt optimization, this means using an LLM to intelligently modify prompts.

**Crossover/Recombination**: Combining features from two parent individuals to create a child. For prompts, this might mean taking the structure from one prompt and the tone from another.

**Elitism**: Preserving the best individuals from one generation to the next without modification, ensuring we don't lose good solutions.

**Selection Methods**:
- **Tournament Selection**: Randomly select a small group and choose the best performer
- **Roulette Selection**: Select individuals with probability proportional to their fitness

## Prompt Optimization with Genetic Algorithm (GA)

This section explains how the GA evolves prompt parameters when `do_prompt_optimization` is enabled.

### Workflow

1. Seed an initial population:
   - The first individual uses your original prompts.
   - The remaining `ga_population_size - 1` individuals are created by applying `prompt_population_init_function` to each prompt parameter with its `prompt_purpose`.
2. Evaluate all individuals with your configured `eval_metrics` and `reps_per_param_set`. Metrics are averaged per evaluator.
3. Normalize metrics per generation so that higher is always better, respecting each metric's `direction`.
4. `Scalarize` normalized scores per `multi_objective_combination_mode` to compute a fitness value. Optionally subtract a diversity penalty if `ga_diversity_lambda > 0`.
5. Create the next generation:
   - Elitism: carry over the top `ga_elitism` individuals.
   - Selection: choose parents using `ga_selection_method` (`tournament` with `ga_tournament_size`, or `roulette`).
   - Crossover: with probability `ga_crossover_rate`, recombine two parent prompts for a parameter using `prompt_recombination_function` (if provided), otherwise pick from a parent.
   - Mutation: with probability `ga_mutation_rate`, apply `prompt_population_init_function` to mutate the child's parameter.
   - Repeat until the new population reaches `ga_population_size` (or `ga_offspring_size` offspring plus elites).
6. Repeat steps 2–5 for `ga_generations` generations.

All LLM calls and evaluations are executed asynchronously with a concurrency limit of `ga_parallel_evaluations`.

---

> ### 🎯 Tuning Guidance
>
> **Population and Generations**
> - `ga_population_size`, `ga_generations`: Increase to explore more of the search space at higher cost.
> - **Tip**: Start with 10-16 population size and 5-8 generations for quick testing.
>
> **Crossover and Mutation**
> - `ga_crossover_rate`: Higher crossover helps combine good parts of prompts.
> - `ga_mutation_rate`: Higher mutation increases exploration.
> - **Tip**: Use 0.7 for crossover and 0.2 for mutation as balanced starting points.
>
> **Elitism**
> - `ga_elitism`: Preserves top performers; too high can reduce diversity.
> - **Tip**: Keep at 1-2 for most cases.
>
> **Selection Method**
> - `ga_selection_method`, `ga_tournament_size`: Tournament is robust; larger tournaments increase selection pressure.
> - **Tip**: Use tournament selection with size 3 for balanced exploration.
>
> **Diversity**
> - `ga_diversity_lambda`: Penalizes duplicate prompt sets to encourage variety.
> - **Tip**: Start at 0.0, increase to 0.2 if seeing too many similar prompts.
>
> **Concurrency**
> - `ga_parallel_evaluations`: Tune based on your environment to balance throughput and rate limits.
> - **Tip**: Start with 8 and increase until hitting rate limits.

---

### Outputs

During GA prompt optimization, the optimizer saves:

- `optimized_prompts_gen<N>.json`: Best prompt set after generation N.
- `optimized_prompts.json`: Final best prompt set after all generations.
- `ga_history_prompts.csv`: Per-individual fitness and metric history across generations.

Numeric optimization outputs (Optuna) remain unchanged and can be used alongside GA outputs.

## Running the Optimizer

Once you have your optimizer configuration and optimizable fields set up, you can run the optimizer from the command line using the `nat optimize` command.

### CLI Command

```bash
nat optimize --config_file <path_to_config>
```

### Options

-   `--config_file`: (Required) Path to the JSON or YAML configuration file for your workflow, for example, `config.yaml`. This file should contain the `optimizer` section as described above.
-   `--dataset`: (Optional) Path to a JSON file containing the dataset for evaluation, such as `eval_dataset.json`. This will override any dataset path specified in the config file. The dataset should be a list of dictionaries, where each dictionary represents a data point and includes the necessary inputs for your workflow and the ground truth for evaluation.
-   `--result_json_path`: A `JSONPath` expression to extract the result from the workflow's output. Defaults to `$`.
-   `--endpoint`: If you are running your workflow as a service, you can provide the endpoint URL. For example, `http://localhost:8000/generate`.
-   `--endpoint_timeout`: The timeout in seconds for requests to the endpoint. Defaults to `300`.

Example:
```bash
nat optimize --config_file <path to configuraiton file>
```

This command will start the optimization process. You will see logs in your terminal showing the progress of the optimization, including the parameters being tested and the scores for each trial.

## Understanding the Output

When the optimizer finishes, it will save the results in the directory specified by the `output_path` in your `OptimizerConfig`. This directory will contain several files:

-   `optimized_config.yml`: Tuned configuration derived from the selected trial.
-   `trials_dataframe_params.csv`: Full Optuna trials `dataframe` (`values`, `params`, `timings`, `rep_scores`).
-   `pareto_front_2d.png`: 2D Pareto front (when 2 metrics).
-   `pareto_parallel_coordinates.png`: Parallel coordinates plot.
-   `pareto_pairwise_matrix.png`: Pairwise metric matrix.

By examining these output files, you can understand the results of the optimization, choose the best parameters for your needs (for example, picking a point on the Pareto front that represents your desired trade-off), and gain insights into your workflow's behavior.

### Understanding the Pareto Visualizations

The optimizer generates three types of visualizations to help you understand the trade-offs between different objectives:

#### 1. 2D Pareto Front (`pareto_front_2d.png`)
*Generated only when optimizing exactly 2 metrics, for example in ![this image](../_static/pareto_front_2d.png)*

This scatter plot shows:
- **Light blue dots**: All trials tested during optimization
- **Red stars**: Pareto optimal trials (solutions where improving one metric would worsen another)
- **Red dashed line**: The Pareto front connecting optimal solutions

**How to interpret**:
- The arrows (↑ or ↓) indicate the direction of improvement for each metric
- For "maximize" metrics, higher values are better (look up/right)
- For "minimize" metrics, lower values are better (look down/left)
- Points on the Pareto front represent different trade-offs - choose based on your priorities

**Example**: If optimizing accuracy (maximize) vs latency (minimize), the ideal point would be top-left (high accuracy, low latency). The Pareto front shows the best achievable trade-offs.

#### 2. Parallel Coordinates Plot (`pareto_parallel_coordinates.png`)
*Works with any number of metrics, for example in ![this image](../_static/pareto_parallel_coordinates.png)*

This plot normalizes all metrics to a 0-1 scale where higher is always better:
- **Blue lines**: All trials (shown with low opacity)
- **Red lines**: Pareto optimal trials (shown with high opacity)
- **Y-axis**: Normalized performance (0 = worst, 1 = best)
- **X-axis**: Different metrics with their optimization direction

**How to interpret**:
- Each line represents one complete parameter configuration
- Follow a line across to see how it performs on each metric
- Parallel lines indicate independent metrics
- Crossing lines suggest trade-offs between metrics
- The best solutions have lines staying high across all metrics

**Choosing a solution**: Look for red lines that maintain good performance (stay high) across the metrics you care most about.

#### 3. Pairwise Matrix Plot (`pareto_pairwise_matrix.png`)
*Provides detailed metric relationships, for example in ![this image](../_static/pareto_pairwise_matrix.png)*

This matrix visualization shows:
- **Diagonal cells (histograms)**: Distribution of values for each individual metric
  - Light blue bars: All trials
  - Red bars: Pareto optimal trials
  - Shows the range and frequency of values achieved
- **Off-diagonal cells (scatter plots)**: Relationships between pairs of metrics
  - Light blue dots: All trials
  - Red stars: Pareto optimal trials
  - Reveals correlations and trade-offs between metrics

**How to interpret**:
- **Histograms**: Check if Pareto optimal solutions (red) cluster at desirable values
- **Scatter plots**: Look for patterns:
  - Positive correlation: Metrics improve together (dots trend up-right)
  - Negative correlation: Trade-off exists (dots trend down-right)
  - No correlation: Metrics are independent (random scatter)

**Example interpretation**: If the accuracy-latency scatter shows a negative correlation, it confirms that improving accuracy typically increases latency.

### Selecting the Best Configuration

1. **Identify your priorities**: Decide which metrics matter most for your use case
2. **Examine the Pareto visualizations**: Look for configurations that excel in your priority metrics
3. **Find the trial number**: Use the `trials_dataframe_params.csv` to identify specific trial numbers
4. **Use the configuration**: Load the corresponding `config_numeric_trial_N.yml` file

**Example decision process**:
- If latency is critical: Choose a Pareto optimal point with the lowest latency that still meets your accuracy requirements
- If accuracy is paramount: Select the highest accuracy configuration and accept the latency trade-off
- For balanced performance: Pick a point in the middle of the Pareto front 

## A Complete Example of Optimization

For a complete example of using the optimizer, see the `email_phishing_analyzer` example in the `evaluation_and_profiling` section of the examples in the NeMo Agent toolkit repository.

## Best Practices and Tuning Guide

### Choosing Optimizer Parameters

#### For Numeric Optimization (Optuna)

**Number of Trials (`n_trials`)**:
- Start with 20-50 trials for initial exploration
- Increase to 100-200 for production optimization
- More trials = better results but higher cost
- Use early stopping with `target` parameter to save time

**Repetitions (`reps_per_param_set`)**:
- Use 3-5 `reps` for deterministic workflows
- Increase to 10-20 for highly stochastic outputs
- Higher `reps` reduce noise but increase cost

#### For Prompt Optimization (GA)

**Population Size (`ga_population_size`)**:
- Start with 10-20 individuals
- Larger populations explore more diversity
- Cost scales linearly with population size

**Generations (`ga_generations`)**:
- 5-10 generations often sufficient for convergence
- Monitor fitness improvement across generations
- Stop early if fitness plateaus

**Mutation vs. Crossover**:
- High mutation rate (0.2-0.3): More exploration, good for initial search
- High crossover rate (0.7-0.8): More exploitation, good when you have good candidates
- Balance both for optimal results

**Selection Pressure**:
- Tournament size 2-3: Low pressure, maintains diversity
- Tournament size 5-7: High pressure, faster convergence
- Elitism 1-2: Preserves best solutions without reducing diversity

### Interpreting Optimization Results

#### Understanding Pareto Fronts

The Pareto front visualization shows trade-offs between objectives:
- Points on the front are optimal (no other point is better in all metrics)
- Points closer to the top-right are generally better
- Choose based on your priorities (e.g., accuracy vs. speed)

#### Reading the Trials DataFrame

Look for patterns:
- Which parameters have the most impact?
- Are certain parameter ranges consistently better?
- Is there high variance in certain configurations?

#### Analyzing Parallel Coordinates

This plot helps identify parameter relationships:
- Parallel lines indicate independent parameters
- Crossing lines suggest parameter interactions
- Color intensity shows performance (darker = better)

### Common Pitfalls and Solutions

**Problem**: Optimization converges too quickly to suboptimal solutions
- **Solution**: Increase population diversity, reduce selection pressure, increase mutation rate

**Problem**: High variance in evaluation metrics
- **Solution**: Increase `reps_per_param_set`, ensure consistent evaluation conditions

**Problem**: Optimization is too expensive
- **Solution**: Reduce search space, use `step` for discrete parameters, set `target` for early stopping

**Problem**: Prompt optimization produces similar outputs
- **Solution**: Increase `ga_diversity_lambda`, ensure `prompt_purpose` is specific and actionable

### Multi-Objective Optimization Strategies

**Harmonic Mean** (default):
- Balances all objectives
- Penalizes poor performance in any metric
- Good for ensuring minimum quality across all metrics

**Sum**:
- Simple addition of weighted scores
- Allows compensation (good in one metric offsets bad in another)
- Use when total performance matters more than balance

**`Chebyshev`**:
- Minimizes worst-case deviation from ideal
- Good for risk-averse optimization
- Ensures no metric is too far from optimal

### Workflow-Specific Tips

**For Classification Tasks**:
- Prioritize accuracy or score with high weight (0.7-0.9)
- Include latency with lower weight (0.1-0.3)
- Use 5-10 `reps` to handle class imbalance

**For Generation Tasks**:
- Balance quality metrics (coherence, relevance) equally
- Include diversity metrics to avoid mode collapse
- Use prompt optimization for style or tone control

**For Real-time Applications**:
- Set strict latency targets
- Use `Chebyshev` combination to ensure consistency
- Consider p95 latency instead of mean

### Advanced Techniques

**Staged Optimization**:
1. First optimize prompts with small population or generations
2. Fix best prompts, then optimize numeric parameters
3. Finally, fine-tune both together

**Transfer Learning**:
- Start with parameters from similar optimized workflows
- Use previous optimization results to set tighter search spaces
- Reduces optimization time significantly
