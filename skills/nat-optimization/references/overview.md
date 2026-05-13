<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# What `nat optimize` Can Optimize

Two complementary tuning targets, each with its own search method:

- **LLM hyperparameters** (Optuna, Bayesian) — fast and cheap; always try first.
- **System and user prompts** (Genetic Algorithm) — significantly slower; requires Python config classes.

## Prerequisites

The optimizer is shipped as the `[config-optimizer]` extra and is **not** in the base install. See [`../../nat-installation/references/installation.md`](../../nat-installation/references/installation.md) for the install command.

If `nat optimize` returns `Error: No such command 'optimize'`, the extra isn't installed yet — install it and retry.

## LLM Hyperparameters

Tuned by **Optuna** (Bayesian search). Fast and cheap — always try this first.

| Parameter | What it controls | Default search space | Provider |
| --- | --- | --- | --- |
| `temperature` | Randomness of output — higher values produce more varied responses | 0.1 → 0.8, step 0.2 | openai, nim |
| `top_p` | Nucleus sampling threshold — lower values focus on the most probable tokens | 0.5 → 1.0, step 0.1 | openai, nim |
| `max_tokens` | Maximum output length | 128 → 2176, step 512 | nim only |
| `model_name` | Which model to use — categorical choice across a list of candidates | custom `values` list | any |
| Custom fields | Any numeric or categorical field in your config class | custom `low`/`high` or `values` | any |

**What it needs:**

Add `optimizable_params` to the LLM block in `workflow.yaml`. Built-in parameters use their default search spaces automatically — no `search_space` block required unless you want to override them:

```yaml
llms:
  my_llm:
    _type: openai
    model_name: <your-model-name>
    base_url: <your-base-url>
    api_key: $YOUR_API_KEY
    temperature: 0.0
    optimizable_params: [temperature, top_p]
```

To override the search space, use a custom range, or optimize a categorical parameter:

```yaml
    optimizable_params: [temperature, model_name]
    search_space:
      temperature:
        low: 0.1
        high: 0.8
        step: 0.2
        log: false      # set true to sample on a log scale
      model_name:
        values: ["model-a", "model-b"]   # categorical choices
```

**SearchSpace fields:**

| Field | Type | When to use |
| --- | --- | --- |
| `low` | float | Lower bound for numeric parameter |
| `high` | float | Upper bound for numeric parameter |
| `step` | float | Sampling step size (optional) |
| `log` | bool | Sample on a log scale — useful for learning rates |
| `values` | list | Categorical choices (mutually exclusive with low/high) |
| `is_prompt` | bool | `true` for string fields tuned by the GA |
| `prompt` | str | Base prompt text to evolve (optional if using field default) |
| `prompt_purpose` | str | Guides the LLM during mutation — be specific |

> **Note:** Some models do not support `temperature` and `top_p` together. Setting both can cause validation errors. If trials fail, start with `temperature` only and add `top_p` separately once you confirm the model accepts it.

## System and User Prompts

Tuned by a **Genetic Algorithm (GA)**. The optimizer mutates and recombines prompt text using an LLM across multiple generations. Significantly slower than numeric optimization — try hyperparameters first.

**What it does:** starts from your existing prompts, generates variants (mutations + crossover), evaluates each generation, and keeps the best-performing prompts.

**What it needs:**

Prompts must be exposed via `OptimizableField` in a Python config class. They cannot be marked as optimizable from YAML alone:

```python
from nat.data_models.optimizable import OptimizableField, OptimizableMixin, SearchSpace
from nat.data_models.function import FunctionBaseConfig

class MyAgentConfig(FunctionBaseConfig, OptimizableMixin, name="my_agent"):
    system_prompt: str = OptimizableField(
        default="You are a helpful assistant.",
        space=SearchSpace(
            is_prompt=True,
            prompt_purpose="Describe the agent's role precisely — the GA uses this to guide prompt mutation.",
        ),
    )
```

Make `prompt_purpose` specific — it directly guides the LLM during mutation. A vague purpose produces generic variants that don't improve quality.

File-loaded prompts (`file://prompts/system.j2`) are also supported and automatically included if the field name ends in `prompt`. Supported extensions: `.txt`, `.md`, `.j2`, `.jinja2`, `.jinja`, `.prompt`, `.tpl`, `.template`.

**Oracle feedback:**

The GA can use an LLM to explain why specific outputs failed and incorporate that feedback into subsequent mutations:

| Mode | Behavior |
| --- | --- |
| `never` | No oracle feedback (default) |
| `always` | Gather feedback on every sample |
| `failing_only` | Gather feedback only on failing samples |
| `adaptive` | Dynamically decides when feedback is needed |

| Field | Default | Description |
| --- | --- | --- |
| `oracle_feedback_mode` | `never` | When to gather feedback: `never`, `always`, `failing_only`, `adaptive` |
| `oracle_feedback_worst_n` | `5` | Number of worst-performing samples to extract feedback from |
| `oracle_feedback_max_chars` | `4000` | Character limit for feedback injected into the mutation prompt |
| `oracle_feedback_fitness_threshold` | `0.3` | Fitness ceiling that triggers `failing_only` mode |
| `oracle_feedback_stagnation_generations` | `3` | Generations without improvement before `adaptive` triggers |
| `oracle_feedback_fitness_variance_threshold` | `0.01` | Variance threshold for `adaptive` collapse detection |
| `oracle_feedback_diversity_threshold` | `0.5` | Prompt duplication ratio threshold for `adaptive` mode |

## Next: sizing the run

Before running `nat optimize`, read [`choosing-parameters.md`](choosing-parameters.md) to pick `sampler`, `n_trials`, `reps_per_param_set`, GA budget, and concurrency. Sizing decides whether the run finishes in minutes or hours, and whether the search produces a useful result — defaults are a good starting point but worth tuning before the first run, not after.
