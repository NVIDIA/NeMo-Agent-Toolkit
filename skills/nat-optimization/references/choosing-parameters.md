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

# Choosing Optimizer Parameters

`nat optimize` has several config parameters that shape a run. This file covers how to pick them and which choice is best in each case.

Defaults are a good starting point for most workflows — tune when there's a concrete reason (different sampler, deterministic agent, larger search space, endpoint capacity limits).

---

## Parameter reference

| Parameter | Default | Description |
| --- | --- | --- |
| `n_trials` | `20` | Number of Optuna trials |
| `sampler` | `bayesian` | `bayesian` (TPE/NSGA-II), `grid`, or `random` |
| `target` | — | Stop early when this combined score is reached |
| `reps_per_param_set` | `3` | Repetitions per trial; `1` for deterministic agents |
| `ga_population_size` | `24` | Individuals per generation in the prompt GA |
| `ga_generations` | `15` | Number of generations to evolve in the prompt GA |
| `ga_crossover_rate` | `0.8` | Probability of applying crossover during reproduction |
| `ga_mutation_rate` | `0.3` | Probability of mutating a child after crossover |
| `ga_elitism` | `2` | Top individuals carried over unchanged each generation |
| `ga_parallel_evaluations` | `8` | Concurrent GA individual evaluations |
| `eval.general.max_concurrency` | `8` | Parallel dataset items within each eval |
| `ga_diversity_lambda` | `0.0` | Diversity pressure in the GA |
| `prompt_population_init_function` | — | Custom function for initial population seeding |
| `prompt_recombination_function` | — | Custom function for combining parent prompts |
| `multi_objective_combination_mode` | `harmonic` | How to combine multiple metrics: `harmonic`, `sum`, or `chebyshev` |

The sections below explain how to pick values for each.

---

## Choosing a Sampler

NeMo Agent toolkit exposes three Optuna samplers via `optimizer.numeric.sampler`:

| Sampler | When to use | Strengths | Weaknesses |
| --- | --- | --- | --- |
| `bayesian` *(default)* | Continuous or mixed-type spaces, 2+ parameters, any single or multi-objective setup. Uses **TPE** for single-objective and **NSGA-II** for multi-objective (auto-selected based on `eval_metrics`). | Learns from past trials — exploits good regions, explores gaps | Needs ~10 random startup trials before the model helps; under ~15 trials it behaves like random search |
| `grid` | Small discrete spaces where exhaustive coverage is feasible (e.g. 1 param × 4 values, or 2 params × 3×3 = 9 values). | Exhaustive — no chance of missing the optimum | Scales badly with dimensions; ignores learning signal from past trials |
| `random` | Baseline for comparison, or very high-dimensional spaces where TPE's model offers little leverage. | No startup cost, embarrassingly parallel | No learning — pure luck |

**Rule of thumb:** keep the default `bayesian` unless the search space is discrete and small enough that `grid` covers it completely.

---

## Sizing `n_trials`

Pick `n_trials` based on what the sampler needs:

| Sampler | Minimum useful | Sweet spot |
| --- | --- | --- |
| `bayesian` | **15** — below this you're paying for TPE but getting random search | 20–30 for 1–2 params; 40–60 for 3+ |
| `grid` | `= \|search_space\|` (one trial per combination) | Same — more trials just repeat existing combinations |
| `random` | 15–30 | Scales with search volume; prefer `bayesian` once you've validated the space |

For the default `bayesian` sampler: **15 is the floor, 20 is the default, 30+ for larger search spaces.**

> **Never kill `nat optimize` mid-run to cut trials short.** The optimizer only produces its final artifacts when the study finishes cleanly. If a run is taking too long, raise parallelism (next section) — or lower `n_trials` in the config and restart if you're certain the search was oversized.

---

## Choosing `reps_per_param_set`

Each trial evaluates one parameter set on the full dataset `reps_per_param_set` times and averages the scores. Default: `3`.

| Case | Use | Why |
| --- | --- | --- |
| `temperature: 0` and fixed (not in `optimizable_params`) | `1` | Each trial runs a deterministic config — extra reps produce identical scores |
| `temperature` is being optimized (or fixed at any non-zero value) | `3` | Non-zero temperatures introduce sampling variance; `reps: 1` means each trial's score is one lucky (or unlucky) sample |
| Workflow uses noisy tools (web search, flaky APIs, LLM-as-judge evaluators) | `3–5` | High run-to-run variance needs more averaging regardless of which parameters are swept |

Don't base this on the baseline config's current temperature value — base it on what will actually be used during trials. A workflow with `temperature: 0` in the baseline but `optimizable_params: [temperature]` is **not** deterministic during optimization, because the optimizer sweeps non-zero values.

---

## Sizing the Prompt GA

Keep the defaults unless your case needs the deviation below or you're told otherwise. Each generation evaluates `pop` prompt variants, selects parents by fitness, breeds children via crossover and mutation, and carries the top `elitism` individuals forward unchanged. Shrinking the budget doesn't make the run faster — it breaks the GA:

- **Smaller `pop`** → fewer distinct variants to recombine. After one selection round, only the top few prompts survive; crossover then produces near-clones each generation. With `pop: 4`, the GA collapses to a single variant slowly mutating — no search.
- **Fewer `gen`** → the search doesn't converge. Generations 1–3 are exploratory (random mutations spread the population); refinement happens around generations 8–15. With `gen: 2`, you barely finish initialization.
- **`elitism: 1`** → the single best can be wiped out by a bad mutation between generations. Two elites give the runner-up as genetic backup; the GA can't regress.

If the run is slow, raise parallelism (next section) — that scales linearly without losing search quality.

### When to deviate from the defaults

Most workflows: stick with the defaults. Deviate only when the prompt space genuinely justifies it.

| Case | `ga_population_size` | `ga_generations` | When this applies |
|---|---|---|---|
| **Default** | **`24`** | **`15`** | Typical workflow — most cases |
| High | `32`–`40` | `20`–`25` | Multiple prompts evolved together, or long/complex prompts |

Set `ga_parallel_evaluations` at least to `ga_population_size` so every individual in a generation evaluates concurrently; lower only when you hit 429 rate-limit errors.

---

## Tuning Parallelism

`nat optimize` has two parallelism levers. Use both to speed up a slow run — **never shrink the dataset, and never cut `n_trials` below the sampler minimum**.

**GA individuals per generation — `ga_parallel_evaluations`.** Set to at least `ga_population_size` unless you hit 429 rate-limit errors.

**Dataset items per eval — `eval.general.max_concurrency`.** Set to `8`. Decrease only on 429 rate-limit errors or if explicitly told. See the Parallelism subsection under Evaluation in the main SKILL.md.

**Rules:**

- Total concurrent LLM calls ≈ `ga_parallel_evaluations × max_concurrency`. Keep under endpoint capacity.
- If wall-clock is too long, raise both — in this order.
- Numeric wall-clock scales as `(n_trials × dataset × reps) / max_concurrency`.
- GA wall-clock scales as `(pop × gen × dataset) / (ga_parallel × max_concurrency)`.
