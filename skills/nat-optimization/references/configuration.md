# Optimizer Configuration

All optimizer settings live under a single `optimizer:` block in `workflow.yaml`:

```yaml
optimizer:
  output_path: optimizer_results   # where results are written
  target: 0.95                     # optional early stopping — stop when this score is reached
  reps_per_param_set: 3            # repetitions per trial (default 3; use 1 for deterministic agents)
  multi_objective_combination_mode: harmonic   # harmonic, sum, or chebyshev

  numeric:
    enabled: true
    n_trials: 20                   # number of Optuna trials
    sampler: bayesian              # bayesian (default), grid, or omit for auto

  prompt:
    enabled: false
    ga_population_size: 24
    ga_generations: 15
    ga_mutation_rate: 0.3          # probability of LLM-based mutation per child
    ga_crossover_rate: 0.8
    ga_elitism: 2
    ga_selection_method: tournament   # tournament or roulette
    ga_tournament_size: 3
    ga_parallel_evaluations: 8
    ga_diversity_lambda: 0.0          # >0 adds diversity pressure to avoid convergence
    prompt_population_init_function: null   # custom function for initial population seeding
    prompt_recombination_function: null     # custom function for combining parent prompts
    oracle_feedback_mode: never       # never, always, failing_only, adaptive
    oracle_feedback_worst_n: 5

  eval_metrics:
    accuracy:
      evaluator_name: correctness   # must match a key in eval.evaluators — see Evaluation section in SKILL.md
      direction: maximize
      weight: 1.0
```

> **Prerequisites:** `eval_metrics` references evaluators you've already configured in `eval.evaluators`. See the Evaluation section in `SKILL.md` for how to set up evaluators and run `nat eval`.

**Key parameters:**

> **Important**: Proper optimization and evaluation depend on choosing the right parameters. Before writing optimizer config values, read [`choosing-parameters.md`](choosing-parameters.md) — it lists every parameter, its default, and how to pick the right value for your search space and agent type. Defaults aren't final values; match them to the task.
>
> **Never kill `nat optimize` mid-run.** The optimizer only produces its final artifacts when the study finishes cleanly. If a run is slow, raise parallelism; if it's truly oversized, lower `n_trials` in the config and restart.

## Multi-Objective Optimization

Use multi-objective optimization whenever the evaluation has multiple **independent quality dimensions** — not just for cost/accuracy tradeoffs. Two dimensions are independent when the agent's output can be correct on one and wrong on the other. Configure one evaluator per dimension and one `eval_metrics` entry per evaluator; the optimizer combines them into a single objective via the metric weights and `multi_objective_combination_mode` below. Don't bake the combination inside a single composite evaluator — that hides format breakage and per-facet regressions (per-dimension evaluators all register zero on a parse failure, while a composite giving partial credit can mask it). See [`../../nat-evaluation/references/methodology.md` § Decomposing evaluators by quality dimension](../../nat-evaluation/references/methodology.md#decomposing-evaluators-by-quality-dimension).

To optimize against multiple criteria simultaneously, add multiple entries to `eval_metrics`:

```yaml
  eval_metrics:
    accuracy:
      evaluator_name: correctness     # key from eval.evaluators
      direction: maximize
      weight: 1.0
    conciseness:
      evaluator_name: length_penalty  # key from eval.evaluators
      direction: minimize
      weight: 0.5
```

Each metric has:

- `evaluator_name` — must match a key defined in `eval.evaluators` (see Evaluation section in `SKILL.md`)
- `direction` — `maximize` or `minimize`
- `weight` — relative weight when combining scores

The `multi_objective_combination_mode` controls how weights are combined:

- **`harmonic`** (default) — penalizes imbalance; good when all objectives should be met together
- **`sum`** — simple weighted sum; allows one metric to compensate for another
- **`chebyshev`** — minimizes the worst-performing objective; use when you cannot afford to sacrifice any single metric
