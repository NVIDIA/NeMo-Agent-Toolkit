# Optimizer Output, CLI, and Callbacks

## Output Files

After `nat optimize` completes, results are written to `output_path/` (default: `optimizer_results/`):

| File | When | Contents |
| --- | --- | --- |
| `optimized_config.yml` | Always | Best-found configuration — ready to use as a workflow config |
| `trials_dataframe_params.csv` | Always | Full Optuna trial history — parameters, scores, and timings for every trial |
| `optimized_prompts.json` | Prompt GA | Final best prompt set |
| `optimized_prompts_gen<N>.json` | Prompt GA | Best prompt set after generation N |
| `ga_history_prompts.csv` | Prompt GA | Per-individual fitness and metrics across all generations |
| `pareto_front_2d.png` | Multi-objective (2 metrics) | 2-metric Pareto front scatter plot |
| `pareto_parallel_coordinates.png` | Multi-objective | Normalized performance across all metrics |
| `pareto_pairwise_matrix.png` | Multi-objective | Metric distribution and correlation matrix |

`trials_dataframe_params.csv` is the most useful for understanding which parameters had the most impact before accepting `optimized_config.yml` blindly.

## CLI Reference

```bash
nat optimize --config_file workflow.yaml
nat optimize --config_file workflow.yaml --dataset path/to/dataset.json
nat optimize --config_file workflow.yaml --result_json_path '$'
nat optimize --config_file workflow.yaml --endpoint http://your-llm-endpoint
```

| Flag | Default | Description |
| --- | --- | --- |
| `--config_file` | — | Path to the workflow YAML containing the `optimizer:` block |
| `--dataset` | — | Override the dataset path (can cause type errors in some NeMo Agent toolkit versions — prefer setting it in the config) |
| `--result_json_path` | `$` | JSONPath expression to extract the workflow result from the output |
| `--endpoint` | — | Override the LLM endpoint for remote workflow execution |
| `--endpoint_timeout` | `300` | Request timeout in seconds for endpoint calls |

> Set the dataset path in the config (`eval.general.dataset`) rather than via `--dataset` — the CLI flag can cause type errors with some NeMo Agent toolkit versions.
>
> `nat optimize` is a long-running command. Run it in the background with a generous timeout. See [`choosing-parameters.md`](choosing-parameters.md) for wall-clock scaling and the no-kill rule.

## Callbacks

The optimizer supports a callback protocol (`OptimizerCallback`) for integrating with experiment tracking systems (MLflow, Weights & Biases, etc.). Callbacks fire at key points:

| Method | When it fires | Use case |
| --- | --- | --- |
| `pre_create_experiment(dataset_items)` | Before trials begin | Set up shared experiment context |
| `get_trial_project_name(trial_number)` | Before each trial eval | Return per-trial project identifier |
| `on_trial_end(result: TrialResult)` | After each trial completes | Log metrics, link traces, record config |
| `on_study_end(best_trial, total_trials)` | After all trials | Tag best artifacts, generate summary |

`TrialResult` contains `trial_number`, `parameters`, `metric_scores`, `is_best`, `prompts`, `prompt_formats`, and the full `eval_result`.
