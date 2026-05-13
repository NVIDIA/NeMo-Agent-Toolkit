# Complete Optimizer Config Example

> ⚠ **Before setting parameters in this config, read [`choosing-parameters.md`](choosing-parameters.md).** That file defines the rules for some parameters, samplers, and parallelism. Copying the YAML below without reading the rules leads to under-budgeted searches that look fine but produce noise instead of signal.

`nat optimize` reads a single config file that must contain **all four top-level sections**: `llms`, `workflow`, `eval`, and `optimizer`.

The example below is a minimal but complete `workflow.yaml` you can adapt. It uses `tunable_rag_evaluator` (bundled with `nvidia-nat[langchain]`, no extra install). Replace the model names and workflow type with your actual agent.

## Complete Example

```yaml
# workflow.yaml — complete file, all sections required

llms:
  main_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1
    api_key: $NVIDIA_API_KEY
    temperature: 0.0
    max_tokens: 1024
    optimizable_params: [temperature]   # marks temperature for Optuna search

  judge_llm:
    _type: nim
    model_name: nvidia/llama-3.3-nemotron-super-49b-v1
    base_url: https://integrate.api.nvidia.com/v1
    api_key: $NVIDIA_API_KEY
    temperature: 0.0
    max_tokens: 1024
    # Note: do not optimize the judge LLM — it must stay stable across all trials

workflow:
  _type: react_agent          # or predict, tool_calling_agent, etc.
  llm_name: main_llm
  tool_names: []              # list your tool names here
  verbose: false

eval:
  general:
    dataset: eval_dataset.json   # path to your question/answer pairs
    output_path: output
    max_concurrency: 8           # lower if you encounter rate limits

  evaluators:
    correctness:               # key used in optimizer.eval_metrics
      _type: tunable_rag_evaluator
      llm_name: judge_llm
      judge_llm_prompt: ""     # required field even when default_scoring: true
      default_scoring: true

optimizer:
  output_path: optimizer_results

  numeric:
    enabled: true
    n_trials: 20

  prompt:
    enabled: false

  reps_per_param_set: 3

  eval_metrics:
    accuracy:
      evaluator_name: correctness   # must match a key in eval.evaluators above
      direction: maximize
      weight: 1.0
```

## Key Points

**Evaluator key must match `evaluator_name`.** The value of `optimizer.eval_metrics.<metric>.evaluator_name` must exactly match a key under `eval.evaluators`. In the example both are `correctness`.

**Use a different LLM for judging.** Never use the same model for both the workflow and the judge — self-evaluation inflates scores. The judge LLM should not be in `optimizable_params`.

## Using a Different Evaluator

To swap the evaluator, replace the `eval.evaluators` block. The `optimizer.eval_metrics.evaluator_name` key must still match.

### With `trajectory` evaluator

```yaml
eval:
  evaluators:
    quality:
      _type: trajectory
      llm_name: judge_llm

optimizer:
  eval_metrics:
    accuracy:
      evaluator_name: quality
      direction: maximize
      weight: 1.0
```

### With `langsmith_judge` evaluator

```yaml
eval:
  evaluators:
    quality:
      _type: langsmith_judge
      llm_name: judge_llm
      prompt: correctness      # prebuilt openevals prompt

optimizer:
  eval_metrics:
    accuracy:
      evaluator_name: quality
      direction: maximize
      weight: 1.0
```

For full field reference and config examples for each evaluator type, see [`../../nat-evaluation/references/evaluators/`](../../nat-evaluation/references/evaluators/).
