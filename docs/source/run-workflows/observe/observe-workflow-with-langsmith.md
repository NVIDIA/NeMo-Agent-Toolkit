<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Observing a Workflow with LangSmith

This guide provides a step-by-step process to enable observability in a NeMo Agent Toolkit workflow using LangSmith for tracing. By the end of this guide, you will have:

- Configured telemetry to send OTel traces to LangSmith.
- Ability to view workflow traces in the LangSmith UI.
- Understanding of how evaluation and optimization results are tracked as structured experiments.

### Prerequisites

An account on LangSmith is required. You can create an account at
[LangSmith](https://smith.langchain.com/).

Set your API key as an environment variable:

```bash
export LANGSMITH_API_KEY=<your-langsmith-api-key>
```

## Step 1: Install the LangChain Subpackage

Install the LangChain dependencies (which include LangSmith) to enable tracing capabilities:

```bash
uv pip install -e '.[langchain]'
```

## Step 2: Modify Workflow Configuration

Update your workflow configuration file to include the telemetry settings.

Example configuration:

```yaml
general:
  telemetry:
    tracing:
      langsmith:
        _type: langsmith
        project: default
```

This setup enables tracing through LangSmith, with traces grouped into the `default` project.

## Step 3: Run Your Workflow

From the root directory of the NeMo Agent Toolkit library, install dependencies and run the pre-configured `simple_calculator_observability` example.

**Example:**

```bash
# Install the workflow and plugins
uv pip install -e examples/observability/simple_calculator_observability/

# Run the workflow with LangSmith telemetry settings
nat run --config_file examples/observability/simple_calculator_observability/configs/config-langsmith.yml --input "What is 2 * 4?"
```

As the workflow runs, telemetry data will start showing up in LangSmith.

To override the LangSmith project name from the command line without editing the config file, use the `--override` flag:

```bash
nat run --config_file examples/observability/simple_calculator_observability/configs/config-langsmith.yml \
  --override general.telemetry.tracing.langsmith.project <your_project_name> \
  --input "What is 2 * 4?"
```

The `--override` flag accepts a dot-notation path into the YAML config hierarchy followed by the new value. It can be specified multiple times to override multiple fields.

## Step 4: View Traces in LangSmith

- Open your browser and navigate to [LangSmith](https://smith.langchain.com/).
- Locate your workflow traces under your project name in the Projects section.
- Inspect function execution details, latency, token counts, and other information for individual traces.

## Structured Evaluation Experiments

:::{note}
The `nat eval` command is provided by the evaluation package. If the command is not available, install the eval extra first:

```bash
uv pip install -e '.[eval]'
```

Or, for a package install:

```bash
uv pip install "nvidia-nat[eval]"
```

For more details, see [Agent Evaluation Prerequisites](../../improve-workflows/evaluate.md#prerequisites).
:::

LangSmith implements the [evaluation callback](../../improve-workflows/evaluate.md#evaluation-callbacks) pattern to create structured experiments in the LangSmith Datasets & Experiments UI. When you run `nat eval` with LangSmith tracing enabled, the following happens automatically:

- A **Dataset** is created from your eval questions (named "Benchmark Dataset (\<dataset-name\>)"). Each dataset entry becomes a LangSmith example with inputs and expected outputs.
- An **Experiment** project (named "\<project\> (Run #N)") is linked to the dataset. Each evaluation run increments the run number.
- Per-example **runs** are linked to their corresponding dataset examples with evaluator scores attached as **feedback** on each run.
- **OTel span traces** capture each LLM call within each workflow run.

### Running an Evaluation with LangSmith

Use the pre-configured evaluation example:

```bash
nat eval --config_file examples/observability/simple_calculator_observability/configs/config-langsmith-eval.yml
```

<!-- path-check-skip-begin -->
This configuration includes both the LangSmith telemetry settings and an evaluation section:

```yaml
general:
  telemetry:
    tracing:
      langsmith:
        _type: langsmith
        project: nat-eval-demo

eval:
  general:
    max_concurrency: 1
    output_dir: .tmp/nat/examples/langsmith_eval
    dataset:
      _type: json
      file_path: examples/getting_started/simple_calculator/src/nat_simple_calculator/data/simple_calculator.json
  evaluators:
    accuracy:
      _type: tunable_rag_evaluator
      llm_name: eval_llm
      default_scoring: true
```
<!-- path-check-skip-end -->

After running, check your LangSmith project for:

- A dataset created from the eval questions.
- Per-example runs with model answers linked to dataset examples.
- Evaluator scores as feedback on each run.
- OTel span traces for each LLM call.

## Structured Optimization Experiments

LangSmith implements the [optimization callback](../../improve-workflows/optimizer.md#optimization-callbacks) pattern to track each optimization trial as a separate experiment. When you run `nat optimize` with LangSmith tracing enabled, the following happens automatically:

- A **shared Dataset** is created for the entire optimization run.
- Each trial gets its own **Experiment** project (named "\<base\> (Run #N, Trial M)"), all linked to the shared dataset. This enables per-trial comparison in the Datasets & Experiments UI.
- Parameter configurations are recorded as project **metadata** on each trial.
- Evaluator scores are attached as **feedback** per trial.
- For prompt optimization, prompt versions are pushed to **LangSmith prompt repositories** with commit tags for each trial (e.g., `trial-1`, `trial-2`). The best trial's prompt is tagged with `best`.

### Running an Optimization with LangSmith

Use the pre-configured optimization example:

```bash
nat optimize --config_file examples/observability/simple_calculator_observability/configs/config-langsmith-optimize.yml
```

<!-- path-check-skip-begin -->
This configuration includes LangSmith telemetry, an evaluation section, and an optimizer section:

```yaml
general:
  telemetry:
    tracing:
      langsmith:
        _type: langsmith
        project: nat-optimize-demo

eval:
  general:
    max_concurrency: 1
    output_dir: .tmp/nat/examples/langsmith_optimize
    dataset:
      _type: json
      file_path: examples/getting_started/simple_calculator/src/nat_simple_calculator/data/simple_calculator.json
  evaluators:
    accuracy:
      _type: tunable_rag_evaluator
      llm_name: eval_llm
      default_scoring: true

optimizer:
  output_path: .tmp/nat/examples/langsmith_optimize/optimizer
  reps_per_param_set: 1
  eval_metrics:
    accuracy:
      evaluator_name: accuracy
      direction: maximize
  numeric:
    enabled: true
    n_trials: 3
  prompt:
    enabled: false
```
<!-- path-check-skip-end -->

After running, check your LangSmith project for:

- Trial runs with parameter configurations recorded as metadata.
- Feedback scores per trial for each configured metric.
- OTel span traces for each LLM call within each trial.

### Resources

For more information about LangSmith, view the documentation [here](https://docs.smith.langchain.com/).
