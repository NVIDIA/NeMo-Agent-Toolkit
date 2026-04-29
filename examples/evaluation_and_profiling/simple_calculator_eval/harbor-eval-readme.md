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

# Harbor Eval Commands (Simple Calculator)

Run these commands from the repository root with the repo virtual environment
active, or call `.venv/bin/harbor` explicitly.

In these `NemoAgent` examples, `--model` primarily satisfies Harbor CLI inputs; the effective workflow model comes from `--ak config_file=...` when that config defines `llms`.

Use this README as the simple calculator example cookbook. It
covers library mode with inline NeMo Agent Toolkit evaluators first, then shell
compatibility mode for parity checks against generated Harbor task verifiers.

## Mode guide

| Term | Meaning in these examples |
|---|---|
| Local environment | Host-local Harbor execution via `nat_harbor.environments.local:LocalEnvironment`; `--env docker` is only a temporary CLI enumeration workaround. |
| Shell mode | Default `NemoAgent` path; the agent runs through a wrapper process and the generated Harbor task verifier performs a simple ground-truth check. NeMo Agent Toolkit evaluators are not used. |
| Library mode | `--ak library_mode=true`; the NeMo Agent Toolkit workflow runs in-process through the active Harbor Python. |
| Inline verifier | `--verifier-import-path nat_harbor.verifier.inline_verifier:ATIFInlineVerifier`; ATIF evaluator dispatch runs in-process. |

Prefer library mode plus inline verifier for local development. Use shell mode for compatibility checks against script-based Harbor task behavior.

## Phoenix tracing

The configuration files used by the inline evaluator examples enable Phoenix tracing at
`http://localhost:6006/v1/traces`. Start Phoenix in a separate terminal before
running those examples:

```bash
uv venv --python 3.13 --seed .venv-phoenix
source .venv-phoenix/bin/activate
uv pip install arize-phoenix
phoenix serve
```

## 1) Generate Harbor datasets via adapters

```bash
# path-check-skip-next-line
NAT_HARBOR_POWER_OF_TWO_ADAPTER=examples/evaluation_and_profiling/simple_calculator_eval/harbor_adapters/simple_calculator_power_of_two/run_adapter.py
NAT_HARBOR_POWER_OF_TWO_DATASET_DIR=.tmp/harbor/datasets/simple-calculator-power-of-two

python "$NAT_HARBOR_POWER_OF_TWO_ADAPTER" \
  --output-dir "$NAT_HARBOR_POWER_OF_TWO_DATASET_DIR" \
  --overwrite

# path-check-skip-next-line
NAT_HARBOR_TUNABLE_RAG_ADAPTER=examples/evaluation_and_profiling/simple_calculator_eval/harbor_adapters/simple_calculator/run_adapter.py
NAT_HARBOR_TUNABLE_RAG_SOURCE=examples/getting_started/simple_calculator/data/simple_calculator.json
NAT_HARBOR_TUNABLE_RAG_DATASET_DIR=.tmp/harbor/datasets/simple-calculator-tunable-rag

python "$NAT_HARBOR_TUNABLE_RAG_ADAPTER" \
  --source-file "$NAT_HARBOR_TUNABLE_RAG_SOURCE" \
  --output-dir "$NAT_HARBOR_TUNABLE_RAG_DATASET_DIR" \
  --overwrite
```

## 2) Remove previous job directories

This removes only the job directories created by the examples in this file.

```bash
NAT_HARBOR_JOBS_DIR=.tmp/harbor/jobs-local

rm -rf \
  "$NAT_HARBOR_JOBS_DIR/sc-power-of-two-atif-trajectory" \
  "$NAT_HARBOR_JOBS_DIR/sc-tunable-rag-atif" \
  "$NAT_HARBOR_JOBS_DIR/sc-power-of-two-atif-custom" \
  "$NAT_HARBOR_JOBS_DIR/sc-power-of-two-local" \
  "$NAT_HARBOR_JOBS_DIR/sc-power-of-two-local-smoke"
```

## Command parameter guide

| Parameter | Purpose |
|---|---|
| `--agent-import-path` | Loads the `NemoAgent` implementation from `nvidia-nat-harbor`. |
| `--environment-import-path` | Loads the host-local Harbor environment implementation. |
| `--env docker` | Satisfies current Harbor CLI enumeration validation; the imported environment keeps execution host-local. |
| `--ak config_file=...` | Selects the NeMo Agent Toolkit workflow config used by the trial run. |
| `--ak local_install_policy=skip` | Skips per-run package installs when the local environment is already prepared. |
| `--ak library_mode=true` | Runs the workflow in the active Harbor Python process. |
| `--ak python_bin=...` | Points shell-mode agent execution at the repo virtual environment Python. |
| `--ve NAT_HARBOR_ATIF_*` | Passes evaluator selection and config to the inline verifier. |
| `--ve NAT_HARBOR_PYTHON_BIN=...` | Points shell-mode verifier scripts at the repo virtual environment Python. |
| `-l 1` | Runs one generated Harbor task for a quick smoke check. |

## 3) Library mode with ATIF inline verifier lanes

Use this path first for local development. Library mode runs the NeMo Agent
Toolkit workflow in-process, and the inline verifier runs ATIF evaluator
dispatch in-process too:

```bash
--verifier-import-path nat_harbor.verifier.inline_verifier:ATIFInlineVerifier
--ak library_mode=true
```

This requires Harbor verifier import-hook support. If `harbor run --help` does
not list `--verifier-import-path`, install the Harbor side branch used by this
workspace before running these inline verifier examples:

```bash
git clone https://github.com/AnuradhaKaruppiah/harbor.git external/harbor
git -C external/harbor checkout ak-harbor-libary-mode
uv pip install -e external/harbor
```

If `external/harbor` already exists, update it to the side branch instead of
cloning again.

You can use any evaluator registered as a NeMo Agent Toolkit evaluator, as long as it exposes an ATIF lane (`evaluate_atif_fn`), and you can also use simple Python evaluator callables via `NAT_HARBOR_ATIF_EVALUATOR_REF=module:function`.

### Built-in trajectory lane

This is the clearest starter command: the same config drives the trial workflow
and the trajectory evaluator. It uses power-of-two tasks because the workflow
exposes the `power_of_two` tool, which internally calls `calculator__multiply`.

```bash
NAT_HARBOR_JOBS_DIR=.tmp/harbor/jobs-local
NAT_HARBOR_POWER_OF_TWO_DATASET_DIR=.tmp/harbor/datasets/simple-calculator-power-of-two
NAT_HARBOR_TRAJECTORY_CONFIG=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-trajectory-eval.yml

harbor run \
  --path "$NAT_HARBOR_POWER_OF_TWO_DATASET_DIR" \
  -l 1 \
  --job-name sc-power-of-two-atif-trajectory \
  --jobs-dir "$NAT_HARBOR_JOBS_DIR" \
  --yes -n 1 --max-retries 1 \
  --agent-import-path nat_harbor.agents.installed.nemo_agent:NemoAgent \
  --environment-import-path nat_harbor.environments.local:LocalEnvironment \
  --verifier-import-path nat_harbor.verifier.inline_verifier:ATIFInlineVerifier \
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file="$NAT_HARBOR_TRAJECTORY_CONFIG" \
  --ak local_install_policy=skip \
  --ak library_mode=true \
  --ve NAT_HARBOR_ATIF_EVALUATOR_KIND=trajectory \
  --ve NAT_HARBOR_ATIF_CONFIG_FILE="$NAT_HARBOR_TRAJECTORY_CONFIG" \
  --ve NAT_HARBOR_ATIF_EVALUATOR_NAME=trajectory_eval
```

### Built-in tunable-rag lane

This command uses the same config for the trial workflow and the tunable-rag
evaluator. It also uses a Harbor dataset generated from the same source file
referenced by the tunable-rag eval config.

```bash
NAT_HARBOR_JOBS_DIR=.tmp/harbor/jobs-local
NAT_HARBOR_TUNABLE_RAG_DATASET_DIR=.tmp/harbor/datasets/simple-calculator-tunable-rag
NAT_HARBOR_TUNABLE_RAG_ATIF_CONFIG=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-tunable-rag-eval-atif.yml

harbor run \
  --path "$NAT_HARBOR_TUNABLE_RAG_DATASET_DIR" \
  -l 1 \
  --job-name sc-tunable-rag-atif \
  --jobs-dir "$NAT_HARBOR_JOBS_DIR" \
  --yes -n 1 --max-retries 1 \
  --agent-import-path nat_harbor.agents.installed.nemo_agent:NemoAgent \
  --environment-import-path nat_harbor.environments.local:LocalEnvironment \
  --verifier-import-path nat_harbor.verifier.inline_verifier:ATIFInlineVerifier \
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file="$NAT_HARBOR_TUNABLE_RAG_ATIF_CONFIG" \
  --ak local_install_policy=skip \
  --ak library_mode=true \
  --ve NAT_HARBOR_ATIF_EVALUATOR_KIND=tunable_rag \
  --ve NAT_HARBOR_ATIF_CONFIG_FILE="$NAT_HARBOR_TUNABLE_RAG_ATIF_CONFIG" \
  --ve NAT_HARBOR_ATIF_EVALUATOR_NAME=tuneable_eval
```

The tunable-rag lane requires valid judge-model credentials from the selected evaluator config.

### Custom evaluator ref lane

```bash
NAT_HARBOR_JOBS_DIR=.tmp/harbor/jobs-local
NAT_HARBOR_POWER_OF_TWO_DATASET_DIR=.tmp/harbor/datasets/simple-calculator-power-of-two
NAT_HARBOR_TRAJECTORY_CONFIG=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-trajectory-eval.yml

harbor run \
  --path "$NAT_HARBOR_POWER_OF_TWO_DATASET_DIR" \
  -l 1 \
  --job-name sc-power-of-two-atif-custom \
  --jobs-dir "$NAT_HARBOR_JOBS_DIR" \
  --yes -n 1 --max-retries 1 \
  --agent-import-path nat_harbor.agents.installed.nemo_agent:NemoAgent \
  --environment-import-path nat_harbor.environments.local:LocalEnvironment \
  --verifier-import-path nat_harbor.verifier.inline_verifier:ATIFInlineVerifier \
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file="$NAT_HARBOR_TRAJECTORY_CONFIG" \
  --ak local_install_policy=skip \
  --ak library_mode=true \
  --ve NAT_HARBOR_ATIF_EVALUATOR_KIND=custom \
  --ve NAT_HARBOR_ATIF_EVALUATOR_REF=nat_simple_calculator_eval.scripts.harbor_sample_evaluators:artifact_presence_evaluator
```

## 4) Shell mode

Shell mode uses child processes, so point both the agent wrapper and verifier
script at the repo virtual environment.

Shell-mode evaluation is intentionally simple: it runs the generated Harbor
task verifier against the expected answer. It does not use NeMo Agent Toolkit
evaluators.

### Single-task smoke check

```bash
NAT_HARBOR_JOBS_DIR=.tmp/harbor/jobs-local
NAT_HARBOR_POWER_OF_TWO_DATASET_DIR=.tmp/harbor/datasets/simple-calculator-power-of-two
NAT_HARBOR_TRAJECTORY_CONFIG=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-trajectory-eval.yml
NAT_HARBOR_HOST_PYTHON="$(pwd)/.venv/bin/python"

PATH="$(dirname "$NAT_HARBOR_HOST_PYTHON"):$PATH" \
harbor run \
  --path "$NAT_HARBOR_POWER_OF_TWO_DATASET_DIR" \
  -l 1 \
  --job-name sc-power-of-two-local-smoke \
  --jobs-dir "$NAT_HARBOR_JOBS_DIR" \
  --yes -n 1 --max-retries 2 \
  --agent-import-path nat_harbor.agents.installed.nemo_agent:NemoAgent \
  --environment-import-path nat_harbor.environments.local:LocalEnvironment \
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file="$NAT_HARBOR_TRAJECTORY_CONFIG" \
  --ak local_install_policy=skip \
  --ak python_bin="$NAT_HARBOR_HOST_PYTHON" \
  --ve NAT_HARBOR_PYTHON_BIN="$NAT_HARBOR_HOST_PYTHON"
```

### Full local job

```bash
NAT_HARBOR_JOBS_DIR=.tmp/harbor/jobs-local
NAT_HARBOR_POWER_OF_TWO_DATASET_DIR=.tmp/harbor/datasets/simple-calculator-power-of-two
NAT_HARBOR_TRAJECTORY_CONFIG=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-trajectory-eval.yml
NAT_HARBOR_HOST_PYTHON="$(pwd)/.venv/bin/python"

PATH="$(dirname "$NAT_HARBOR_HOST_PYTHON"):$PATH" \
harbor run \
  --path "$NAT_HARBOR_POWER_OF_TWO_DATASET_DIR" \
  --job-name sc-power-of-two-local \
  --jobs-dir "$NAT_HARBOR_JOBS_DIR" \
  --yes -n 1 --max-retries 2 \
  --agent-import-path nat_harbor.agents.installed.nemo_agent:NemoAgent \
  --environment-import-path nat_harbor.environments.local:LocalEnvironment \
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file="$NAT_HARBOR_TRAJECTORY_CONFIG" \
  --ak local_install_policy=skip \
  --ak python_bin="$NAT_HARBOR_HOST_PYTHON" \
  --ve NAT_HARBOR_PYTHON_BIN="$NAT_HARBOR_HOST_PYTHON"
```

## Related docs

- [Package README](../../../packages/nvidia_nat_harbor/README.md)
- [Simple calculator Harbor adapter guide](harbor_adapters/README.md)
