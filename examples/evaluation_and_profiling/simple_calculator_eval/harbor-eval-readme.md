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

# Harbor Eval Commands (Simple Calculator Nested)

Run these commands from the repository root with the repo virtual environment
active, or call `.venv/bin/harbor` explicitly.

In these `NemoAgent` examples, `--model` primarily satisfies Harbor CLI inputs; the effective workflow model comes from `--ak config_file=...` when that config defines `llms`.

## Mode guide

| Term | Meaning in these examples |
|---|---|
| Local environment | Host-local Harbor execution via `nat_harbor.environments.local:LocalEnvironment`; `--env docker` is only a temporary CLI enumeration workaround. |
| Shell mode | Default `NemoAgent` path; the agent runs through a wrapper process and verifier logic runs through the task verifier script. |
| Library mode | `--ak library_mode=true`; the NeMo Agent Toolkit workflow runs in-process through the active Harbor Python. |
| Inline verifier | `--verifier-import-path nat_harbor.verifier.inline_verifier:ATIFInlineVerifier`; ATIF evaluator dispatch runs in-process. |

Prefer library mode plus inline verifier for local development. Use shell mode for compatibility checks against script-based Harbor task behavior.

## 1) Generate Harbor dataset via adapter

```bash
# path-check-skip-next-line
ADAPTER=examples/evaluation_and_profiling/simple_calculator_eval/harbor_adapters/simple_calculator_nested/run_adapter.py
python "$ADAPTER" \
  --output-dir .tmp/harbor/datasets/simple-calculator-nested \
  --overwrite
```

## 2) Remove previous job directory

```bash
rm -rf .tmp/harbor/jobs-local/sc-nested-local .tmp/harbor/jobs-local/sc-nested-local-smoke
```

## 3) Run a single task (shell-mode smoke check)

Shell mode uses child processes, so point both the agent wrapper and verifier
script at the repo virtual environment.

```bash
HOST_PYTHON="$(pwd)/.venv/bin/python"

PATH="$(dirname "$HOST_PYTHON"):$PATH" \
harbor run \
  --path .tmp/harbor/datasets/simple-calculator-nested \
  -l 1 \
  --job-name sc-nested-local-smoke \
  --jobs-dir .tmp/harbor/jobs-local \
  --yes -n 1 --max-retries 2 \
  --agent-import-path nat_harbor.agents.installed.nemo_agent:NemoAgent \
  --environment-import-path nat_harbor.environments.local:LocalEnvironment \
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-harbor-eval.yaml \
  --ak local_install_policy=skip \
  --ak python_bin="$HOST_PYTHON" \
  --ve NAT_HARBOR_PYTHON_BIN="$HOST_PYTHON"
```

## 4) Run the full shell-mode local job

```bash
HOST_PYTHON="$(pwd)/.venv/bin/python"

PATH="$(dirname "$HOST_PYTHON"):$PATH" \
harbor run \
  --path .tmp/harbor/datasets/simple-calculator-nested \
  --job-name sc-nested-local \
  --jobs-dir .tmp/harbor/jobs-local \
  --yes -n 1 --max-retries 2 \
  --agent-import-path nat_harbor.agents.installed.nemo_agent:NemoAgent \
  --environment-import-path nat_harbor.environments.local:LocalEnvironment \
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-harbor-eval.yaml \
  --ak local_install_policy=skip \
  --ak python_bin="$HOST_PYTHON" \
  --ve NAT_HARBOR_PYTHON_BIN="$HOST_PYTHON"
```

## 5) Library mode with ATIF inline verifier lanes

Use Harbor's verifier import hook for inline ATIF scoring, and enable
`library_mode=true` so NeMo Agent Toolkit workflow execution is in-process too:

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

### Builtin trajectory lane

```bash
harbor run \
  --path .tmp/harbor/datasets/simple-calculator-nested \
  -l 1 \
  --job-name sc-nested-atif-trajectory \
  --jobs-dir .tmp/harbor/jobs-local \
  --yes -n 1 --max-retries 1 \
  --agent-import-path nat_harbor.agents.installed.nemo_agent:NemoAgent \
  --environment-import-path nat_harbor.environments.local:LocalEnvironment \
  --verifier-import-path nat_harbor.verifier.inline_verifier:ATIFInlineVerifier \
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-harbor-eval.yaml \
  --ak local_install_policy=skip \
  --ak library_mode=true \
  --ve NAT_HARBOR_ATIF_EVALUATOR_KIND=trajectory \
  --ve NAT_HARBOR_ATIF_CONFIG_FILE=examples/evaluation_and_profiling/simple_calculator_eval/src/nat_simple_calculator_eval/configs/config-nested-trajectory-eval.yml \
  --ve NAT_HARBOR_ATIF_EVALUATOR_NAME=trajectory_eval
```

### Builtin tunable-rag lane

```bash
harbor run \
  --path .tmp/harbor/datasets/simple-calculator-nested \
  -l 1 \
  --job-name sc-nested-atif-tunable-rag \
  --jobs-dir .tmp/harbor/jobs-local \
  --yes -n 1 --max-retries 1 \
  --agent-import-path nat_harbor.agents.installed.nemo_agent:NemoAgent \
  --environment-import-path nat_harbor.environments.local:LocalEnvironment \
  --verifier-import-path nat_harbor.verifier.inline_verifier:ATIFInlineVerifier \
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-harbor-eval.yaml \
  --ak local_install_policy=skip \
  --ak library_mode=true \
  --ve NAT_HARBOR_ATIF_EVALUATOR_KIND=tunable_rag \
  --ve NAT_HARBOR_ATIF_CONFIG_FILE=examples/evaluation_and_profiling/simple_calculator_eval/src/nat_simple_calculator_eval/configs/config-tunable-rag-eval-atif.yml \
  --ve NAT_HARBOR_ATIF_EVALUATOR_NAME=tuneable_eval
```

The tunable-rag lane requires valid judge-model credentials from the selected evaluator config.

### Custom evaluator ref lane

```bash
harbor run \
  --path .tmp/harbor/datasets/simple-calculator-nested \
  -l 1 \
  --job-name sc-nested-atif-custom \
  --jobs-dir .tmp/harbor/jobs-local \
  --yes -n 1 --max-retries 1 \
  --agent-import-path nat_harbor.agents.installed.nemo_agent:NemoAgent \
  --environment-import-path nat_harbor.environments.local:LocalEnvironment \
  --verifier-import-path nat_harbor.verifier.inline_verifier:ATIFInlineVerifier \
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-harbor-eval.yaml \
  --ak local_install_policy=skip \
  --ak library_mode=true \
  --ve NAT_HARBOR_ATIF_EVALUATOR_KIND=custom \
  --ve NAT_HARBOR_ATIF_EVALUATOR_REF=nat_simple_calculator_eval.scripts.harbor_sample_evaluators:artifact_presence_evaluator
```
