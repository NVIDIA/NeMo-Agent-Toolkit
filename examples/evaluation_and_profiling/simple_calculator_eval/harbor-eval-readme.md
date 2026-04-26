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

Run these commands from the repository root.

In these `NemoAgent` examples, `--model` primarily satisfies Harbor CLI inputs; the effective workflow model comes from `--ak config_file=...` when that config defines `llms`.

## 1) Generate Harbor dataset via adapter

```bash
python examples/evaluation_and_profiling/simple_calculator_eval/harbor_adapters/simple_calculator_nested/run_adapter.py \
  --output-dir .tmp/harbor/datasets/simple-calculator-nested \
  --overwrite
```

## 2) Remove previous job directory

```bash
rm -rf .tmp/harbor/jobs-local/sc-nested-local .tmp/harbor/jobs-local/sc-nested-local-smoke
```

## 3) Run a single task (smoke check)

```bash
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
  --ak local_install_policy=skip
```

## 4) Run the full local-mode job

```bash
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
  --ak local_install_policy=skip
```

## 5) ATIF bridge evaluator lanes

Set these verifier env flags on `harbor run` to enable ATIF bridge scoring in `tests/test.sh`.
You can use any evaluator registered as a NAT evaluator, as long as it exposes an ATIF lane (`evaluate_atif_fn`), and you can also use simple Python evaluator callables via `NAT_HARBOR_ATIF_EVALUATOR_REF=module:function`.

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
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-harbor-eval.yaml \
  --ak local_install_policy=skip \
  --ve NAT_HARBOR_ATIF_BRIDGE_ENABLED=1 \
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
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-harbor-eval.yaml \
  --ak local_install_policy=skip \
  --ve NAT_HARBOR_ATIF_BRIDGE_ENABLED=1 \
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
  --env docker \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak config_file=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-harbor-eval.yaml \
  --ak local_install_policy=skip \
  --ve NAT_HARBOR_ATIF_BRIDGE_ENABLED=1 \
  --ve NAT_HARBOR_ATIF_EVALUATOR_KIND=custom \
  --ve NAT_HARBOR_ATIF_EVALUATOR_REF=nat_simple_calculator_eval.scripts.harbor_sample_evaluators:artifact_presence_evaluator
```

