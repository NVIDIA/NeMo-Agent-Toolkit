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

# ATIF Eval Temporary Testing Guide

This temporary guide is for quickly testing ATIF evaluation flows in the `simple_web_query_eval` example.
ATIF evaluation uses canonical trajectory samples (`workflow_output_atif.json`) so evaluators can score model outputs using
both final responses and structured agent-step context in a consistent format.

## Scope

- **ATIF built-in evaluators** (RAGAS + trajectory lane)
- **ATIF custom evaluator** (`atif_cosine_similarity`)

## Prerequisites

From the repo root:

```bash
uv pip install -e examples/evaluation_and_profiling/simple_web_query_eval
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

`simple_web_query` is pulled in as a dependency of `simple_web_query_eval`.

## 1) Test ATIF built-in evaluators

Run:

```bash
nat eval --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config_atif.yml
```

> [!NOTE]
> Other ATIF config files are also available for different models (for example `eval_config_llama31_atif.yml` and `eval_config_llama33_atif.yml`).

Expected output directory:

`./.tmp/nat/examples/evaluation_and_profiling/simple_web_query_eval/atif/`

Expected key files:

- `workflow_output.json`
- `workflow_output_atif.json`
- `accuracy_output.json`
- `groundedness_output.json`
- `relevance_output.json`
- `trajectory_accuracy_output.json`

## 2) Test ATIF custom evaluator only

Run:

```bash
nat eval --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config_atif_custom_evaluator.yml
```

Expected output directory:

`./.tmp/nat/examples/evaluation_and_profiling/simple_web_query_eval/atif_custom_evaluator/`

Expected key files:

- `workflow_output.json`
- `workflow_output_atif.json`
- `atif_cosine_similarity_eval_output.json`

Notes:

- The custom evaluator is ATIF-only and registered from `nat_simple_web_query_eval`.
- It scores using token cosine similarity and includes trajectory metadata (`trajectory_tool_call_count`) in reasoning.

## 3) Optional quick compare

Compare two run directories:

```bash
python packages/nvidia_nat_eval/scripts/compare_eval_runs.py \
  --run_a ./.tmp/nat/examples/evaluation_and_profiling/simple_web_query_eval/atif \
  --run_b ./.tmp/nat/examples/evaluation_and_profiling/simple_web_query_eval/atif_custom_evaluator
```

This is mostly useful to verify file presence/differences, since evaluator sets differ between these two configuration files.
