---
name: nat-optimization
description: Use when configuring or running NeMo Agent toolkit optimization with `nat optimize`, including Optuna parameter tuning, prompt evolution, optimizer sizing, output interpretation, and optimizer datasets.
metadata:
  version: "0.1.1"
  status: initial
---

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

# NeMo Agent toolkit Optimization

Use this skill when improving workflow quality through `nat optimize`.

## Workflow

1. Fix workflow correctness issues before optimizing.
2. Size the run and explain the chosen `n_trials`, parallelism, and stopping behavior.
3. Use separate evaluators for separate quality dimensions.
4. Run `nat optimize` with a generous timeout.
5. Inspect output artifacts before writing tuned values back to workflow YAML.

## Guardrail

Do not kill `nat optimize` mid-run unless the user asks. It writes final artifacts when the study finishes cleanly.

## References

- `references/overview.md`
- `references/choosing-parameters.md`
- `references/configuration.md`
- `references/output-and-cli.md`
- `references/complete-config-example.md`
- `references/optimizer_example_dataset.json`
