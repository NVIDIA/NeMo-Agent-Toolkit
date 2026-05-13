---
name: nat-evaluation
description: Use when designing, configuring, running, or troubleshooting NeMo Agent toolkit evaluations, datasets, evaluator selection, ATIF surfaces, quality gates, custom evaluators, and `nat eval`.
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

# NeMo Agent toolkit Evaluation

Use this skill for measuring agent quality and behavior.

## Workflow

1. Decide the evaluation surface and output format.
2. Decompose quality goals into separate evaluators.
3. Choose built-in evaluators before writing custom evaluators.
4. Keep datasets small and explicit for local validation.
5. Run `nat eval` and inspect generated artifacts.

## References

- `references/operating-mode.md`
- `references/methodology.md`
- `references/agent-eval-framework.md`
- `references/evaluation-surfaces.md`
- `references/evaluation-contract.md`
- `references/evaluators/`
- `references/code-patterns.md`
