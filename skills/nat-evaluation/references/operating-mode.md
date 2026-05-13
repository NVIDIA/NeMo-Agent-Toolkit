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

# Operating Mode

Evaluation is designed for **autonomous execution**. Minimize user interaction:

1. **Infer context from the codebase.** Read the agent's NeMo Agent toolkit workflow config YAML, tool definitions, custom functions, and existing eval configs before asking questions. Look for `_type: react_agent`, `functions:` blocks, `llms:` blocks, and existing `eval_config.yml` files.
2. **Only ask when you truly cannot infer.** If you can determine the agent's workflow type, tools, and LLM config from YAML — proceed. Reserve questions for ambiguous business requirements.
3. **Generate complete, runnable configs and code.** Every step should produce YAML configs or Python files the user can execute with `nat eval` immediately.
4. **Start where the user needs you.** If they ask about CI/CD quality gates, jump to Step 7. If they want evaluators, start at Step 3.

## Quick Start Paths

Not everyone needs the full 8-step walkthrough. Match the user's intent:

| User wants... | Start at | Skip |
| ------------- | --------- | ----- |
| "Just run an eval on my agent" | Step 5 (infer Steps 0-4 automatically) | Steps 6-8 |
| "Set up evaluators for my agent" | Step 0 → Step 4 | Steps 5-8 until asked |
| "Should I use ATIF or legacy NeMo Agent toolkit eval?" | `evaluation-surfaces.md` | Methodology until the surface is chosen |
| "Full eval setup from scratch" | Step 0 | Nothing |
| "CI/CD quality gates" | Step 7 | Steps 0-6 unless context is missing |
| "What should I evaluate?" | Step 1 | Generate code only when asked |

## External Documentation

When you need API-level detail beyond what's in the code patterns (exact config options, latest evaluator types, new NeMo Agent toolkit features), look up the official docs:

- **NeMo Agent toolkit Evaluation docs** — Fetch from `https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/evaluate.html` via WebFetch. Covers `eval_config.yml` schema, built-in evaluator types, profiler configuration, and `nat eval` CLI options.
- **NeMo Agent toolkit Custom Evaluators** — Fetch from `https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/custom-evaluators.html` via WebFetch. Covers `BaseEvaluator`, `EvaluatorBaseConfig`, `@register_evaluator` pattern.
- **NeMo Agent toolkit GitHub** — `github.com/NVIDIA/NeMo-Agent-Toolkit`. Check `examples/` directory for working evaluation examples.
- **RAGAS** — `docs.ragas.io`. Check for available metrics beyond AnswerAccuracy, ResponseGroundedness, ContextRelevance.

Only fetch docs when the code patterns reference file doesn't have what you need — don't load docs speculatively.
