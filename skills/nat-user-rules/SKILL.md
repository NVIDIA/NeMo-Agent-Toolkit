---
name: nat-user-rules
description: Use first for general NVIDIA NeMo Agent toolkit coding-agent behavior, task routing, naming conventions, component discovery rules, and cross-skill guidance.
metadata:
  version: "0.1.0"
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

# NeMo Agent toolkit User Rules

Use this skill first when working in the NeMo Agent toolkit repository. It routes tasks to focused skills and states rules that apply across all toolkit work.

## Mandatory Rules

- Discover registered component `_type` values with `nat info components` before writing workflow, evaluation, optimizer, logging, or tracing YAML.
- Do not invent `_type` names or configuration keys from memory.
- Use `nat` only for technical identifiers such as the CLI, package name, Python namespace, paths, and environment variables.
- In prose, use "NVIDIA NeMo Agent toolkit" on first use, then "NeMo Agent toolkit" or "the toolkit".
- Prefer existing examples and docs before creating new patterns.
- Keep generated examples runnable from the repository root unless the surrounding example uses another convention.

## Task Routing

| Task | Skill |
| --- | --- |
| Installing or configuring the toolkit | `skills/nat-installation/SKILL.md` |
| Creating, editing, validating, or running workflow YAML | `skills/nat-workflow-creation/SKILL.md` |
| Choosing or composing agents | `skills/nat-agent-configuration/SKILL.md` |
| Writing custom tools, functions, or function groups | `skills/nat-tools-and-functions/SKILL.md` |
| Designing or running evaluation | `skills/nat-evaluation/SKILL.md` |
| Running optimizer workflows | `skills/nat-optimization/SKILL.md` |
| Adding tracing, logging, profiling, or telemetry exporters | `skills/nat-telemetry/SKILL.md` |
| Serving workflows or wiring MCP | `skills/nat-mcp-and-serving/SKILL.md` |
| Creating or improving skills | `skills/skill-evolution/SKILL.md` |

## Discovery Commands

```bash
uv run nat info components -t function
uv run nat info components -t llm_provider
uv run nat info components -t evaluator
uv run nat info components -t logging
uv run nat info components -t tracing
```

## Skill Evolution

If a user corrects the skill routing, a command fails and the recovery is reusable, or a reference is stale, finish the user task first. Then read `skills/skill-evolution/SKILL.md` and update the relevant focused skill if the lesson should generalize.
