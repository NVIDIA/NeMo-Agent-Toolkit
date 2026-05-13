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

# AGENTS.md - NVIDIA NeMo Agent toolkit AI Agent Entry Point

AI agent skills for the NVIDIA NeMo Agent toolkit live in **`skills/`** at the repository root and use a **flat layout**. Read this file first, then choose skills from the task routing index below.

## Mandatory Repository Rules

- Preserve user changes. Check `git status` before editing and do not revert unrelated work.
- Prefer existing project patterns, examples, and documentation over new abstractions.
- Do not add dependencies, update lock files, or run package installation commands unless the task requires it.
- Use the full product name "NVIDIA NeMo Agent toolkit" on first use in public documentation, then "NeMo Agent toolkit" or "the toolkit". Use `nat` only for the CLI, Python namespace, package metadata, and other technical identifiers.
- Keep examples runnable from the repository root unless the surrounding file uses a different convention.
- For ambiguous tasks, clarify the intended outcome or cover each plausible interpretation and report what you did.

## Skills Directory (Flat)

- `skills/nat-user-rules/` - General behavior, naming conventions, discovery rules, and task routing.
- `skills/nat-installation/` - Installation, optional extras, CLI verification, and first workflow setup.
- `skills/nat-workflow-creation/` - Workflow YAML authoring, component discovery, LLM configuration, and common CLI commands.
- `skills/nat-agent-configuration/` - Agent selection, built-in agent configuration, control flow, and sub-agent composition.
- `skills/nat-tools-and-functions/` - Custom functions, tools, function groups, and Python extension patterns.
- `skills/nat-evaluation/` - Evaluation methodology, datasets, evaluator selection, ATIF surfaces, and `nat eval`.
- `skills/nat-optimization/` - `nat optimize`, optimizer configuration, parameter selection, and output interpretation.
- `skills/nat-telemetry/` - Logging, tracing, profiling, OpenTelemetry, and telemetry exporters.
- `skills/nat-mcp-and-serving/` - MCP clients and servers, FastAPI, and workflow serving.
- `skills/nat-path-checks/` - Documentation path-check failures, Markdown escaping, and slash-delimited token fixes.
- `skills/skill-evolution/` - Creating, refining, and maintaining AI coding agent skills.

## Task Routing

Use the routing table before opening detailed references inside the skill.

| Task | Skill and Reference |
| --- | --- |
| Installing or configuring NeMo Agent toolkit | `skills/nat-installation/SKILL.md` |
| Discovering registered component `_type` values | `skills/nat-workflow-creation/SKILL.md` |
| Writing or editing workflow YAML | `skills/nat-workflow-creation/SKILL.md` |
| Configuring LLMs | `skills/nat-workflow-creation/SKILL.md` |
| Choosing or configuring agents | `skills/nat-agent-configuration/SKILL.md` |
| Writing custom tools or functions | `skills/nat-tools-and-functions/SKILL.md` |
| Wiring MCP client or server workflows | `skills/nat-mcp-and-serving/SKILL.md` |
| Serving with FastAPI | `skills/nat-mcp-and-serving/SKILL.md` |
| Composing sub-agents | `skills/nat-agent-configuration/SKILL.md` |
| Adding tracing or telemetry | `skills/nat-telemetry/SKILL.md` |
| Designing an evaluation suite | `skills/nat-evaluation/SKILL.md` |
| Choosing an evaluator | `skills/nat-evaluation/SKILL.md` |
| Running `nat optimize` | `skills/nat-optimization/SKILL.md` |
| Fixing documentation path-check failures | `skills/nat-path-checks/SKILL.md` |
| Creating or improving skills | `skills/skill-evolution/SKILL.md` |

## Skill Evolution

When a user corrects your approach, a command fails and you recover, or you discover a generalizable gotcha, finish the task first and then read `skills/skill-evolution/SKILL.md` to decide whether the skills should be updated.

## Quick Commands

Run commands from the repository root unless a skill or package README says otherwise.

```bash
uv run nat --help
uv run pytest packages/nvidia_nat_core
NAT_DISABLE_API_BUILD=1 make -C docs html
```

For local documentation details, read `docs/README.md`. For contribution details, read `docs/source/resources/contributing/index.md`.
