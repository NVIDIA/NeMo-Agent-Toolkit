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

# AI Coding Agent Skills

The NeMo Agent toolkit repository includes reusable AI coding agent skills under `skills/`. The root `AGENTS.md` file provides the entry point and routes agents to focused skill folders for installation, workflow authoring, agent configuration, tools and functions, evaluation, optimization, telemetry, and serving.

Use the skill when you want an AI coding agent to help with NeMo Agent toolkit development tasks such as writing workflow YAML, creating functions, configuring `nat eval`, or adding telemetry.

## Install the Skill

Copy the root `AGENTS.md` file and the `skills/` directory into the repository or skill directory used by your coding agent. Restart the agent session after copying the skills so the agent can discover each `SKILL.md`.

### Claude Code

For a user-level install:

```bash
mkdir -p ~/.claude/skills
cp -r skills/nat-* ~/.claude/skills/
```

For a project-level install:

```bash
mkdir -p .claude/skills
cp -r skills/nat-* .claude/skills/
```

### Codex

For a user-level install:

```bash
mkdir -p ~/.codex/skills
cp -r skills/nat-* ~/.codex/skills/
```

### Other Coding Agents

Copy the relevant `skills/nat-*` folders into the skills or rules directory supported by your agent. Each skill follows the common `SKILL.md` directory pattern:

```text
skills/nat-workflow-creation/
├── SKILL.md
└── references/
```

## Example Prompts

Use prompts like the following after installing the skill:

- "Use the NeMo Agent toolkit skill to scaffold a ReAct workflow that calls a custom support-ticket lookup function."
- "Create an evaluation config for this workflow with a JSON dataset and a trajectory evaluator."
- "Add OpenTelemetry tracing to this workflow and export spans to a local file."
- "Run `nat info components` and fix this workflow YAML so every `_type` is registered."
- "Add an optimizer config that tunes the LLM temperature and prompt for this workflow."

## Skill Contents

The NeMo Agent toolkit skills include:

- `skills/nat-user-rules/SKILL.md`: General behavior and cross-skill routing.
- `skills/nat-installation/SKILL.md`: Installation and first workflow setup.
- `skills/nat-workflow-creation/SKILL.md`: Workflow YAML, LLM configuration, and CLI usage.
- `skills/nat-agent-configuration/SKILL.md`: Agent selection and sub-agent composition.
- `skills/nat-tools-and-functions/SKILL.md`: Custom tools, functions, and extension patterns.
- `skills/nat-evaluation/SKILL.md`: Evaluation methodology, datasets, and evaluators.
- `skills/nat-optimization/SKILL.md`: Optimizer configuration and output interpretation.
- `skills/nat-telemetry/SKILL.md`: Logging, tracing, profiling, and telemetry exporters.
- `skills/nat-mcp-and-serving/SKILL.md`: MCP and serving workflows.
- `skills/skill-evolution/SKILL.md`: Skill maintenance, routing changes, and reusable lessons learned.

For the top-level task routing table, see `AGENTS.md`.
