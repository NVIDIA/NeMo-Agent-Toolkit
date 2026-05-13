---
name: nat-agent-configuration
description: Use when selecting, configuring, composing, or troubleshooting NeMo Agent toolkit agents and control-flow components, including ReAct, tool-calling, ReWOO, reasoning, router, sequential, parallel, and sub-agent patterns.
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

# NeMo Agent toolkit Agent Configuration

Use this skill when the task is about choosing or wiring agent components.

## Workflow

1. Classify the workflow shape before choosing an agent.
2. Prefer built-in agents and control-flow components before custom Python.
3. Use sub-agents as tools when composing larger systems.
4. Validate agent behavior with a small `nat run` request before broad tests.

## References

- `references/agents.md`
- `references/additional-agent-types.md`
- `references/subagents.md`
- `references/subagent-patterns.md`
