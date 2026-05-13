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

# Sub-Agents: Composing Agents as Tools

Composing one agent as a tool inside another. Companion: [`subagent-patterns.md`](subagent-patterns.md) has four detailed composition patterns.

A powerful pattern in NeMo Agent toolkit is defining an agent under `functions:` and then referencing it as a tool for another agent. This enables hierarchical multi-agent systems where a top-level agent delegates to specialized sub-agents.

**Key principle:** Any agent defined under `functions:` can be referenced by name in another agent's `tool_names`, `branches`, or `tool_list`.

Four detailed composition patterns (Router with Sub-Agents, Reasoning wrapping, Parallel Fan-Out, Sequential Pipeline) are documented in [subagent-patterns.md](subagent-patterns.md). These patterns have not yet been adopted in internal projects but demonstrate the full composability of NeMo Agent toolkit agents.
