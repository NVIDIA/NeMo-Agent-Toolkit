<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Agents

An [agent](https://developer.nvidia.com/blog/introduction-to-llm-agents/#what_is_an_ai_agent) is a system that can use an [LLM](../../build-workflows/llms/index.md) to reason through a problem, create a plan to solve the problem, and execute the plan with the help of a set of [tools](../../build-workflows/functions-and-function-groups/functions.md#agents-and-tools). Refer to [Introduction to LLM Agents](https://developer.nvidia.com/blog/introduction-to-llm-agents/) for more details on this. In NeMo Agent toolkit, agents are implemented as a special type of [function](../../build-workflows/functions-and-function-groups/functions.md) that can orchestrate other functions.

NeMo Agent toolkit includes several agents out of the box to choose from. In addition to this NeMo Agent toolkit makes it easy to write a custom agent, for an example of this refer to the Alert Triage example (`examples/advanced_agents/alert_triage_agent`) in the [repository](https://github.com/NVIDIA/NeMo-Agent-Toolkit).

NeMo Agent Toolkit also provides an [Automatic Memory Wrapper](./auto-memory-wrapper/index.md) that enhances any existing agent with automatic memory capture and retrieval capabilities.


```{toctree}
:titlesonly:

ReAct Agent <./react-agent/index.md>
Reasoning Agent <./reasoning-agent/index.md>
ReWOO Agent <./rewoo-agent/index.md>
Responses API and Agent <./responses-api-and-agent/index.md>
Router Agent <./router-agent/index.md>
Sequential Executor <./sequential-executor/index.md>
Tool Calling Agent <./tool-calling-agent/index.md>
Automatic Memory Wrapper <./auto-memory-wrapper/index.md>
```
