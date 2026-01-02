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
# About the Tool Calling Agent
A tool calling agent is an AI system that directly invokes external tools based on structured function definitions. Unlike ReAct agents, it does not reason between steps but instead selects tools based on predefined function schemas. The agent examines the tool name, description, and input parameter schema to determine which tool to invoke. In order to use the tool calling agent, you must choose an LLM that has tool-calling support.

The agent uses the NVIDIA NeMo Agent toolkit core library agents and tools to simplify your development experience and deployment. Additionally, you can customize prompts in your YAML config files for your specific needs.

```{toctree}
:hidden:
:caption: Tool

Configure Tool Calling Agent<./tool-calling-agent.md>
```