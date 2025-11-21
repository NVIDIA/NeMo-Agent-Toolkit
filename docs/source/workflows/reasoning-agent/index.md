<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# About Reasoning Agent

The reasoning agent is an AI system that uses the NVIDIA NeMo Agent toolkit core library agents and tools to directly invoke an underlying function while performing reasoning on top. Unlike ReAct agents, it reasons ahead of time through planning rather than between steps. However, an LLM that supports reasoning needs to be chosen for use with a reasoning agent. Additionally, you can customize prompts in your YAML config file for your specific needs. 

To configure your reasoning agent, refer to [Configure the Reasoning Agent](./reasoning-agent.md).

```{toctree}
:hidden:
:caption: Reasoning

Configure Reasoning Agent<./reasoning-agent.md>
```