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

<!--
  SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Agent Examples

The agent examples demonstrate how AIQ Toolkit accelerates and enables AI Agent development.
The examples showcase 5 distinct AI Agent architectures solving a similar problem in different ways.
By leveraging AIQ Toolkit’s plugin system and the `Builder` object, we can utilize both pre-built and custom agentic workflows and tools in a flexible manner.


* [ReAct Agent Example](./react/README.md)
* [Tool Calling Agent Example](./tool_calling/README.md)
* [Mixture of Agents Example](./mixture_of_agents/README.md) - A ReAct Agent calling multiple Tool Calling Agents

## Learn More

For a deeper dive into the AI Agents utilized in the examples, refer to the component documentation:
- [ReAct Agent](../../docs/source/components/react-agent.md)
- [Reasoning Agent](../../docs/source/components/reasoning-agent.md)
- [Tool Calling Agent](../../docs/source/components/tool-calling-agent.md)
