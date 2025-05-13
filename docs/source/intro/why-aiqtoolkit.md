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

# Why Use NVIDIA Agent Intelligence Toolkit?

Agent Intelligence Toolkit (AIQ Toolkit) is a flexible library that allows for easy connection of existing enterprise agents, across any framework, to data sources and tools. The core principle of this library is every agent, tool, and agentic workflow exists as a function call - enabling composability between these agents, tools, and workflows that allow developers to build once and reuse in different scenarios. This makes AIQ Toolkit able to work across any agentic framework, combining existing development work and reducing the need to replatform. This library is agentic framework agnostic, long term memory, and data source agnostic. It also allows development teams to move quickly if they already are developing with agents- focusing on what framework best meets their needs, while providing a holistic approach to evaluation and observability. A core component of AIQ Toolkit is the profiler, which can be run to uncover hidden latencies and suboptimal models/tools for specific, granular parts of pipelines. An evaluation system is provided to help users verify and maintain the accuracy of the RAG and E2E system configurations.

## Key Features

- [**Framework Agnostic:**](./../concepts/plugins.md) Works with any agentic framework, so you can use your current technology stack without replatforming.
- [**Reusability:**](./../guides/sharing-workflows-and-tools.md) Every agent, tool, or workflow can be combined and repurposed, allowing developers to leverage existing work in new scenarios.
- [**Rapid Development:**](./../guides/create-customize-workflows.md) Start with a pre-built agent, tool, or workflow, and customize it to your needs.
- [**Profiling:**](./../guides/profiler.md) Profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.
- [**Observability:**](./../guides/observe-workflow-with-phoenix.md) Monitor and debug your workflows with any OpenTelemetry-compatible observability tool.
- [**Evaluation System:**](./../guides/evaluate.md) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.
- [**User Interface:**](./../guides/using-aiqtoolkit-ui-and-server.md) Use the AIQ Toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.
- [**Full MCP Support**](./../guides/mcp-server.md) Compatible with Model Context Protocol (MCP). You can use AIQ Toolkit as an MCP Client to connect to and use tools served by remote MCP servers. You can also use AIQ Toolkit as an MCP Server to publish tools via MCP.



## Coming Soon

AIQ Toolkit is still under active development. Here are some of the features we are working on adding in the near future:

- Additional end-to-end agentic use case for software engineering (Q2 2025)
<!-- vale off -->
- AI-Q Blueprint customization guide (Q2 2025)
<!-- vale on -->
- Optional integration of NeMo Guardrails (Q2 2025)
- Agentic system level accelerations in partnership with Dynamo (Q3 2025)
- Flexible deployment and workflow instantiation option (Q4 2025)
- Addition of data feedback and improvement loop (flywheel) (Q4 2025)


## What AIQ Toolkit Is

- A **lightweight, unifying library** that makes every agent, tool, and workflow you already have work together, just as simple function calls work together in complex software applications.
- An **end-to-end agentic profiler**, allowing you to track input/output tokens and timings at a granular level for every tool and agent, regardless of the amount of nesting.
- A way to accomplish **end-to-end evaluation and observability**. With the potential to wrap and hook into every function call, AIQ Toolkit can output observability data to your platform of choice. It also includes an end-to-end evaluation system, allowing you to consistently evaluate your complex, multi-framework workflows in the exact same way as you develop and deploy them.
- A **compliment to existing agentic frameworks** and memory tools, not a replacement.
- **100% opt in.** While we encourage users to wrap (decorate) every tool and agent to get the most out of the profiler, you have the freedom to integrate to whatever level you want - tool level, agent level, or entire workflow level. You have the freedom to start small and where you believe you’ll see the most value and expand from there.


## What AIQ Toolkit Is Not

- **An agentic framework.** AIQ Toolkit is built to work side-by-side and around existing agentic frameworks, including LangChain, Llama Index, Crew.ai, Microsoft Semantic Kernel, MCP, and many more - including customer enterprise frameworks and simple Python agents.
- **An attempt to solve agent-to-agent communication.** Agent communication is best handled over existing protocols, such as HTTP, gRPC, and sockets.
- **An observability platform.** While AIQ Toolkit is able to collect and transmit fine-grained telemetry to help with optimization and evaluation, it does not replace your preferred observability platform and data collection application.
