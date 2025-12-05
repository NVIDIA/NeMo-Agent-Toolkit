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

![NVIDIA NeMo Agent Toolkit](./_static/banner.png "NeMo Agent toolkit banner image")

# NVIDIA NeMo Agent Toolkit Overview

NVIDIA NeMo Agent toolkit is a flexible, lightweight, and unifying library that allows you to easily connect existing enterprise agents to data sources and tools across any framework.

## Install

::::{tab-set}
:sync-group: install-tool

   :::{tab-item} uv
   :selected:
   :sync: uv
   ```bash
   uv pip install nvidia-nat
   ```
   :::


   :::{tab-item} pip
   :sync: pip
   ```bash
   pip install nvidia-nat
   ```
   :::

::::

For detailed installation instructions, including optional dependencies, please refer to the [Install Guide](./get-started/installing.md).

## Key Features

- [**Framework Agnostic:**](./components/integrations/frameworks.md) NeMo Agent toolkit works side-by-side and around existing agentic frameworks, such as [LangChain](https://www.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/), [CrewAI](https://www.crewai.com/), [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/), [Google ADK](https://github.com/google/adk-python), as well as customer enterprise frameworks and simple Python agents. This allows you to use your current technology stack without replatforming. NeMo Agent toolkit complements any existing agentic framework or memory tool you're using and isn't tied to any specific agentic framework, long-term memory, or data source.

- [**Reusability:**](./components/sharing-components.md) Every agent, tool, and agentic workflow in this library exists as a function call that works together in complex software applications. The composability between these agents, tools, and workflows allows you to build once and reuse in different scenarios.

- [**Rapid Development:**](./get-started/tutorials/index.md) Start with a pre-built agent, tool, or workflow, and customize it to your needs. This allows you and your development teams to move quickly if you're already developing with agents.

- [**Profiling:**](./improve-workflows/profiler.md) Use the profiler to profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.

- [**Observability:**](./run-workflows/observe/observe.md) Monitor and debug your workflows with dedicated integrations for popular observability platforms such as Phoenix, Weave, and Langfuse, plus compatibility with OpenTelemetry-based systems. Track performance, trace execution flows, and gain insights into your agent behaviors.

- [**Evaluation System:**](./improve-workflows/evaluate.md) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.

- [**User Interface:**](./run-workflows/launching-ui.md) Use the NeMo Agent toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.

- [**Full MCP Support:**](./build-workflows/mcp.md) Compatible with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). You can use NeMo Agent toolkit as an [MCP client](./build-workflows/mcp.md) to connect to and use tools served by remote MCP servers. You can also use NeMo Agent toolkit as an [MCP server](./run-workflows/mcp-server.md) to publish tools via MCP.

## FAQ
For frequently asked questions, refer to [FAQ](./resources/faq.md).

## Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) if you have any feedback or feature requests.


:::{note}
NeMo Agent toolkit was previously known as <!-- vale off -->AgentIQ<!-- vale on -->, however the API has not changed and is fully compatible with previous releases. Users should update their dependencies to depend on `nvidia-nat` instead of `aiqtoolkit` or `agentiq`. The transitional packages named `aiqtoolkit` and `agentiq` are available for backwards compatibility, but will be removed in the future.
:::


```{toctree}
:hidden:
About <self>
```

```{toctree}
:hidden:
:caption: Get Started

Install <./get-started/installing.md>
Quick Start Guide <./get-started/quick-start.md>
Tutorials <./get-started/tutorials/index.md>
Release Notes <./get-started/release-notes.md>
```

```{toctree}
:hidden:
:caption: Build Workflows

About Workflows <./build-workflows/about-workflows.md>
Workflow Configuration <./build-workflows/workflow-configuration.md>
./build-workflows/functions-and-function-groups/index.md
./build-workflows/llms/index.md
./build-workflows/embedders.md
./build-workflows/retrievers.md
Memory Module <./build-workflows/memory.md>
Object Store <./build-workflows/object-store.md>
MCP <./build-workflows/mcp.md>
Interactive Workflows <./build-workflows/interactive-workflows.md>
Middleware <./build-workflows/middleware.md>
```

```{toctree}
:hidden:
:caption: Run Workflows

Run Workflows <./run-workflows/run-workflows.md>
Command Line Interface (CLI) <./run-workflows/cli.md>
./run-workflows/observe/observe.md
API Server and User Interface <./run-workflows/launching-ui.md>
MCP Server <./run-workflows/mcp-server.md>
```

```{toctree}
:hidden:
:caption: Improve Workflows

Evaluate Workflows <./improve-workflows/evaluate.md>
Profiling and Performance Monitoring <./improve-workflows/profiler.md>
Optimizer Guide <./improve-workflows/optimizer.md>
Sizing Calculator <./improve-workflows/sizing-calc.md>
```

```{toctree}
:hidden:
:caption: Components

Agents <./components/agents/index.md>
./components/functions/index.md
./components/auth/index.md
./components/integrations/index.md
Test Time Compute <./components/test-time-compute.md>
Sharing Components <./components/sharing-components.md>
```

```{toctree}
:hidden:
:caption: Extend

Plugins <./extend/plugins.md>
Custom Components <./extend/custom-components/index.md>
./extend/testing/index.md
```

```{toctree}
:hidden:
:caption: Reference

Python API <./api/index.rst>
API Server Endpoints <./reference/api-server-endpoints.md>
Evaluation API Endpoints <./reference/evaluate-api.md>
WebSocket Message Schema <./reference/websockets.md>
```

```{toctree}
:hidden:
:caption: Resources

FAQ <./resources/faq.md>
./resources/support.md
Troubleshooting <./resources/troubleshooting.md>
Migration Guide <./resources/migration-guide.md>
Security Considerations <./resources/security-considerations.md>
Contributing <./resources/contributing/index.md>
```

<!-- This role is needed at the index to set the default backtick role -->
```{eval-rst}
.. role:: py(code)
   :language: python
   :class: highlight
```
