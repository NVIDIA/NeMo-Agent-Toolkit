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

## Installation


::::{tab-set}
:sync-group: category

   :::{tab-item} pip
   :selected:
   :sync: pip
   pip install nvidia-nat
   :::

   :::{tab-item} uv
   :sync: uv
   uv pip install nvidia-nat
   :::

::::

## Key Features

- [**Framework Agnostic:**](./reference/frameworks-overview.md) NeMo Agent toolkit works side-by-side and around existing agentic frameworks, such as [LangChain](https://www.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/), [CrewAI](https://www.crewai.com/), [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/), [Google ADK](https://github.com/google/adk-python), as well as customer enterprise frameworks and simple Python agents. This allows you to use your current technology stack without replatforming. NeMo Agent toolkit complements any existing agentic framework or memory tool you're using and isn't tied to any specific agentic framework, long-term memory, or data source.

- [**Reusability:**](./extend/sharing-components.md) Every agent, tool, and agentic workflow in this library exists as a function call that works together in complex software applications. The composability between these agents, tools, and workflows allows you to build once and reuse in different scenarios.

- [**Rapid Development:**](./tutorials/index.md) Start with a pre-built agent, tool, or workflow, and customize it to your needs. This allows you and your development teams to move quickly if you're already developing with agents.

- [**Profiling:**](./workflows/profiler.md) Use the profiler to profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.

- [**Observability:**](./workflows/observe/index.md) Monitor and debug your workflows with dedicated integrations for popular observability platforms such as Phoenix, Weave, and Langfuse, plus compatibility with OpenTelemetry-based systems. Track performance, trace execution flows, and gain insights into your agent behaviors.

- [**Evaluation System:**](./workflows/evaluate.md) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.

- [**User Interface:**](./quick-start/launching-ui.md) Use the NeMo Agent toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.

- [**Full MCP Support:**](./workflows/mcp/index.md) Compatible with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). You can use NeMo Agent toolkit as an [MCP client](./workflows/mcp/mcp-client.md) to connect to and use tools served by remote MCP servers. You can also use NeMo Agent toolkit as an [MCP server](./workflows/mcp/mcp-server.md) to publish tools via MCP.

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
./get-started/getting-help.md
Troubleshooting <./get-started/troubleshooting.md>
```

```{toctree}
:hidden:
:caption: Build Workflows

About Workflows <./build-workflows/about-workflows.md>
Workflow Configuration <./build-workflows/workflow-configuration.md>
Using Remote MCP Functions <./build-workflows/using-remote-mcp-functions.md>
Interactive Models <./build-workflows/interactive-models.md>
Middleware <./build-workflows/middleware.md>
/build-workflows/tuning/index.md
```

```{toctree}
:hidden:
:caption: Run Workflows

Run Workflows <./run-workflows/run-workflows.md>
Command Line Interface (CLI) <./run-workflows/cli.md>
API Server and User Interface <./run-workflows/launching-ui.md>
MCP Server <./run-workflows/mcp-server.md>
```

```{toctree}
:hidden:
:caption: Components

Agents <./components/agents/index.md>
Functions <./components/functions/index.md>
./components/embedders.md
./components/retrievers.md
./components/llms/index.md
./components/auth/index.md
./components/integrations/index.md
Memory Module <./components/memory.md>
Object Store <./components/object-store.md>
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

./api/index.rst
API Authentication <./reference/api-authentication.md>
API Server Endpoints <./reference/api-server-endpoints.md>
WebSockets <./reference/websockets.md>

Cursor Rules Reference <./reference/cursor-rules-reference.md>
Evaluation <./reference/evaluate.md>
Evaluation Endpoints <./reference/evaluate-api.md>
```

```{toctree}
:hidden:
:caption: Resources

FAQ <./resources/faq.md>
Security Considerations <./resources/security-considerations.md>
Code of Conduct <./resources/code-of-conduct.md>
Migration Guide <./resources/migration-guide.md>
Contributing <./resources/contributing.md>
Running Tests <./resources/running-tests.md>
./resources/running-ci-locally.md
./resources/licensing.md
Release Notes <./release-notes.md>
```

<!-- This role is needed at the index to set the default backtick role -->
```{eval-rst}
.. role:: py(code)
   :language: python
   :class: highlight
```
