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


<!-- This role is needed at the index to set the default backtick role -->
```{eval-rst}
.. role:: py(code)
   :language: python
   :class: highlight
```

![NVIDIA Agent Intelligence Toolkit](./_static/aiqtoolkit_banner.png "AIQ toolkit banner image")

# NVIDIA Agent Intelligence Toolkit Overview

NVIDIA Agent Intelligence (AIQ) toolkit is a flexible, lightweight, and unifying library that allows you to easily connect existing enterprise agents to data sources and tools across any framework.


:::{note}
Agent Intelligence toolkit was previously known as <!-- vale off -->AgentIQ<!-- vale on -->, however the API has not changed and is fully compatible with previous releases. Users should update their dependencies to depend on `aiqtoolkit` instead of `agentiq`. The transitional package named `agentiq` is available for backwards compatibility, but will be removed in the future.
:::

## Key Features

- [**Framework Agnostic:**](./quick-start/installing.md#framework-integrations) AIQ toolkit works side-by-side and around existing agentic frameworks, such as [LangChain](https://www.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/), [CrewAI](https://www.crewai.com/), and [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/), as well as customer enterprise frameworks and simple Python agents. This allows you to use your current technology stack without replatforming. AIQ toolkit complements any existing agentic framework or memory tool you're using and isn't tied to any specific agentic framework, long-term memory, or data source.

- [**Reusability:**](./extend/sharing-components.md) Every agent, tool, and agentic workflow in this library exists as a function call that works together in complex software applications. The composability between these agents, tools, and workflows allows you to build once and reuse in different scenarios.

- [**Rapid Development:**](./tutorials/index.md) Start with a pre-built agent, tool, or workflow, and customize it to your needs. This allows you and your development teams to move quickly if you're already developing with agents.

- [**Profiling:**](./workflows/profiler.md) Use the profiler to profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.

- [**Observability:**](./workflows/observe/index.md) Monitor and debug your workflows with any OpenTelemetry-compatible observability tool, with examples using [Phoenix](./workflows/observe/observe-workflow-with-phoenix.md) and [W&B Weave](./workflows/observe/observe-workflow-with-weave.md).

- [**Evaluation System:**](./workflows/evaluate.md) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.

- [**User Interface:**](./quick-start/launching-ui.md) Use the AIQ toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.

- [**Full MCP Support:**](./workflows/mcp/index.md) Compatible with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). You can use AIQ toolkit as an [MCP client](./workflows/mcp/mcp-client.md) to connect to and use tools served by remote MCP servers. You can also use AIQ toolkit as an [MCP server](./workflows/mcp/mcp-server.md) to publish tools via MCP.

## FAQ
For frequently asked questions, refer to [FAQ](./resources/faq.md).

## Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) if you have any feedback or feature requests.

```{toctree}
:hidden:
:caption: About NVIDIA Agent Intelligence Toolkit
Overview <self>
Release Notes <./release-notes.md>
```

```{toctree}
:hidden:
:caption: Get Started

Quick Start Guide <./quick-start/index.md>
Tutorials <./tutorials/index.md>
```

```{toctree}
:hidden:
:caption: Manage Workflows

About Workflows <./workflows/about/index.md>
./workflows/run-workflows.md
Workflow Configuration <./workflows/workflow-configuration.md>
Functions <./workflows/functions/index.md>
./workflows/mcp/index.md
Evaluate Workflows <./workflows/evaluate.md>
Profiling Workflows <./workflows/profiler.md>
Sizing Calculator <./workflows/sizing-calc.md>
./workflows/using-local-llms.md
./workflows/observe/index.md
```

```{toctree}
:hidden:
:caption: Store and Retrieve

Memory Module <./store-and-retrieve/memory.md>
./store-and-retrieve/retrievers.md
```

```{toctree}
:hidden:
:caption: Extend

Writing Custom Functions <./extend/functions.md>
Extending the AIQ Toolkit Using Plugins <./extend/plugins.md>
Sharing Components <./extend/sharing-components.md>
Adding a Custom Evaluator <./extend/custom-evaluator.md>
./extend/adding-a-retriever.md
./extend/memory.md
Adding an LLM Provider <./extend/adding-an-llm-provider.md>
Integrating AWS Bedrock Models <./extend/integrating-aws-bedrock-models.md>
Cursor Rules Developer Guide <./extend/cursor-rules-developer-guide.md>
```

```{toctree}
:hidden:
:caption: Reference

./api/index.rst
./reference/interactive-models.md
API Server Endpoints <./reference/api-server-endpoints.md>
./reference/websockets.md
Command Line Interface (CLI) <./reference/cli.md>
Cursor Rules Reference <./reference/cursor-rules-reference.md>
Evaluation <./reference/evaluate.md>
Evaluation Endpoints <./reference/evaluate-api.md>
Troubleshooting <./troubleshooting.md>
```

```{toctree}
:hidden:
:caption: Resources

FAQ <./resources/faq.md>
Code of Conduct <./resources/code-of-conduct.md>
Contributing <./resources/contributing.md>
./resources/running-ci-locally.md
./support.md
./resources/licensing
```
