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

![NVIDIA Agent Intelligence Toolkit](./_static/aiqtoolkit_banner.png "AIQ Toolkit banner image")

# NVIDIA Agent Intelligence Toolkit Overview

Agent Intelligence Toolkit (AIQ Toolkit) is a flexible library that allows for easy connection of existing enterprise agents, across any framework, to data sources and tools. The core principle of this library is every agent, tool, and agentic workflow exists as a function call - enabling composability between these agents, tools, and workflows that allow developers to build once and reuse in different scenarios. This makes AIQ Toolkit able to work across any agentic framework, combining existing development work and reducing the need to replatform. This library is agentic framework agnostic, long term memory, and data source agnostic. It also allows development teams to move quickly if they already are developing with agents- focusing on what framework best meets their needs, while providing a holistic approach to evaluation and observability. A core component of AIQ Toolkit is the profiler, which can be run to uncover hidden latencies and suboptimal models/tools for specific, granular parts of pipelines. An evaluation system is provided to help users verify and maintain the accuracy of the RAG and E2E system configurations.

With AIQ Toolkit, you can move quickly, experiment freely, and ensure reliability across all your agent-driven projects.

:::{note}
Agent Intelligence Toolkit was previously known as <!-- vale off -->AgentIQ<!-- vale on -->, however the API has not changed and is fully compatible with previous releases. Users should update their dependencies to depend on `aiqtoolkit` instead of `agentiq`. I transitional package named `agentiq` is available for backwards compatibility, but will be removed in the future.
:::

## Key Features

- [**Framework Agnostic:**](./extend/plugins.md) Works with any agentic framework, so you can use your current technology stack without replatforming.
- [**Reusability:**](./extend/sharing-components.md) Every agent, tool, or workflow can be combined and repurposed, allowing developers to leverage existing work in new scenarios.
- [**Rapid Development:**](./tutorials/index.md) Start with a pre-built agent, tool, or workflow, and customize it to your needs.
- [**Profiling:**](./workflows/profiler.md) Profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.
- [**Observability:**](./workflows/observe/index.md) Monitor and debug your workflows with any OpenTelemetry-compatible observability tool, with examples using [Phoenix](./workflows/observe/observe-workflow-with-phoenix.md) and [W&B Weave](./workflows/observe/observe-workflow-with-weave.md).
- [**Evaluation System:**](./workflows/evaluate.md) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.
- [**User Interface:**](./quick-start/launching-ui.md) Use the AIQ Toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.
- [**MCP Compatibility**](./workflows/mcp/mcp-client.md) Compatible with Model Context Protocol (MCP), allowing tools served by MCP Servers to be used as AIQ Toolkit functions.

## Framework Integrations

To keep the library lightweight, many of the first party plugins supported by AIQ Toolkit are located in separate distribution packages. For example, the `aiqtoolkit-langchain` distribution contains all the LangChain specific plugins and the `aiqtoolkit-mem0ai` distribution contains the Mem0 specific plugins.

To install these first-party plugin libraries, you can use the full distribution name (for example, `aiqtoolkit-langchain`) or use the `aiqtoolkit[langchain]` extra distribution. A full list of the supported extras is listed below:

- `aiqtoolkit[crewai]` or `aiqtoolkit-crewai` - [CrewAI](https://www.crewai.com/) specific plugins
- `aiqtoolkit[langchain]` or `aiqtoolkit-langchain` - [LangChain](https://www.langchain.com/) specific plugins
- `aiqtoolkit[llama-index]` or `aiqtoolkit-llama-index` - [LlamaIndex](https://www.llamaindex.ai/) specific plugins
- `aiqtoolkit[mem0ai]` or `aiqtoolkit-mem0ai` - [Mem0](https://mem0.ai/) specific plugins
- `aiqtoolkit[semantic-kernel]` or `aiqtoolkit-semantic-kernel` - [Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/) specific plugins
- `aiqtoolkit[test]` or `aiqtoolkit-test` - AIQ Toolkit Test specific plugins
- `aiqtoolkit[zep-cloud]` or `aiqtoolkit-zep-cloud` - [Zep](https://www.getzep.com/) specific plugins

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

<!-->
## Links

To learn more about AIQ Toolkit, see the following links:
* [Get Started](./quick-start/installing.md)
* [Create and Customize Workflows](./tutorials/index.md)
* [Sharing Components](./extend/sharing-components.md)
* [Evaluating Workflows](./workflows/evaluate.md)
* [Profiling a Workflow](./workflows/profiler.md)
* [Observing a Workflow with Phoenix](./workflows/observe/observe-workflow-with-phoenix.md)
* [Command Line Interface](./reference/cli.md)
-->

## Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/AIQToolkit/issues) if you have any feedback or feature requests.

```{toctree}
:hidden:
:caption: About Agent Intelligence Toolkit
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
```

```{toctree}
:hidden:
:caption: Reference

./api/index.rst
./reference/interactive-models.md
API Server Endpoints <./reference/api-server-endpoints.md>
./reference/websockets.md
Command Line Interface (CLI) <./reference/cli.md>
Evaluation <./reference/evaluate.md>
Evaluation Endpoints <./reference/evaluate-api.md>
Troubleshooting <./troubleshooting.md>
```

```{toctree}
:hidden:
:caption: Resources

Code of Conduct <./resources/code-of-conduct.md>
Contributing <./resources/contributing.md>
./resources/running-ci-locally.md
./support.md
./resources/licensing
```
