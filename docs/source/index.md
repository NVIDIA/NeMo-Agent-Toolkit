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

![NVIDIA NeMo Agent Toolkit](./_static/banner.png "NeMo Agent toolkit banner image")

# NVIDIA NeMo Agent Toolkit Overview

NVIDIA NeMo Agent toolkit is a flexible, lightweight, and unifying library that allows you to easily connect existing enterprise [agents](./components/agents/index.md) to data sources and [tools](./build-workflows/functions-and-function-groups/functions.md#agents-and-tools) across any framework.

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

For detailed installation instructions, including optional dependencies, please refer to the [Install Guide](./get-started/installation.md).

## Key Features

- [**Framework Agnostic:**](./components/integrations/frameworks.md) NeMo Agent toolkit works side-by-side and around existing agentic frameworks, such as [LangChain](https://www.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/), [CrewAI](https://www.crewai.com/), [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/), [Google ADK](https://github.com/google/adk-python), as well as customer enterprise frameworks and simple Python agents. This allows you to use your current technology stack without replatforming. NeMo Agent toolkit complements any existing agentic framework or [memory](./build-workflows/memory.md) tool you're using and isn't tied to any specific agentic framework, long-term memory, or data source.

- [**Reusability:**](./components/sharing-components.md) Every agent, tool, and agentic [workflow](./build-workflows/about-building-workflows.md) in this library exists as a function call that works together in complex software applications. The composability between these agents, tools, and workflows allows you to build once and reuse in different scenarios.

- [**Rapid Development:**](./get-started/tutorials/index.md) Start with a pre-built agent, tool, or workflow, and customize it to your needs. This allows you and your development teams to move quickly if you're already developing with agents.

- [**Profiling:**](./improve-workflows/profiler.md) Use the profiler to profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.

- [**Observability:**](./run-workflows/observe/observe.md) Monitor and debug your workflows with dedicated integrations for popular observability platforms such as Phoenix, Weave, and Langfuse, plus compatibility with OpenTelemetry-based systems. Track performance, trace execution flows, and gain insights into your agent behaviors.

- [**Evaluation System:**](./improve-workflows/evaluate.md) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.

- [**User Interface:**](./run-workflows/launching-ui.md) Use the NeMo Agent toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.

- [**Full MCP Support:**](./build-workflows/mcp-client.md) Compatible with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). You can use NeMo Agent toolkit as an [MCP client](./build-workflows/mcp-client.md) to connect to and use tools served by remote MCP servers. You can also use NeMo Agent toolkit as an [MCP server](./run-workflows/mcp-server.md) to publish tools via MCP.

- [**A2A Protocol Support:**](./components/integrations/a2a.md) Compatible with [Agent-to-Agent (A2A) Protocol](https://a2aproject.org/). You can use NeMo Agent toolkit as an [A2A client](./build-workflows/a2a-client.md) to connect to and delegate tasks to remote A2A agents. You can also use NeMo Agent toolkit as an [A2A server](./run-workflows/a2a-server.md) to publish workflows as discoverable A2A agents.

## Hello World Example

Before getting started, it's possible to run this simple workflow and many other examples in Google Colab with no setup. Click here to open the introduction notebook: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/NeMo-Agent-Toolkit/).

1. Install NeMo Agent toolkit along with the LangChain integration plugin:

   ::::{tab-set}
   :sync-group: install-tool

   :::{tab-item} uv
   :selected:
   :sync: uv

   ```bash
   uv pip install "nvidia-nat[langchain]"
   ```

   :::

   :::{tab-item} pip
   :sync: pip

   ```bash
   pip install "nvidia-nat[langchain]"
   ```

   :::

   ::::

2. Ensure you have set the `NVIDIA_API_KEY` environment variable to allow the example to use NVIDIA NIMs. An API key can be obtained by visiting [`build.nvidia.com`](https://build.nvidia.com/) and creating an account.

   ```bash
   export NVIDIA_API_KEY=<your_api_key>
   ```

3. Create the NeMo Agent Toolkit workflow configuration file. This file will define the agents, tools, and workflows that will be used in the example. Save the following as `workflow.yml`:

   ```yaml
   functions:
     # Add a tool to search wikipedia
     wikipedia_search:
       _type: wiki_search
       max_results: 2

   llms:
     # Tell NeMo Agent Toolkit which LLM to use for the agent
     nim_llm:
       _type: nim
       model_name: meta/llama-3.1-70b-instruct
       temperature: 0.0

   workflow:
     # Use an agent that 'reasons' and 'acts'
     _type: react_agent
     # Give it access to our wikipedia search tool
     tool_names: [wikipedia_search]
     # Tell it which LLM to use
     llm_name: nim_llm
     # Make it verbose
     verbose: true
     # Retry up to 3 times
     parse_agent_response_max_retries: 3
   ```

4. Run the Hello World example using the `nat` CLI and the `workflow.yml` file.

   ```bash
   nat run --config_file workflow.yml --input "List five subspecies of Aardvarks"
   ```

   This will run the workflow and output the results to the console.

   ```console
   Workflow Result:
   ['Here are five subspecies of Aardvarks:\n\n1. Orycteropus afer afer (Southern aardvark)\n2. O. a. adametzi  Grote, 1921 (Western aardvark)\n3. O. a. aethiopicus  Sundevall, 1843\n4. O. a. angolensis  Zukowsky & Haltenorth, 1957\n5. O. a. erikssoni  Lönnberg, 1906']
   ```

## FAQs

For frequently asked questions, refer to [FAQs](./resources/faq.md).

## Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) if you have any feedback or feature requests.


```{toctree}
:hidden:
:caption: About NeMo Agent Toolkit
Overview <self>
Release Notes <./release-notes.md>
```

```{toctree}
:hidden:
:caption: Get Started

Installation <./get-started/installation.md>
Quick Start <./get-started/quick-start.md>
Tutorials <./get-started/tutorials/index.md>
```

```{toctree}
:hidden:
:caption: Build Workflows

About <./build-workflows/about-building-workflows.md>
Workflow Configuration <./build-workflows/workflow-configuration.md>
./build-workflows/functions-and-function-groups/index.md
./build-workflows/llms/index.md
Embedders <./build-workflows/embedders.md>
Retrievers <./build-workflows/retrievers.md>
Memory <./build-workflows/memory.md>
Object Stores <./build-workflows/object-store.md>
MCP <./build-workflows/mcp-client.md>
A2A <./build-workflows/a2a-client.md>
./build-workflows/advanced/index.md
```

```{toctree}
:hidden:
:caption: Run Workflows

About <./run-workflows/about-running-workflows.md>
Existing Agents <./run-workflows/existing-agents/index.md>
./run-workflows/observe/observe.md
API Server and User Interface <./run-workflows/launching-ui.md>
MCP Server <./run-workflows/mcp-server.md>
A2A Server <./run-workflows/a2a-server.md>
```

```{toctree}
:hidden:
:caption: Improve Workflows

About <./improve-workflows/about-improving-workflows.md>
Evaluate Workflows <./improve-workflows/evaluate.md>
Profiling and Performance Monitoring <./improve-workflows/profiler.md>
Optimizer Guide <./improve-workflows/optimizer.md>
Sizing Calculator <./improve-workflows/sizing-calc.md>
Test Time Compute <./improve-workflows/test-time-compute.md>
Finetuning <./improve-workflows/finetuning/index.md>
```

```{toctree}
:hidden:
:caption: Components

Agents <./components/agents/index.md>
./components/functions/index.md
./components/auth/index.md
./components/integrations/index.md
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
./reference/rest-api/index.md
Command Line Interface (CLI) <./reference/cli.md>
```

```{toctree}
:hidden:
:caption: Resources

FAQs <./resources/faq.md>
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
