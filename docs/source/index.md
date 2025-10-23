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

```{raw} html
<div style="max-width:800px margin:auto">

   <img src="./_static/toolkit.png" 
      usemap="#toolkitmap" 
      style="width:100%; height:auto; display:block;">

   <map name="toolkitmap">
      <area shape="rect" coords="1,1,120,50" href="tutorials/create-a-new-workflow.html#customizing-the-tool-function" alt="Tool creator allows creating customization and integration of tools." title="a Tool is essentially a component that an Agent can call to perform some action. It acts as a bridge between the Agent (which handles reasoning, planning, and dialogue) and some external functionality (like APIs, file operations, or calculations)">
   
      <area shape="rect" coords="140,1,260,50" href="tutorials/create-a-new-workflow.html" alt="Workflow Creation" title="Workflows are the heart of the NeMo Agent toolkit because they define which agentic tools and models are used to perform a given task or series of tasks. The workflow configuration is a YAML file that specifies the tools and models to use in a workflow, along with general configuration settings.">
   
      <area shape="rect" coords="280,1,400,50" href="tutorials/create-a-new-workflow.html" alt="" title="The NeMo Agent toolkit Profiler Module provides insight by collecting and recording invocation level statistics. It persists data which can be used for measuring and predicting bottlenecks, latency and concurrency spikes among other performance metrics.">
   
      <area shape="rect" coords="430,1,550,50" href="workflows/about/react-agent.html" alt="Agent Configuration" title="Agent configuration defines the identity and capabilities of an individual agent, including its name, description, the tools it can invoke, LLM settings (like model and temperature), and optional memory for context tracking.">
   
      <area shape="rect" coords="570,1,690,50" href="workflows/observe/index.html" alt="" title="Observability is the ability to monitor and track an agent’s internal behavior, decisions, and interactions with tools. It provides insights into which tools were invoked, the inputs and outputs of each tool, the agent’s reasoning process, and any errors or warnings.">
   
      <area shape="rect" coords="725,1,850,50" href="store-and-retrieve/memory.html" alt="Tool creator allows creationg customization and integration of tools." title="Memory modules enable multi-turn or context-aware interactions. Advanced configurations can include tool prioritization, callbacks for logging or monitoring, and multi-agent orchestration">

      

      <area shape="rect" coords="100,320,225,370" href="workflows/about/reasoning-agent.html" alt="Agent Types" title="The reasoning agent is an AI system that directly invokes an underlying function while performing reasoning on top, &#10;&#10; ReAct (Reasoning and Acting) agent, Reasoning Agent which directly invokes an underlying function while performing reasoning on top. &#10; &#10; ReWOO (Reasoning WithOut Observation) are among the supported types. ">
      <area shape="rect" coords="650,320,775,370" href="extend/memory.html" alt="Memory subsystem" title="The NeMo Agent toolkit Memory subsystem allows storage and retrieve a user’s data in long-term memory. mem0, redis or zep memory is integrated in the code." >
      <area shape="rect" coords= "40,385,160,440" href="workflows/llms/index.html#llms" alt="LLMs Supported" title="NVIDIA NeMo Agent toolkit supports many LLM providers: NVIDIA NIM, OpenAI, AWS Bedrock, Azure OpenAI, LiteLLM">
      <area shape="rect" coords="180,385,300,440" href="workflows/embedders.html" alt="Embedders" title="NEmo Agent Toolkit supports converting raw input data (like text, images, or audio) into vector embeddings using standards like NVIDIA NIM, OpenAI, Azure OpenAI">
      <area shape="rect" coords="320,385,440,440" href="workflows/retrievers.html" alt="Retrievers" title="Retrievers are used to retrieve relevant documents from a vector database. NVIDIA NIM and Milvus are supported.">
      <area shape="rect" coords="470,385,575,440" href="workflows/mcp/index.html" alt="Model Control Protocol" title="NeMo Agent toolkit Model Context Protocol (MCP) integration includes client, server and transport configuration support">
      <area shape="rect" coords="600,385,710,440" href="workflows/observe/index.html" alt="Observation" title="The NeMo Agent toolkit uses a flexible, plugin-based observability system that provides comprehensive support for configuring logging, tracing, and metrics for workflows. Catalyst, Dynatrace, Galileo, Langfuse, OpenTelemetry Collector, Patronus, Phoenix, and W&B Weave are all supported with examples.">
      <area shape="rect" coords="730,385,850,440" href="reference/optimizer.html" alt="Optimizing" title="The NeMo Agent toolkit Optimizer uses a combination of techniques to find the best parameters for your workflow. Hyperparameter optimizing uses Optuna. Prompt optimization uses a genetic algorithm (GA) that evolves a population of prompt candidates.">

   </map>
   
<script>
window.addEventListener("load", function() {
  setTimeout(function() {
    if (typeof imageMapResize === "function") imageMapResize();
  }, 300);
  window.addEventListener("resize", function() {
    setTimeout(function() { if (typeof imageMapResize === "function") imageMapResize(); }, 200);
  });
});
</script
</div>
```

<!-- Load image-map-resizer to make regions scale correctly -->
<script src="_static/imageMapResizer.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function () {
  if (typeof imageMapResize === 'function') imageMapResize();
});
</script>


# NVIDIA NeMo Agent Toolkit Overview

NVIDIA NeMo Agent toolkit is a flexible, lightweight, and unifying library that allows you to easily connect existing enterprise agents to data sources and tools across any framework.


:::{note}
NeMo Agent toolkit was previously known as <!-- vale off -->AgentIQ<!-- vale on -->, however the API has not changed and is fully compatible with previous releases. Users should update their dependencies to depend on `nvidia-nat` instead of `aiqtoolkit` or `agentiq`. The transitional packages named `aiqtoolkit` and `agentiq` are available for backwards compatibility, but will be removed in the future.
:::

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

```{toctree}
:hidden:
:caption: About NVIDIA NeMo Agent Toolkit
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
./workflows/llms/index.md
./workflows/embedders.md
./workflows/retrievers.md
Functions <./workflows/functions/index.md>
./workflows/function-groups.md
./workflows/mcp/index.md
Evaluate Workflows <./workflows/evaluate.md>
Add Unit Tests for Tools <./workflows/add-unit-tests-for-tools.md>
Profiling Workflows <./workflows/profiler.md>
Sizing Calculator <./workflows/sizing-calc.md>
./workflows/observe/index.md
```

```{toctree}
:hidden:
:caption: Store and Retrieve

Memory Module <./store-and-retrieve/memory.md>
./store-and-retrieve/retrievers.md
Object Store <./store-and-retrieve/object-store.md>
```

```{toctree}
:hidden:
:caption: Extend

Writing Custom Functions <./extend/functions.md>
Writing Custom Function Groups <./extend/function-groups.md>
Extending the NeMo Agent Toolkit Using Plugins <./extend/plugins.md>
Sharing Components <./extend/sharing-components.md>
Adding a Custom Evaluator <./extend/custom-evaluator.md>
./extend/adding-a-retriever.md
./extend/memory.md
Adding an LLM Provider <./extend/adding-an-llm-provider.md>
Gated Fields <./extend/gated-fields.md>
Adding an Object Store Provider <./extend/object-store.md>
Adding an Authentication Provider <./extend/adding-an-authentication-provider.md>
Integrating AWS Bedrock Models <./extend/integrating-aws-bedrock-models.md>
Cursor Rules Developer Guide <./extend/cursor-rules-developer-guide.md>
Adding a Telemetry Exporter <./extend/telemetry-exporters.md>
```

```{toctree}
:hidden:
:caption: Reference

./api/index.rst
API Authentication <./reference/api-authentication.md>
Frameworks Overview <./reference/frameworks-overview.md>
Interactive Models <./reference/interactive-models.md>
API Server Endpoints <./reference/api-server-endpoints.md>
WebSockets <./reference/websockets.md>
Command Line Interface (CLI) <./reference/cli.md>
Cursor Rules Reference <./reference/cursor-rules-reference.md>
Evaluation <./reference/evaluate.md>
Evaluation Endpoints <./reference/evaluate-api.md>
Optimizer <./reference/optimizer.md>
Test Time Compute <./reference/test-time-compute.md>
Troubleshooting <./troubleshooting.md>
```

```{toctree}
:hidden:
:caption: Resources

FAQ <./resources/faq.md>
Code of Conduct <./resources/code-of-conduct.md>
Migration Guide <./resources/migration-guide.md>
Contributing <./resources/contributing.md>
Running Tests <./resources/running-tests.md>
./resources/running-ci-locally.md
./support.md
./resources/licensing.md
```
