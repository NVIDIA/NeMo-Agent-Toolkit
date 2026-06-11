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

# About Building NVIDIA NeMo Agent Toolkit Workflows

In NeMo Agent Toolkit, a workflow defines which [functions](./functions-and-function-groups/functions.md) and [models](./llms/index.md) are used to perform a given task or series of tasks. A workflow definition is specified in a [YAML configuration file](#understanding-the-workflow-configuration-file). The `workflow` section of the configuration file defines the workflow itself, and specifies a function, typically an [agent](../components/agents/index.md), which will orchestrate which functions and models are called to complete the given task.

## Workflow Overview

### Understanding the Workflow Configuration File

The workflow configuration file is a YAML file that specifies the [tools](./functions-and-function-groups/functions.md#agents-and-tools) and models to use in a workflow, along with general configuration settings. This section examines the configuration of the `examples/getting_started/simple_web_query` workflow to show how they are organized.

`examples/getting_started/simple_web_query/configs/config.yml`:

```yaml
functions:
  webpage_query:
    _type: webpage_query
    webpage_url: https://docs.smith.langchain.com
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
  current_datetime:
    _type: current_datetime

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0

embedders:
  nv-embedqa-e5-v5:
    _type: nim
    model_name: nvidia/nv-embedqa-e5-v5

workflow:
  _type: react_agent
  tool_names: [webpage_query, current_datetime]
  llm_name: nim_llm
  verbose: true
  parse_agent_response_max_retries: 3
```

This workflow configuration is divided into four sections: `functions`, `llms`, `embedders`, and `workflow`. The `functions` section contains the tools used in the workflow, while `llms` and `embedders` define the models used in the workflow, and lastly the `workflow` section ties the other sections together and defines the workflow itself.

#### The `functions` section

In this workflow, the `webpage_query` tool queries the LangSmith User Guide, and the `current_datetime` tool gets the current date and time. The `description` entry instructs the LLM when and how to use the tool. In this case, the workflow explicitly defines `description` for the `webpage_query` tool.

#### The `llms` and `embedders` sections

The `webpage_query` tool uses the `nv-embedqa-e5-v5` embedder, which is defined in the `embedders` section.

#### The `workflow` section

The workflow itself is typically an agent, however any NeMo Agent Toolkit function can be used as a workflow. Refer to the [Agents](../components/agents/index.md) documentation for more details on the agents that are included in NeMo Agent Toolkit.

For details on workflow configuration, including sections not utilized in the above example, refer to the [Workflow Configuration](./workflow-configuration.md) document.

## Key Concepts

Understanding these concepts will help you build workflows effectively.

### Workflow

A workflow defines which functions and models are used to perform a given task or series of tasks. The `workflow` section of the configuration file defines the workflow itself, and specifies a function, typically an agent, which will orchestrate which functions and models are called to complete the given task.

### Configuration File

A workflow definition is specified in a YAML configuration file. The file specifies the tools and models to use in a workflow, along with general configuration settings, organized into the `functions`, `llms`, `embedders`, and `workflow` sections.

### Functions (Tools)

The `functions` section contains the tools used in the workflow. The `description` entry instructs the LLM when and how to use the tool.

### Agents

The workflow itself is typically an agent, however any NeMo Agent Toolkit function can be used as a workflow. Refer to the [Agents](../components/agents/index.md) documentation for more details on the agents that are included in NeMo Agent Toolkit.

### Control Flow Components

Control flow components are offered by NeMo Agent Toolkit to direct how a workflow runs, including the [Router Agent](../components/agents/router-agent/index.md) and the [Sequential Executor](../components/agents/sequential-executor/index.md).

## Common Approaches

The following are [agents](../components/agents/index.md) offered by NeMo Agent Toolkit. Choose the approach that best fits your needs.

- [Automatic Memory Wrapper Agent](../components/agents/auto-memory-wrapper/index.md) — **Best for**: wrapping any agent to provide automatic memory capture and retrieval without requiring the LLM to invoke memory tools explicitly.
- [ReAct Agent](../components/agents/react-agent/index.md) — **Best for**: performing ReAct (Reasoning and Acting) reasoning between tool calls.
- [Reasoning Agent](../components/agents/reasoning-agent/index.md) — **Best for**: reasoning ahead of time through planning rather than between steps (requires an LLM that supports reasoning).
- [ReWOO Agent](../components/agents/rewoo-agent/index.md) — **Best for**: decoupling reasoning from observations to improve tool usage and token efficiency for reasoning tasks.
- [Responses API and Agent](../components/agents/responses-api-and-agent/index.md) — **Best for**: tool use with OpenAI's Responses API, including built-in tools, MCP remote tools, and toolkit tools.
- [Tool Calling Agent](../components/agents/tool-calling-agent/index.md) — **Best for**: directly invoking external tools based on structured function definitions (requires an LLM with tool-calling support).

## Decision Factors

The following are control flow components offered by NeMo Agent Toolkit. Use the following comparison to select the right component.

| Factor | [Router Agent](../components/agents/router-agent/index.md) | [Sequential Executor](../components/agents/sequential-executor/index.md) |
|--------|------------------|-------------------|
| **What it does** | Analyzes incoming requests and directs them to the most appropriate branch based on the request configuration. | Chains multiple functions together, where each function's output becomes the input for the next function. |
| **How it runs** | Pairs a single-pass architecture with intelligent request routing to analyze prompts and select one branch that best handles the request. | Creates a linear tool execution pipeline that executes functions in a predetermined sequence without requiring LLMs or agents for orchestration. |
| **Best For** | Scenarios where different types of requests need specialized handling. | Linear pipelines where each function feeds the next; supports better error handling and optional compatibility validation. |
