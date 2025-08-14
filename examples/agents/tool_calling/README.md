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

# Tool Calling Agent

A configurable Tool Calling agent. This agent leverages the NeMo Agent toolkit plugin system and `WorkflowBuilder` to integrate pre-built and custom tools into the workflow. Key elements are summarized below:

## Table of Contents

- [Key Features](#key-features)
- [Graph Structure](#graph-structure)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Run the Workflow](#run-the-workflow)
  - [Starting the NeMo Agent Toolkit Server](#starting-the-nemo-agent-toolkit-server)
  - [Making Requests to the NeMo Agent Toolkit Server](#making-requests-to-the-nemo-agent-toolkit-server)
  - [Evaluating the Tool Calling Agent Workflow](#evaluating-the-tool-calling-agent-workflow)

## Key Features

- **Tool Calling Agent Framework:** Demonstrates a `tool_calling_agent` that leverages tool or function input schemas to make precise tool selections and structured function calls.
- **Wikipedia Search Integration:** Shows integration with the `wikipedia_search` tool for retrieving factual information from Wikipedia sources.
- **Code Generation Capabilities:** Includes the `code_generation_tool` for generating code examples and technical content.
- **Schema-Driven Tool Selection:** Uses structured input schemas to appropriately route to the correct tool, providing more deterministic tool calling compared to name or description-based routing.
- **Dual-Node Graph Architecture:** Implements the same operational pattern as other NeMo Agent toolkit agents, alternating between reasoning and tool execution while using schema-based tool selection.

## Graph Structure

The Tool Calling agent uses the same dual-node graph architecture as other agents in the NeMo Agent toolkit, alternating between reasoning and tool execution. The following diagram illustrates the agent's workflow:

<div align="center">
<img src="../../../docs/source/_static/dual_node_agent.png" alt="Tool Calling Agent Graph Structure" width="400" style="max-width: 100%; height: auto;">
</div>

**Workflow Overview:**
- **Start**: The agent begins processing with user input
- **Agent Node**: Leverages tool or function input schemas to decide which tool to call or provide a final answer
- **Conditional Edge**: Routes the flow based on the agent's decision
- **Tool Node**: Executes the selected tool using structured input schemas
- **Cycle**: The agent can loop between reasoning and tool execution until it reaches a final answer

This architecture enables the Tool Calling agent to make precise tool selections based on input schemas while maintaining the same operational pattern as other agents in the NeMo Agent Toolkit.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install this Workflow

From the root directory of the NeMo Agent Toolkit library, run the following commands:

```bash
uv pip install -e .
```

The `code_generation` and `wiki_search` tools are part of the `nvidia-nat[langchain]` package.  To install the package run the following command:
```bash
# local package install from source
uv pip install -e '.[langchain]'
```


### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```
---

## Run the Workflow

The Tool Calling Agent can be used as either a workflow or a function, and there's an example configuration that demonstrates both.
If you’re looking for an example workflow where the Tool Calling Agent runs as the main workflow, refer to [config.yml](configs/config.yml).
To see the Tool Calling Agent used as a function within a workflow, alongside the Reasoning Agent, refer to [config-reasoning.yml](configs/config-reasoning.yml).
This README primarily covers the former case, where the Tool Calling Agent functions as the main workflow, in config.yml.
For more details, refer to the [ReAct Agent documentation](../../../docs/source/workflows/about/tool-calling-agent.md) and the [Reasoning Agent documentation](../../../docs/source/workflows/about/react-agent.md)

Run the following command from the root of the NeMo Agent Toolkit repo to execute this workflow with the specified input:

```bash
nat run --config_file=examples/agents/tool_calling/configs/config.yml --input "who was Djikstra?"
```

**Expected Workflow Output**

> [!NOTE]
> The output from `wikipedia_search` tool may contain odd formatting (extra newlines, additional indentation), especially when a Wikipedia page contains formula or other complex content. This is expected due to the upstream behavior of the `wikipedia` python package.

```console
<snipped for brevity>

[AGENT]
Calling tools: ['wikipedia_search']
Tool's input: content='' additional_kwargs={'tool_calls': [{'id': 'chatcmpl-tool-25c373f4cc544ab995e2b424c30eb00a', 'type': 'function', 'function': {'name': 'wikipedia_search', 'arguments': '{"question": "Djikstra"}'}}]} response_metadata={'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'chatcmpl-tool-25c373f4cc544ab995e2b424c30eb00a', 'type': 'function', 'function': {'name': 'wikipedia_search', 'arguments': '{"question": "Djikstra"}'}}], 'token_usage': {'prompt_tokens': 451, 'total_tokens': 465, 'completion_tokens': 14}, 'finish_reason': 'tool_calls', 'model_name': 'meta/llama-3.1-70b-instruct'} id='run-f82d064d-422a-4241-9d95-e56dd76ed447-0' tool_calls=[{'name': 'wikipedia_search', 'args': {'question': 'Djikstra'}, 'id': 'chatcmpl-tool-25c373f4cc544ab995e2b424c30eb00a', 'type': 'tool_call'}] usage_metadata={'input_tokens': 451, 'output_tokens': 14, 'total_tokens': 465} role='assistant'
Tool's response:
<Document source="https://en.wikipedia.org/wiki/Edsger_W._Dijkstra" page=""/>
Edsger Wybe Dijkstra ( DYKE-strə; Dutch: [ˈɛtsxər ˈʋibə ˈdɛikstraː] ; 11 May 1930 – 6 August 2002) was a Dutch computer scientist, programmer, software engineer, mathematician, and science essayist.
Born in Rotterdam in the Netherlands, Dijkstra studied mathematics and physics and then theoretical physics at the University of Leiden. Adriaan van Wijngaarden offered him a job as the first computer programmer in the Netherlands at the Mathematical Centre in Amsterdam, where he worked from 1952 until 1962. He formulated and solved the shortest path problem in 1956, and in 1960 developed the first compiler for the programming language ALGOL 60 in conjunction with colleague Jaap A. Zonneveld. In 1962 he moved to Eindhoven, and later to Nuenen, where he became a professor in the Mathematics Department at the Technische Hogeschool Eindhoven. In the late 1960s he built the THE multiprogramming system, which influence...
------------------------------
2025-04-23 15:03:59,211 - nat.agent.tool_calling_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: who was Djikstra?

<Document source="https://en.wikipedia.org/wiki/Edsger_W._Dijkstra" page=""/>
Edsger Wybe Dijkstra ( DYKE-strə; Dutch: [ˈɛtsxər ˈʋibə ˈdɛikstraː] ; 11 May 1930 – 6 August 2002) was a Dutch computer scientist, programmer, software engineer, mathematician, and science essayist.
Born in Rotterdam in the Netherlands, Dijkstra studied mathematics and physics and then theoretical physics at the University of Leiden. Adriaan van Wijngaarden offered him a job as the first computer programmer in the Netherlands at the Mathematical Centre in Amsterdam, where he worked from 1952 until 1962. He formulated and solved the shortest path problem in 1956, and in 1960 developed the first compiler for the programming language ALGOL 60 in conjunction with colleague Jaap A. Zonneveld. In 1962 he moved to Eindhoven, and later to Nuenen, where he became a professor in the Mathematics Department at the Technische Hogeschool Eindhoven. In the late 1960s he built the THE multiprogramming system, which influence...

<snipped for brevity>

--------------------------------------------------
Workflow Result:
['Edsger Wybe Dijkstra was a Dutch computer scientist, programmer, software engineer, mathematician, and science essayist. He was born on May 11, 1930, in Rotterdam, Netherlands, and studied mathematics and physics at the University of Leiden. Dijkstra worked as the first computer programmer in the Netherlands at the Mathematical Centre in Amsterdam from 1952 to 1962. He formulated and solved the shortest path problem in 1956 and developed the first compiler for the programming language ALGOL 60 in 1960. Dijkstra moved to Eindhoven in 1962 and became a professor in the Mathematics Department at the Technische Hogeschool Eindhoven. He built the THE multiprogramming system in the late 1960s, which influenced the development of operating systems.']
```
---

### Starting the NeMo Agent Toolkit Server

You can start the NeMo Agent toolkit server using the `nat serve` command with the appropriate configuration file.

**Starting the Tool Calling Agent Example Workflow**

```bash
nat serve --config_file=examples/agents/tool_calling/configs/config.yml
```

### Making Requests to the NeMo Agent Toolkit Server

Once the server is running, you can make HTTP requests to interact with the workflow.

#### Non-Streaming Requests

**Non-Streaming Request to the Tool Calling Agent Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "What are LLMs?"}'
```

#### Streaming Requests

**Streaming Request to the Tool Calling Agent Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate/stream \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "What are LLMs?"}'
```
---
### Evaluating the Tool Calling Agent Workflow
**Run and evaluate the `tool_calling_agent` example Workflow**

```bash
nat eval --config_file=examples/agents/tool_calling/configs/config.yml
```
