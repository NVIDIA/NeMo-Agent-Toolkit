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

# Mixture of Agents Example

An example of a Mixture of Agents (naive Mixture of Experts / naive Agent Hypervisor). This agent leverages the AgentIQ plugin system and `WorkflowBuilder` to integrate pre-built and custom tools into the workflows, and workflows as tools. Key elements are summarized below:

## Key Features

- **Pre-built Tools and Agents:** Leverages core AgentIQ library agents and tools.
- **ReAct Agent:** Performs reasoning between agent / tool call; utilizes agent / tool names and descriptions to appropriately route to the correct agent or tool.
- **Tool Calling Agent** The "Expert Agents" are Tool Calling Agents.  They leverages tool / function input schema to appropriately route to the correct tool.
- **Custom Plugin System:** Developers can bring in new agents and tools using plugins.
- **High-level API:** Enables defining functions that transform agents and tools into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity. Customize agents, agent's tools, prompts, and more.
- **Ease of Use:** Simplifies developer experience and deployment.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/intro/install.md) to create the development environment and install AgentIQ.

### Install this Workflow:

From the root directory of the AgentIQ repository, run the following commands:

```bash
uv pip install -e .
```

The `code_generation` and `wiki_search` tools are part of the `agentiq[langchain]` package.  To install the package run the following command:
```bash
# local package install from source
uv pip install -e '.[langchain]'
```


### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/intro/get-started.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:
```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Run the Workflow

Run the following command from the root of the AgentIQ repo to execute this workflow with the specified input:

```bash
aiq run --config_file=examples/agents/mixture_of_agents/configs/config.yml --input "who was Djikstra?"
```

**Expected Output**

```console
$ aiq run --config_file=examples/agents/mixture_of_agents/configs/config.yml --input "who was Djikstra?"
2025-02-11 17:50:38,377 - aiq.cli.commands.run - INFO - Loading configuration from: examples/agents/mixture_of_agents/configs/config.yml
2025-02-11 17:50:40,013 - aiq.profiler.callbacks.crewai_callback_handler - ERROR - Failed to import crewAI or a sub-module: No module named 'litellm'
/Users/sjaviya/Documents/AgentIQ/sjaviya/ai-query-engine/.tempvenv/lib/python3.12/site-packages/langchain_nvidia_ai_endpoints/chat_models.py:591: UserWarning: Model 'meta/llama-3.3-70b-instruct' is not known to support tools. Your tool binding may fail at inference time.
  warnings.warn(
2025-02-11 17:50:41,337 - aiq.agent.tool_calling_agent.agent - INFO - Initialized Tool Calling Agent Graph
2025-02-11 17:50:41,337 - aiq.agent.tool_calling_agent.agent - INFO - Tool Calling Agent Graph built and compiled successfully
/Users/sjaviya/Documents/AgentIQ/sjaviya/ai-query-engine/.tempvenv/lib/python3.12/site-packages/langchain_nvidia_ai_endpoints/chat_models.py:591: UserWarning: Model 'meta/llama-3.3-70b-instruct' is not known to support tools. Your tool binding may fail at inference time.
  warnings.warn(
2025-02-11 17:50:41,560 - aiq.agent.tool_calling_agent.agent - INFO - Initialized Tool Calling Agent Graph
2025-02-11 17:50:41,561 - aiq.agent.tool_calling_agent.agent - INFO - Tool Calling Agent Graph built and compiled successfully
2025-02-11 17:50:41,561 - aiq.tool.code_generation_tool - INFO - Initializing code generation tool
Getting tool LLM from config
2025-02-11 17:50:41,746 - aiq.tool.code_generation_tool - INFO - Filling tool's prompt variable from config
2025-02-11 17:50:41,746 - aiq.tool.code_generation_tool - INFO - Initialized code generation tool
2025-02-11 17:50:42,182 - aiq.agent.react_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
2025-02-11 17:50:42,182 - aiq.agent.react_agent.agent - INFO - Initialized ReAct Agent Graph
2025-02-11 17:50:42,183 - aiq.agent.react_agent.agent - INFO - ReAct Graph built and compiled successfully

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 9
Number of LLMs: 2
Number of Embedders: 0
Number of Memory: 0

2025-02-11 17:50:42,185 - aiq.cli.commands.run - INFO - Processing input: ('who was Djikstra?',)
2025-02-11 17:50:42,187 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-02-11 17:50:43,906 - aiq.agent.react_agent.agent - INFO - The user's question was: who was Djikstra?
2025-02-11 17:50:43,906 - aiq.agent.react_agent.agent - INFO - The agent's thoughts are:
Thought: I need to find information about a person named Djikstra, I think he might be related to computer science or mathematics.

Action: internet_agent
Action Input: Djikstra


2025-02-11 17:50:43,909 - aiq.agent.react_agent.agent - INFO - Calling tool internet_agent with input: Djikstra

2025-02-11 17:50:43,911 - aiq.agent.tool_calling_agent.agent - INFO - Calling agent
2025-02-11 17:50:50,325 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-02-11 17:50:52,418 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: I now know the final answer
Final Answer: Edsger W. Dijkstra was a computer scientist who created Dijkstra's algorithm, a well-known algorithm in graph theory used to find the shortest path between two nodes in a graph.
2025-02-11 17:50:52,421 - aiq.cli.commands.run - INFO - --------------------------------------------------
Workflow Result:
["Edsger W. Dijkstra was a computer scientist who created Dijkstra's algorithm, a well-known algorithm in graph theory used to find the shortest path between two nodes in a graph."]
--------------------------------------------------
Cleaning up react_agent workflow.
Cleaning up react_agent workflow.
Cleaning up react_agent workflow.
2025-02-11 17:50:52,428 - aiq.cli.entrypoint - INFO - Total time: 14.10 sec
2025-02-11 17:50:52,428 - aiq.cli.entrypoint - INFO - Pipeline runtime: 10.24 sec
```
---

### Starting the AgentIQ Server

You can start the AgentIQ server using the `aiq serve` command with the appropriate configuration file.

**Starting the Mixture of Agents Example Workflow**

```bash
aiq serve --config_file=examples/agents/mixture_of_agents/configs/config.yml
```

### Making Requests to the AgentIQ Server

Once the server is running, you can make HTTP requests to interact with the workflow.

#### Non-Streaming Requests

**Non-Streaming Request to the Mixture of Agents Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "What are LLMs?"}'
```

#### Streaming Requests

**Streaming Request to the Mixture of Agents Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate/stream \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "What are LLMs?"}'
```
---
