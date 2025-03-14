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

# ReAct Agent

A configurable ReAct Agent. This agent leverages the AgentIQ plugin system and `WorkflowBuilder` to integrate pre-built and custom tools into the workflow. Key elements are summarized below:

## Key Features

- **Pre-built Tools:** Leverages core AgentIQ library agent and tools.
- **ReAct Agent:** Performs reasoning between tool call; utilizes tool names and descriptions to appropriately route to the correct tool
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.

## Installation and Setup

### Setup Virtual Environment and Install AgentIQ

From the root directory of the AgentIQ library, run the following commands. Skip this if you've already installed AgentIQ.

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install uv
uv sync --all-extras
```

### Install this Workflow:

From the root directory of the AgentIQ library, run the following commands:

```bash
uv pip install -e .
```

The `code_generation` and `wiki_search` tools are part of the `agentiq[langchain]` package.  To install the package run the following command:
```bash
pip install agentiq[langchain]
```
Alternatively, to install the `agentiq[langchain]` package from source run:
```bash
# local package install from source
uv pip install -e '.[langchain]'
```

### Set Up API Keys

You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```
---

## Run the Workflow

The ReAct Agent can be used as either a workflow or a function, and there's an example configuration that demonstrates both.
If you’re looking for an example workflow where the ReAct Agent runs as the main workflow, refer to [config.yml](configs/config.yml).
To see the ReAct Agent used as a function within a workflow, alongside the Reasoning Agent, refer to [config-reasoning.yml](configs/config-reasoning.yml).
This README primarily covers the former case, where the ReAct Agent functions as the main workflow, in config.yml.
For more details, refer to the [ReAct Agent documentation](../../../docs/source/components/react-agent.md) and the [Reasoning Agent documentation](../../../docs/source/components/react-agent.md)

Run the following command from the root of the AgentIQ repo to execute this workflow with the specified input:

```bash
aiq run  --config_file=examples/agents/react/configs/config.yml --input "who was Djikstra?"
```

**Expected Output**

```console
$ aiq run  --config_file=examples/agents/react/configs/config.yml --input "who was Djikstra?"
2025-02-07 15:53:11,036 - aiq.cli.run - INFO - Loading configuration from: examples/agents/react/configs/config.yml
2025-02-07 15:53:21,508 - aiq.profiler.callbacks.crewai_callback_handler - ERROR - Failed to import crewAI or a sub-module: No module named 'litellm'
2025-02-07 15:53:25,917 - aiq.profiler.callbacks.crewai_callback_handler - ERROR - Failed to import crewAI or a sub-module: No module named 'litellm'
2025-02-07 15:53:26,290 - aiq.profiler.callbacks.crewai_callback_handler - ERROR - Failed to import crewAI or a sub-module: No module named 'litellm'
2025-02-07 15:53:26,291 - aiq.profiler.callbacks.crewai_callback_handler - ERROR - Failed to import crewAI or a sub-module: No module named 'litellm'
2025-02-07 15:53:26,291 - aiq.tool.code_generation_tool - INFO - Initializing code generation tool
Getting tool LLM from config
2025-02-07 15:53:26,578 - aiq.tool.code_generation_tool - INFO - Filling tool's prompt variable from config
2025-02-07 15:53:26,578 - aiq.tool.code_generation_tool - INFO - Initialized code generation tool
2025-02-07 15:53:26,580 - aiq.profiler.callbacks.crewai_callback_handler - ERROR - Failed to import crewAI or a sub-module: No module named 'litellm'
2025-02-07 15:53:28,097 - aiq.agent.base - INFO - Initializing Agent Graph
2025-02-07 15:53:28,097 - aiq.workflows.react_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
2025-02-07 15:53:28,097 - aiq.workflows.react_agent.agent - INFO - Initialized ReAct Agent Graph
2025-02-07 15:53:28,097 - aiq.agent.base - INFO - Building and compiling the Agent Graph
2025-02-07 15:53:28,100 - aiq.workflows.react_agent.agent - INFO - ReAct Graph built and compiled successfully

Configuration Summary:
--------------------
Workflow Type: EmptyFunctionConfig
Number of Functions: 0
Number of LLMs: 0
Number of Embedders: 0
Number of Memory: 0

2025-02-07 15:53:28,103 - aiq.cli.run - INFO - Processing input: ('who was Djikstra?',)
2025-02-07 15:53:28,106 - aiq.workflows.react_agent.agent - INFO - Querying agent, attempt: 1
2025-02-07 15:53:28,962 - aiq.workflows.react_agent.agent - INFO - The user's question was: who was Djikstra?
2025-02-07 15:53:28,962 - aiq.workflows.react_agent.agent - INFO - The agent's thoughts are:
Thought: To answer this question, I need to find information about Djikstra.

Action: wikipedia_search
Action Input: Djikstra


2025-02-07 15:53:28,968 - aiq.workflows.react_agent.agent - INFO - Calling tool wikipedia_search with input: Djikstra

2025-02-07 15:53:31,700 - aiq.workflows.react_agent.agent - INFO - Querying agent, attempt: 1
2025-02-07 15:53:33,313 - aiq.workflows.react_agent.agent - INFO -

The agent's thoughts are:
Thought: I now know the final answer

Final Answer: Djikstra was a Dutch computer scientist, programmer, software engineer, mathematician, and science essayist. He is best known for his work on the shortest path problem and the development of Dijkstra's algorithm, which is used to find the shortest path between nodes in a weighted graph.
2025-02-07 15:53:33,319 - aiq.cli.run - INFO - --------------------------------------------------
Workflow Result:
["Djikstra was a Dutch computer scientist, programmer, software engineer, mathematician, and science essayist. He is best known for his work on the shortest path problem and the development of Dijkstra's algorithm, which is used to find the shortest path between nodes in a weighted graph."]
--------------------------------------------------
Cleaning up react_agent workflow.
2025-02-07 15:53:33,334 - aiq.cli.entrypoint - INFO - Total time: 22.42 sec
2025-02-07 15:53:33,334 - aiq.cli.entrypoint - INFO - Pipeline runtime: 5.23 sec
```
---

### Starting the AgentIQ Server

You can start the AgentIQ server using the `aiq serve` command with the appropriate configuration file.

**Starting the ReAct Agent Example Workflow**

```bash
aiq serve --config_file=examples/agents/react/configs/config.yml
```

### Making Requests to the AgentIQ Server

Once the server is running, you can make HTTP requests to interact with the workflow.

#### Non-Streaming Requests

**Non-Streaming Request to the ReAct Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "What are LLMs?"}'
```

#### Streaming Requests

**Streaming Request to the ReAct Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate/stream \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "What are LLMs?"}'
```
---
### Evaluating the ReAct Agent Workflow
**Run and evaluate the `react_agent` example Workflow**

```bash
aiq eval --config_file=examples/agents/react/configs/config.yml
```
---
