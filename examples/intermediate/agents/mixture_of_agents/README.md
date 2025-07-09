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

An example of a Mixture of Agents (naive Mixture of Experts / naive Agent Hypervisor). This agent leverages the AIQ toolkit plugin system and `WorkflowBuilder` to integrate pre-built and custom tools into the workflows, and workflows as tools. Key elements are summarized below:

## Key Features

- **Pre-built Tools and Agents:** Leverages core AIQ toolkit library agents and tools.
- **ReAct Agent:** Performs reasoning between agent / tool call; utilizes agent / tool names and descriptions to appropriately route to the correct agent or tool.
- **Tool Calling Agent** The "Expert Agents" are Tool Calling Agents.  They leverages tool / function input schema to appropriately route to the correct tool.
- **Custom Plugin System:** Developers can bring in new agents and tools using plugins.
- **High-level API:** Enables defining functions that transform agents and tools into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity. Customize agents, agent's tools, prompts, and more.
- **Ease of Use:** Simplifies developer experience and deployment.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ toolkit.

### Install this Workflow:

From the root directory of the AIQ toolkit repository, run the following commands:

```bash
uv pip install -e .
```

The `code_generation` and `wiki_search` tools are part of the `aiqtoolkit[langchain]` package.  To install the package run the following command:
```bash
# local package install from source
uv pip install -e '.[langchain]'
```

In addition to this the example utilizes some tools from the `examples/basic/functions/simple_calculator` example.  To install the package run the following command:
```bash
uv pip install -e examples/basic/functions/simple_calculator
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:
```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Run the Workflow

Run the following command from the root of the AIQ toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file=examples/intermediate/agents/mixture_of_agents/configs/config.yml --input "who was Djikstra?"
```

**Expected Output**

```console
$ aiq run --config_file=examples/agents/mixture_of_agents/configs/config.yml --input "who was Djikstra?"
2025-04-23 14:57:12,020 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (503.239393 ms). Ensure all imports are inside your registered functions.
2025-04-23 14:57:12,284 - aiq.cli.commands.start - INFO - Starting AIQ toolkit from config file: 'examples/agents/mixture_of_agents/configs/config.yml'
2025-04-23 14:57:12,293 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-23 14:57:12,375 - aiq.profiler.utils - WARNING - Discovered frameworks: {<LLMFrameworkEnum.LANGCHAIN: 'langchain'>} in function code_generation_tool by inspecting source. It is recommended and more reliable to instead add the used LLMFrameworkEnum types in the framework_wrappers argument when calling @register_function.
2025-04-23 14:57:12,375 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Initializing code generation tool
Getting tool LLM from config
2025-04-23 14:57:12,376 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Filling tool's prompt variable from config
2025-04-23 14:57:12,376 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Initialized code generation tool

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 8
Number of LLMs: 2
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-04-23 14:57:14,060 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: who was Djikstra?
Agent's thoughts:
Thought: I should search the internet for information on Djikstra.
Action: internet_agent
Action Input: {'input_message': 'Djikstra'}
Observation
------------------------------
2025-04-23 14:57:20,638 - aiq.agent.tool_calling_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Djikstra
Agent's thoughts:
content="Dijkstra's algorithm is a well-known algorithm in graph theory, named after the Dutch computer scientist Edsger W. Dijkstra. It is used to find the shortest path between two nodes in a graph. The algorithm works by maintaining a list of unvisited nodes and iteratively selecting the node with the shortest distance from the starting node. The distance to each node is updated as the algorithm progresses, and the node with the shortest distance is added to the list of visited nodes. The algorithm terminates when the destination node is reached, and the shortest path is constructed by tracing back the nodes from the destination to the starting node.\n\nDijkstra's algorithm has many applications in computer science and other fields, such as network routing, traffic optimization, and resource allocation. It is also used in many real-world problems, such as finding the shortest path between two cities, optimizing traffic flow, and scheduling tasks.\n\nThe algorithm has a time complexity of O(|E| + |V|log|V|) in the worst case, where |E| is the number of edges and |V| is the number of vertices in the graph. This makes it efficient for large graphs. However, it can be slow for very large graphs or graphs with a large number" additional_kwargs={} response_metadata={'role': 'assistant', 'content': "Dijkstra's algorithm is a well-known algorithm in graph theory, named after the Dutch computer scientist Edsger W. Dijkstra. It is used to find the shortest path between two nodes in a graph. The algorithm works by maintaining a list of unvisited nodes and iteratively selecting the node with the shortest distance from the starting node. The distance to each node is updated as the algorithm progresses, and the node with the shortest distance is added to the list of visited nodes. The algorithm terminates when the destination node is reached, and the shortest path is constructed by tracing back the nodes from the destination to the starting node.\n\nDijkstra's algorithm has many applications in computer science and other fields, such as network routing, traffic optimization, and resource allocation. It is also used in many real-world problems, such as finding the shortest path between two cities, optimizing traffic flow, and scheduling tasks.\n\nThe algorithm has a time complexity of O(|E| + |V|log|V|) in the worst case, where |E| is the number of edges and |V| is the number of vertices in the graph. This makes it efficient for large graphs. However, it can be slow for very large graphs or graphs with a large number", 'token_usage': {'prompt_tokens': 363, 'total_tokens': 613, 'completion_tokens': 250}, 'finish_reason': 'length', 'model_name': 'meta/llama-3.3-70b-instruct'} id='run-44bec667-41ec-43a8-bbe2-ecacfe0580e8-0' usage_metadata={'input_tokens': 363, 'output_tokens': 250, 'total_tokens': 613} role='assistant'
------------------------------
2025-04-23 14:57:20,641 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: internet_agent
Tool's input: {"input_message": "Djikstra"}
Tool's response:
Dijkstra's algorithm is a well-known algorithm in graph theory, named after the Dutch computer scientist Edsger W. Dijkstra. It is used to find the shortest path between two nodes in a graph. The algorithm works by maintaining a list of unvisited nodes and iteratively selecting the node with the shortest distance from the starting node. The distance to each node is updated as the algorithm progresses, and the node with the shortest distance is added to the list of visited nodes. The algorithm terminates when the destination node is reached, and the shortest path is constructed by tracing back the nodes from the destination to the starting node.

Dijkstra's algorithm has many applications in computer science and other fields, such as network routing, traffic optimization, and resource allocation. It is also used in many real-world problems, such as finding the shortest path between two cities, optimizing traffic flow, and scheduling tasks.

The algorithm has a time complexity of O(|E| +...
------------------------------
2025-04-23 14:57:22,680 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: who was Djikstra?
Agent's thoughts:
Thought: I now know the final answer
Final Answer: Edsger W. Dijkstra was a Dutch computer scientist, and Dijkstra's algorithm is a well-known algorithm in graph theory used to find the shortest path between two nodes in a graph.
------------------------------
2025-04-23 14:57:22,684 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
["Edsger W. Dijkstra was a Dutch computer scientist, and Dijkstra's algorithm is a well-known algorithm in graph theory used to find the shortest path between two nodes in a graph."]
--------------------------------------------------
```
---

### Starting the AIQ Toolkit Server

You can start the AIQ toolkit server using the `aiq serve` command with the appropriate configuration file.

**Starting the Mixture of Agents Example Workflow**

```bash
aiq serve --config_file=examples/intermediate/agents/mixture_of_agents/configs/config.yml
```

### Making Requests to the AIQ Toolkit Server

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
