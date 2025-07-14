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

A configurable Tool Calling Agent. This agent leverages the AIQ toolkit plugin system and `WorkflowBuilder` to integrate pre-built and custom tools into the workflow. Key elements are summarized below:

## Key Features

- **Pre-built Tools:** Leverages core AIQ toolkit library agent and tools.
- **Tool Calling / Function calling Agent:** Leverages tool / function input schema to appropriately route to the correct tool
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ toolkit.

### Install this Workflow:

From the root directory of the AIQ toolkit library, run the following commands:

```bash
uv pip install -e .
```

The `code_generation` and `wiki_search` tools are part of the `aiqtoolkit[langchain]` package.  To install the package run the following command:
```bash
# local package install from source
uv pip install -e '.[langchain]'
```


### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```
---

## Run the Workflow

The Tool Calling Agent can be used as either a workflow or a function, and there's an example configuration that demonstrates both.
If you’re looking for an example workflow where the Tool Calling Agent runs as the main workflow, refer to [config.yml](configs/config.yml).
To see the Tool Calling Agent used as a function within a workflow, alongside the Reasoning Agent, refer to [config-reasoning.yml](configs/config-reasoning.yml).
This README primarily covers the former case, where the Tool Calling Agent functions as the main workflow, in config.yml.
For more details, refer to the [ReAct Agent documentation](../../../../docs/source/workflows/about/tool-calling-agent.md) and the [Reasoning Agent documentation](../../../../docs/source/workflows/about/react-agent.md)

Run the following command from the root of the AIQ toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file=examples/intermediate/agents/tool_calling/configs/config.yml --input "who was Djikstra?"
```

**Expected Output**

```console
$ aiq run --config_file=examples/intermediate/agents/tool_calling/configs/config.yml --input "who was Djikstra?"
2025-04-23 15:03:46,312 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (499.885559 ms). Ensure all imports are inside your registered functions.
2025-04-23 15:03:46,573 - aiq.cli.commands.start - INFO - Starting AIQ toolkit from config file: 'examples/intermediate/agents/tool_calling/configs/config.yml'
2025-04-23 15:03:46,581 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-23 15:03:46,613 - aiq.profiler.utils - WARNING - Discovered frameworks: {<LLMFrameworkEnum.LANGCHAIN: 'langchain'>} in function code_generation_tool by inspecting source. It is recommended and more reliable to instead add the used LLMFrameworkEnum types in the framework_wrappers argument when calling @register_function.
2025-04-23 15:03:46,614 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Initializing code generation tool
Getting tool LLM from config
2025-04-23 15:03:46,632 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Filling tool's prompt variable from config
2025-04-23 15:03:46,632 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Initialized code generation tool

Configuration Summary:
--------------------
Workflow Type: tool_calling_agent
Number of Functions: 3
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-04-23 15:03:47,601 - aiq.agent.tool_calling_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: who was Djikstra?
Agent's thoughts:
content='' additional_kwargs={'tool_calls': [{'id': 'chatcmpl-tool-25c373f4cc544ab995e2b424c30eb00a', 'type': 'function', 'function': {'name': 'wikipedia_search', 'arguments': '{"question": "Djikstra"}'}}]} response_metadata={'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'chatcmpl-tool-25c373f4cc544ab995e2b424c30eb00a', 'type': 'function', 'function': {'name': 'wikipedia_search', 'arguments': '{"question": "Djikstra"}'}}], 'token_usage': {'prompt_tokens': 451, 'total_tokens': 465, 'completion_tokens': 14}, 'finish_reason': 'tool_calls', 'model_name': 'meta/llama-3.1-70b-instruct'} id='run-f82d064d-422a-4241-9d95-e56dd76ed447-0' tool_calls=[{'name': 'wikipedia_search', 'args': {'question': 'Djikstra'}, 'id': 'chatcmpl-tool-25c373f4cc544ab995e2b424c30eb00a', 'type': 'tool_call'}] usage_metadata={'input_tokens': 451, 'output_tokens': 14, 'total_tokens': 465} role='assistant'
------------------------------
2025-04-23 15:03:51,894 - aiq.agent.tool_calling_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: ['wikipedia_search']
Tool's input: content='' additional_kwargs={'tool_calls': [{'id': 'chatcmpl-tool-25c373f4cc544ab995e2b424c30eb00a', 'type': 'function', 'function': {'name': 'wikipedia_search', 'arguments': '{"question": "Djikstra"}'}}]} response_metadata={'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'chatcmpl-tool-25c373f4cc544ab995e2b424c30eb00a', 'type': 'function', 'function': {'name': 'wikipedia_search', 'arguments': '{"question": "Djikstra"}'}}], 'token_usage': {'prompt_tokens': 451, 'total_tokens': 465, 'completion_tokens': 14}, 'finish_reason': 'tool_calls', 'model_name': 'meta/llama-3.1-70b-instruct'} id='run-f82d064d-422a-4241-9d95-e56dd76ed447-0' tool_calls=[{'name': 'wikipedia_search', 'args': {'question': 'Djikstra'}, 'id': 'chatcmpl-tool-25c373f4cc544ab995e2b424c30eb00a', 'type': 'tool_call'}] usage_metadata={'input_tokens': 451, 'output_tokens': 14, 'total_tokens': 465} role='assistant'
Tool's response:
<Document source="https://en.wikipedia.org/wiki/Edsger_W._Dijkstra" page=""/>
Edsger Wybe Dijkstra ( DYKE-strə; Dutch: [ˈɛtsxər ˈʋibə ˈdɛikstraː] ; 11 May 1930 – 6 August 2002) was a Dutch computer scientist, programmer, software engineer, mathematician, and science essayist.
Born in Rotterdam in the Netherlands, Dijkstra studied mathematics and physics and then theoretical physics at the University of Leiden. Adriaan van Wijngaarden offered him a job as the first computer programmer in the Netherlands at the Mathematical Centre in Amsterdam, where he worked from 1952 until 1962. He formulated and solved the shortest path problem in 1956, and in 1960 developed the first compiler for the programming language ALGOL 60 in conjunction with colleague Jaap A. Zonneveld. In 1962 he moved to Eindhoven, and later to Nuenen, where he became a professor in the Mathematics Department at the Technische Hogeschool Eindhoven. In the late 1960s he built the THE multiprogramming system, which influence...
------------------------------
2025-04-23 15:03:59,211 - aiq.agent.tool_calling_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: who was Djikstra?

<Document source="https://en.wikipedia.org/wiki/Edsger_W._Dijkstra" page=""/>
Edsger Wybe Dijkstra ( DYKE-strə; Dutch: [ˈɛtsxər ˈʋibə ˈdɛikstraː] ; 11 May 1930 – 6 August 2002) was a Dutch computer scientist, programmer, software engineer, mathematician, and science essayist.
Born in Rotterdam in the Netherlands, Dijkstra studied mathematics and physics and then theoretical physics at the University of Leiden. Adriaan van Wijngaarden offered him a job as the first computer programmer in the Netherlands at the Mathematical Centre in Amsterdam, where he worked from 1952 until 1962. He formulated and solved the shortest path problem in 1956, and in 1960 developed the first compiler for the programming language ALGOL 60 in conjunction with colleague Jaap A. Zonneveld. In 1962 he moved to Eindhoven, and later to Nuenen, where he became a professor in the Mathematics Department at the Technische Hogeschool Eindhoven. In the late 1960s he built the THE multiprogramming system, which influence...
Agent's thoughts:
content='Edsger Wybe Dijkstra was a Dutch computer scientist, programmer, software engineer, mathematician, and science essayist. He was born on May 11, 1930, in Rotterdam, Netherlands, and studied mathematics and physics at the University of Leiden. Dijkstra worked as the first computer programmer in the Netherlands at the Mathematical Centre in Amsterdam from 1952 to 1962. He formulated and solved the shortest path problem in 1956 and developed the first compiler for the programming language ALGOL 60 in 1960. Dijkstra moved to Eindhoven in 1962 and became a professor in the Mathematics Department at the Technische Hogeschool Eindhoven. He built the THE multiprogramming system in the late 1960s, which influenced the development of operating systems.' additional_kwargs={} response_metadata={'role': 'assistant', 'content': 'Edsger Wybe Dijkstra was a Dutch computer scientist, programmer, software engineer, mathematician, and science essayist. He was born on May 11, 1930, in Rotterdam, Netherlands, and studied mathematics and physics at the University of Leiden. Dijkstra worked as the first computer programmer in the Netherlands at the Mathematical Centre in Amsterdam from 1952 to 1962. He formulated and solved the shortest path problem in 1956 and developed the first compiler for the programming language ALGOL 60 in 1960. Dijkstra moved to Eindhoven in 1962 and became a professor in the Mathematics Department at the Technische Hogeschool Eindhoven. He built the THE multiprogramming system in the late 1960s, which influenced the development of operating systems.', 'token_usage': {'prompt_tokens': 747, 'total_tokens': 911, 'completion_tokens': 164}, 'finish_reason': 'stop', 'model_name': 'meta/llama-3.1-70b-instruct'} id='run-51b93700-cd12-40dc-9ee4-632bf30b4a5e-0' usage_metadata={'input_tokens': 747, 'output_tokens': 164, 'total_tokens': 911} role='assistant'
------------------------------
2025-04-23 15:03:59,215 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['Edsger Wybe Dijkstra was a Dutch computer scientist, programmer, software engineer, mathematician, and science essayist. He was born on May 11, 1930, in Rotterdam, Netherlands, and studied mathematics and physics at the University of Leiden. Dijkstra worked as the first computer programmer in the Netherlands at the Mathematical Centre in Amsterdam from 1952 to 1962. He formulated and solved the shortest path problem in 1956 and developed the first compiler for the programming language ALGOL 60 in 1960. Dijkstra moved to Eindhoven in 1962 and became a professor in the Mathematics Department at the Technische Hogeschool Eindhoven. He built the THE multiprogramming system in the late 1960s, which influenced the development of operating systems.']
--------------------------------------------------
```
---

### Starting the AIQ Toolkit Server

You can start the AIQ toolkit server using the `aiq serve` command with the appropriate configuration file.

**Starting the Tool Calling Agent Example Workflow**

```bash
aiq serve --config_file=examples/intermediate/agents/tool_calling/configs/config.yml
```

### Making Requests to the AIQ Toolkit Server

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
aiq eval --config_file=examples/intermediate/agents/tool_calling/configs/config.yml
```
---
