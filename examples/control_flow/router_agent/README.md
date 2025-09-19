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

# Router Agent Example

This example demonstrates how to use a configurable Router Agent with the NeMo Agent toolkit. The Router Agent analyzes incoming requests and directly routes them to the most appropriate branch (other agents, functions or tools) based on the request content. For this purpose, NeMo Agent toolkit provides a [`router_agent`](../../../docs/source/workflows/about/router-agent.md) workflow type.

## Table of Contents

- [Key Features](#key-features)
- [Graph Structure](#graph-structure)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Run the Workflow](#run-the-workflow)
  - [Starting the NeMo Agent Toolkit Server](#starting-the-nemo-agent-toolkit-server)
  - [Making Requests to the NeMo Agent Toolkit Server](#making-requests-to-the-nemo-agent-toolkit-server)

## Key Features

- **Single-Pass Graph Structure:** Uses a single-pass architecture with Router Agent Node (analyzes request and selects branch) and Branch Node (executes the selected branch).
- **Intelligent Request Routing:** Shows how the Router Agent analyzes user input and selects exactly one branch that best handles the request, making it ideal for scenarios when a graph of agents and tools is needed to handle different types of requests.
- **Easy Fine-tuning:** The single pass approach of the Router Agent makes it easy to fine-tune the routing logic by customizing the prompt and the branches.

## Graph Structure

The Router Agent uses a single-pass graph architecture that efficiently analyzes requests and routes them to appropriate branches. The following describes the agent's workflow:

<div align="center">
<img src="../../../docs/source/_static/router_agent.png" alt="Router Agent Graph Structure" width="400" style="max-width: 100%; height: auto;">
</div>

## Configuration

The Router Agent is configured through the `config.yml` file. The following configuration options are available:

### Required Configuration Options

- **`_type`**: Set to `router_agent` to use the Router Agent workflow type
- **`branches`**: List of available branches that the agent can route requests to
- **`llm_name`**: The language model used for request analysis and routing decisions

### Optional Configuration Options

- **`description`**: Description of the workflow (default: "Router Agent Workflow")
- **`system_prompt`**: Custom system prompt to use with the agent (default: uses built-in prompt)
- **`user_prompt`**: Custom user prompt to use with the agent (default: uses built-in prompt)
- **`max_router_retries`**: Maximum number of retries if the router agent fails to choose a branch (default: 3)
- **`detailed_logs`**: Enable detailed logging to see the routing decisions and responses (default: false)
- **`log_response_max_chars`**: Maximum number of characters to display in logs when logging branch responses (default: 1000)

Note on custom prompts:
  - `{branches}` and `{branch_names}` must be included in your customized `system_prompt`.
  - `{chat_history}` and `{request}` must be included in your customized `user_prompt`.
  - Instruct the model to choose exactly one branch and return only its name.

### Example Configuration

**Basic Configuration:**
```yaml
workflow:
  _type: router_agent
  branches: [fruit_advisor, city_advisor, literature_advisor]
  llm_name: nim_llm
  detailed_logs: true
```

**Configuration with Custom Options:**
```yaml
workflow:
  _type: router_agent
  branches: [fruit_advisor, city_advisor, literature_advisor]
  llm_name: nim_llm
  description: "Multi-domain Advisor Router"
  max_router_retries: 5
  detailed_logs: true
  log_response_max_chars: 2000
  system_prompt: "You are an intelligent routing agent that analyzes user requests and selects the most appropriate
  advisor from {branches}.
  You MUST choose exactly one branch and return only its name which is one of the following: {branch_names}."
  user_prompt: "Considering the conversation so far: {chat_history} Routing request: {request}"
```

The agent will automatically analyze incoming requests and route them to the most appropriate branch based on the request content and the descriptions of available branches.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e examples/control_flow/router_agent
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Run the Workflow

This workflow showcases the Router Agent's ability to route requests to the most appropriate branch based on the request content. To simplify the example, we use mock advisor functions that return a static response based on the input, but you can imagine these advisors as real agents that would intelligently analyze the request and return a response.

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
nat run --config_file=examples/control_flow/router_agent/configs/config.yml --input "I want a yellow fruit"
```

**Additional Example Commands:**
```bash
# Test fruit advisor
nat run --config_file=examples/control_flow/router_agent/configs/config.yml --input "What red fruit would you recommend?"

# Test city advisor
nat run --config_file=examples/control_flow/router_agent/configs/config.yml --input "What city should I visit in the United States?"

# Test literature advisor
nat run --config_file=examples/control_flow/router_agent/configs/config.yml --input "Can you recommend something by Shakespeare?"
```

**Expected Workflow Output**
```console
nemo-agent-toolkit % nat run --config_file=examples/control_flow/router_agent/configs/config.yml --input "I want a yellow fruit"
2025-09-10 10:52:59,058 - nat.cli.commands.start - INFO - Starting NAT from config file: 'examples/control_flow/router_agent/configs/config.yml'

Configuration Summary:
--------------------
Workflow Type: router_agent
Number of Functions: 3
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Object Stores: 0
Number of Retrievers: 0
Number of TTC Strategies: 0
Number of Authentication Providers: 0

2025-09-10 10:52:59,927 - nat.agent.router_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: I want a yellow fruit
Agent's thoughts:
content='fruit_advisor' additional_kwargs={} response_metadata={}
------------------------------
2025-09-10 10:52:59,929 - nat.agent.base - INFO -
------------------------------
[AGENT]
Calling tools: fruit_advisor
Tool's input: I want a yellow fruit
Tool's response:
banana
------------------------------
2025-09-10 10:52:59,931 - nat.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['banana']
--------------------------------------------------
```

This demonstrates the Router Agent's efficient single-pass routing and execution pattern, making it ideal for scenarios where different types of requests need to be directed to specialized agents, functions or tools.

### Starting the NeMo Agent Toolkit Server

You can start the NeMo Agent toolkit server using the `nat serve` command with the appropriate configuration file.

**Starting the Router Agent Example Workflow**

```bash
nat serve --config_file=examples/control_flow/router_agent/configs/config.yml
```

### Making Requests to the NeMo Agent Toolkit Server

Once the server is running, you can make HTTP requests to interact with the workflow.

#### Non-Streaming Requests

**Non-Streaming Request to the Router Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "I want a yellow fruit"}'
```

#### Streaming Requests

**Streaming Request to the Router Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate/stream \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "I want a yellow fruit"}'
```
---
