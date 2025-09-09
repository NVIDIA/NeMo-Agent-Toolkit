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

This example demonstrates how to use a configurable Router Agent with the NeMo Agent toolkit. The Router Agent analyzes incoming requests and intelligently routes them to the most appropriate branch (tool or function) based on the request content. For this purpose, NeMo Agent toolkit provides a [`router_agent`](../../../docs/source/workflows/about/router-agent.md) workflow type.

## Table of Contents

- [Key Features](#key-features)
- [Graph Structure](#graph-structure)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Run the Workflow](#run-the-workflow)
  - [Starting the NeMo Agent Toolkit Server](#starting-the-nemo-agent-toolkit-server)
  - [Making Requests to the NeMo Agent Toolkit Server](#making-requests-to-the-nemo-agent-toolkit-server)
  - [Evaluating the Router Agent Workflow](#evaluating-the-router-agent-workflow)

## Key Features

- **Router Agent Architecture:** Demonstrates the `router_agent` workflow type that intelligently analyzes incoming requests and routes them to the most appropriate branch (tool or function).
- **Dual-Node Graph Structure:** Uses a streamlined two-node architecture with Agent Node (analyzes request and selects branch) and Tool Node (executes the selected branch).
- **Intelligent Request Routing:** Shows how the Router Agent analyzes user input and selects exactly one branch that best handles the request, making it ideal for scenarios where different types of requests need different specialized tools.
- **Calculator Function Integration:** Includes multiple calculator functions (`calculator_multiply`, `calculator_inequality`, `calculator_divide`, `calculator_subtract`) to demonstrate routing mathematical operations to appropriate specialized tools.
- **Single Branch Execution:** Demonstrates the Router Agent approach of analyzing the request, selecting one optimal branch, executing it, and returning the result without additional iterations.

## Graph Structure

The Router Agent uses a streamlined dual-node graph architecture that efficiently analyzes requests and routes them to appropriate branches. The following describes the agent's workflow:

**Workflow Overview:**
- **Start**: The agent begins processing with user input
- **Agent Node**: Analyzes the incoming request and selects the most appropriate branch from the available options
- **Conditional Edge**: Determines whether to proceed to tool execution or end the process
- **Tool Node**: Executes the selected branch (tool/function) with the original user input
- **End**: Process completes with the result from the executed branch

**Key Architecture Benefits:**
- **Efficient Routing**: Single-pass analysis and routing without iterative planning
- **Specialized Tool Selection**: Each branch can be a specialized tool optimized for specific types of requests
- **Deterministic Execution**: Once a branch is selected, it executes exactly once and returns the result
- **Scalable Branch Management**: Easy to add new branches without modifying the core routing logic

This architecture is ideal for scenarios where different types of user requests require different specialized tools or functions, such as routing mathematical operations to specific calculator functions, or directing different query types to appropriate search or processing tools.

## Configuration

The Router Agent is configured through the `config.yml` file. Key configuration elements include:

- **workflow._type**: Set to `router_agent` to use the Router Agent workflow type
- **workflow.branches**: List of available tools/functions that the agent can route requests to
- **workflow.llm_name**: The language model used for request analysis and routing decisions
- **workflow.verbose**: Enable detailed logging to see the routing decisions

**Example Configuration:**
```yaml
workflow:
  _type: router_agent
  branches: [calculator_multiply, calculator_inequality, calculator_divide, calculator_subtract]
  llm_name: nim_llm
  verbose: true
```

The agent will automatically analyze incoming requests and route them to the most appropriate branch based on the request content and the descriptions of available branches.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv sync --all-groups --all-extras
uv pip install -e .
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Run the Workflow

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
nat run --config_file=examples/agents/router/configs/config.yml --input "What is 48 multiplied by 37?"
```

**Additional Example Commands:**
```bash
# Test inequality comparison
nat run --config_file=examples/agents/router/configs/config.yml --input "Is 2004 greater than 1996?"

# Test division
nat run --config_file=examples/agents/router/configs/config.yml --input "What is 6054 divided by 3?"

# Test subtraction
nat run --config_file=examples/agents/router/configs/config.yml --input "What is 1900 minus 21?"
```

**Expected Workflow Output**
```console
<snipped for brevity>

2025-01-15 10:30:45,123 - nat.agent.router_agent.agent - INFO - Router Agent has chosen branch: calculator_multiply
2025-01-15 10:30:45,124 - nat.agent.router_agent.agent - INFO - Router Agent Tool Node - Calling tools: calculator_multiply
Tool's input: What is 48 multiplied by 37?
Tool's response:
48 * 37 = 1776

2025-01-15 10:30:45,456 - nat.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
48 * 37 = 1776
--------------------------------------------------
```

**How the Router Agent Works:**
1. **Request Analysis**: The agent analyzes the input "What is 48 multiplied by 37?" and identifies it as a multiplication operation
2. **Branch Selection**: Based on the analysis, it selects the `calculator_multiply` branch from the available options
3. **Tool Execution**: The selected branch (calculator function) processes the request and returns the mathematical result
4. **Result Return**: The agent returns the result without additional processing or iterations

This demonstrates the Router Agent's efficient single-pass routing and execution pattern, making it ideal for scenarios where different types of requests need to be directed to specialized tools or functions.

### Starting the NeMo Agent Toolkit Server

You can start the NeMo Agent toolkit server using the `nat serve` command with the appropriate configuration file.

**Starting the Router Agent Example Workflow**

```bash
nat serve --config_file=examples/agents/router/configs/config.yml
```

### Making Requests to the NeMo Agent Toolkit Server

Once the server is running, you can make HTTP requests to interact with the workflow.

#### Non-Streaming Requests

**Non-Streaming Request to the Router Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "What is 48 multiplied by 37?"}'
```

#### Streaming Requests

**Streaming Request to the Router Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate/stream \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "What is 48 multiplied by 37?"}'
```
---

### Evaluating the Router Agent Workflow
**Run and evaluate the `router_agent` example Workflow**

```bash
nat eval --config_file=examples/agents/router/configs/config.yml
```
