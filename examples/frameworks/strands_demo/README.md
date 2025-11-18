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

# Strands Example

A minimal example showcasing a Strands agent that answers questions about Strands documentation using a curated URL knowledge base and the native Strands `http_request` tool.

## Table of Contents

- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Run the Workflow](#run-the-workflow)
  - [Run the workflow (config.yml)](#1-run-the-workflow-configyml)
  - [Serve AgentCore-compatible endpoints (agentcore_config.yml)](#2-serve-agentcore-compatible-endpoints-agentcore_configyml)
  - [Evaluate accuracy and performance (eval_config.yml)](#3-evaluate-accuracy-and-performance-eval_configyml)

## Key Features

- **Strands framework integration**: Demonstrates support for Strands Agents in the NeMo Agent toolkit.
- **AgentCore Integration**: Demonstrates an agent that can be run on Amazon Bedrock AgentCore runtime.
- **Evaluation and Performance Metrics**: Runs dataset-driven evaluation and performance analysis via `nat eval`.
- **Support for Model Providers**: Configuration includes NIM, OpenAI, and AWS Bedrock options.

## Prerequisites

- NeMo Agent toolkit installed. See the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source).
- API keys as required by your chosen models.

## Installation and Setup

### Install this Workflow

```bash
uv pip install -e examples/frameworks/strands_demo
```

### Set Up API Keys

```bash
export NVIDIA_API_KEY=<YOUR_NVIDIA_API_KEY>
# Optional if you switch models in the config
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
export AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
export AWS_DEFAULT_REGION=us-east-1
```

## Run the Workflow

The `configs/` directory contains four ready-to-use configurations. Use the commands below.

### 1) Run the workflow (config.yml)

```bash
nat run --config_file examples/frameworks/strands_demo/configs/config.yml \
  --input "How do I use the Strands Agents API?"
```

**Expected Workflow Output**
The workflow produces a large amount of output, the end of the output should contain something similar to the following:

```console
Workflow Result:
['To answer your question about using the Strands Agents API, I\'ll need to search for the relevant documentation. Let me do that for you.Thank you for providing that information. To get the most relevant information about using the Strands Agents API, I\'ll fetch the content from the "strands_agent_loop" URL, as it seems to be the most relevant to your question about using the API.Based on the information from the Strands Agents documentation, I can provide you with an overview of how to use the Strands Agents API. Here\'s a summary of the key points:\n\n1. Initialization:\n   To use the Strands Agents API, you start by initializing an agent with the necessary components:\n\n   ```python\n   from strands import Agent\n   from strands_tools import calculator\n\n   agent = Agent(\n       tools=[calculator],\n       system_prompt="You are a helpful assistant."\n   )\n   ```\n\n   This sets up the agent with tools (like a calculator in this example) and a system prompt.\n\n2. Processing User Input:\n   You can then use the agent to process user input:\n\n   ```python\n   result = agent("Calculate 25 * 48")\n   ```\n\n3. Agent Loop:\n   The Strands Agents API uses an "agent loop" to process requests. This loop includes:\n   - Receiving user input and context\n   - Processing the input using a language model (LLM)\n   - Deciding whether to use tools to gather information or perform actions\n   - Executing tools and receiving results\n   - Continuing reasoning with new information\n   - Producing a final response or iterating through the loop again\n\n4. Tool Execution:\n   The agent can use tools as part of its processing. When the model decides to use a tool, it will format a request like this:\n\n   ```json\n   {\n     "role": "assistant",\n     "content": [\n       {\n         "toolUse": {\n           "toolUseId": "tool_123",\n           "name": "calculator",\n           "input": {\n             "expression": "25 * 48"\n           }\n         }\n       }\n     ]\n   }\n   ```\n\n   The API then executes the tool and returns the result to the model for further processing.\n\n5. Recursive Processing:\n   The agent loop can continue recursively if more tool executions or multi-step reasoning is required.\n\n6. Completion:\n   The loop completes when the model generates a final text response or when an unhandled exception occurs.\n\nTo effectively use the Strands Agents API, you should:\n- Initialize your agent with appropriate tools and system prompts\n- Design your tools carefully, considering token limits and complexity\n- Handle potential exceptions, such as the MaxTokensReachedException\n\nRemember that the API is designed to support complex, multi-step reasoning and actions with seamless integration of tools and language models. It\'s flexible enough to handle a wide range of tasks and can be customized to your specific needs.']
```

### 2) Serve AgentCore-compatible endpoints (agentcore_config.yml)

To run the agent on Amazon Bedrock AgentCore [runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/getting-started-custom.html).  Note that `agentcore_config.yml` defines two required endpoints for AgentCore.  This configuration is a general requirement for any agent, regardless of whether it uses the Strands integration.

```bash
nat serve --config_file examples/frameworks/strands_demo/configs/agentcore_config.yml
```

### 3) Evaluate accuracy and performance (eval_config.yml)

Runs the workflow over a dataset and computes evaluation and performance metrics.  See the evaluation guide and profiling guides in `docs/source/workflows/` for more information.

```bash
nat eval --config_file examples/frameworks/strands_demo/configs/eval_config.yml
```
> Tip: If you hit rate limits, lower concurrency: `--override eval.general.max_concurrency 1`.
