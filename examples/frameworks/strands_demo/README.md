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
  - [Optimize workflow parameters (optimizer_config.yml)](#4-optimize-workflow-parameters-optimizer_configyml)
  - [Profile for GPU cluster sizing (sizing_config.yml)](#5-profile-for-gpu-cluster-sizing-sizing_configyml)

## Key Features

- **Strands framework integration**: Demonstrates support for Strands Agents in the NeMo Agent Toolkit.
- **AgentCore Integration**: Demonstrates an agent that can be run on Amazon Bedrock AgentCore runtime.
- **Evaluation and Performance Metrics**: Runs dataset-driven evaluation and performance analysis via `nat eval`.
- **Support for Model Providers**: Configuration includes NIM, OpenAI, and AWS Bedrock options.

## Prerequisites

- NeMo Agent Toolkit installed. See the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source).
- API keys as required by your chosen models.

## Installation and Setup

### Install this Workflow

```bash
uv pip install -e examples/frameworks/strands_demo
```

### Set Up API Keys

> **Note:** The `NVIDIA_API_KEY` is required only when using NVIDIA-hosted NIM endpoints (default configuration). If you are using a self-hosted NVIDIA NIM or model with OAI compatible endpoint and a custom `base_url` specified in your configuration file (such as in `examples/frameworks/strands_demo/configs/sizing_config.yml`), you do not need to set the `NVIDIA_API_KEY`.

```bash
export NVIDIA_API_KEY=<YOUR_NVIDIA_API_KEY>
```

**Optional:** Set these only if you switch to different LLM providers in the config:

```bash
# For OpenAI models
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>

# For AWS Bedrock models
export AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
export AWS_DEFAULT_REGION=us-east-1
```

## Run the Workflow

The `configs/` directory contains five ready-to-use configurations. Use the commands below.

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

<!-- path-check-skip-next-line -->
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

### 4) Optimize workflow parameters (optimizer_config.yml)

Automatically finds optimal LLM parameters (temperature, top_p, max_tokens) through systematic experimentation. The optimizer evaluates multiple parameter combinations across multiple trials and repetitions, balancing accuracy, groundedness, relevance, trajectory correctness, latency, and token efficiency.

```bash
nat optimize --config_file examples/frameworks/strands_demo/configs/optimizer_config.yml
```

**What it optimizes:**
- **Temperature**: Tests values from 0.1 to 0.7
- **Top_p**: Tests values from 0.7 to 1.0  
- **Max_tokens**: Tests values from 4096 to 8192

The optimizer runs 20 trials with 3 repetitions each for statistical stability and generates a report showing the best parameter combination based on weighted multi-objective scoring.

> Note: Optimization can take significant time. Reduce `n_trials` or adjust the search space in the config for faster experimentation.

### 5) Profile for GPU cluster sizing (sizing_config.yml)

Profiles workflow performance and calculates GPU cluster sizing requirements based on target users and workflow runtime. This configuration requires updating the `base_url` parameter to point to your self-hosted NVIDIA NIM or model with OAI compatible endpoint.

**Step 1: Collect profiling data**

First, update the `base_url` in `examples/frameworks/strands_demo/configs/sizing_config.yml` to point to your self-hosted NVIDIA NIM or model endpoint, then run the sizing profiler to collect performance metrics at different concurrency levels:

```bash
nat sizing calc --config_file examples/frameworks/strands_demo/configs/sizing_config.yml \
  --calc_output_dir /tmp/strands_demo/sizing_calc_run1/ \
  --concurrencies 1,2,4,8,16,32 \
  --num_passes 2
```

This command profiles the workflow at multiple concurrency levels (1, 2, 4, 8, 16, and 32 concurrent requests) with 2 passes for each level to establish baseline performance characteristics.

**Step 2: Calculate GPU sizing for target workload**

Use the profiling data to determine GPU requirements for your target user count and workflow runtime:

```bash
# For 100 concurrent users with 20-second target runtime
nat sizing calc --offline_mode \
  --calc_output_dir /tmp/strands_demo/sizing_calc_run1/ \
  --test_gpu_count 8 \
  --target_workflow_runtime 20 \
  --target_users 100

# For 25 concurrent users with 20-second target runtime
nat sizing calc --offline_mode \
  --calc_output_dir /tmp/strands_demo/sizing_calc_run1/ \
  --test_gpu_count 8 \
  --target_workflow_runtime 20 \
  --target_users 25
```

**Parameters:**
- `--offline_mode`: Uses previously collected profiling data
- `--calc_output_dir`: Directory containing the profiling results
- `--test_gpu_count`: Number of GPUs used during profiling (8 in this example)
- `--target_workflow_runtime`: Desired workflow completion time in seconds
- `--target_users`: Number of concurrent users to support

The sizing calculator will output the recommended GPU count needed to meet your performance targets.
