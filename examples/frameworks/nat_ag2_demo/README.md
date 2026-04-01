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
<!-- path-check-skip-file -->

# AG2 Framework Example

**Complexity:** 🟨 Intermediate

A quick example using the AG2 framework (formerly AutoGen), showcasing a multi-agent Los Angeles traffic information system where agents collaborate through AG2's `ConversableAgent` and `AutoPattern` to provide real-time traffic status for highways based on the current time of day.

## Table of Contents

- [AG2 Framework Example](#ag2-framework-example)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Prerequisites](#prerequisites)
  - [Installation and Setup](#installation-and-setup)
    - [Install this Workflow](#install-this-workflow)
    - [Export Required Environment Variables](#export-required-environment-variables)
  - [Run the Workflow](#run-the-workflow)
    - [Set up the MCP Server](#set-up-the-mcp-server)
    - [Expected Output](#expected-output)
  - [Async Workflow](#async-workflow)
  - [Research Team Example](#research-team-example)
  - [Observability with Phoenix](#observability-with-phoenix)
    - [Start Phoenix Server](#start-phoenix-server)
    - [Run with Tracing Enabled](#run-with-tracing-enabled)
    - [View Traces in Phoenix](#view-traces-in-phoenix)
  - [Evaluate the Workflow](#evaluate-the-workflow)
    - [Evaluation Dataset](#evaluation-dataset)
    - [Run the Evaluation](#run-the-evaluation)
    - [Understanding Evaluation Results](#understanding-evaluation-results)
  - [Architecture](#architecture)
    - [Async Execution](#async-execution)
    - [Tool Integration](#tool-integration)
  - [Available Configs](#available-configs)

## Key Features

- **AG2 Framework Integration:** Demonstrates NVIDIA NeMo Agent Toolkit support for AG2 (formerly AutoGen) alongside other frameworks like LangChain/LangGraph and Semantic Kernel.
- **Native Async Support:** All workflows use AG2's `a_initiate_group_chat` for non-blocking async execution, with tools awaited natively via `async`/`await`.
- **Multi-Agent Collaboration:** Shows two specialized agents working together — a TrafficAgent for data retrieval and a FinalResponseAgent for response formatting.
- **Time-Aware Traffic Status:** Provides realistic traffic information that varies based on time of day (morning rush, evening rush, off-peak hours).
- **Unified Tool Integration:** Uses the unified abstraction provided by the toolkit to integrate both local tools (traffic status) and MCP tools (time service) without framework-specific code.
- **AutoPattern Group Chat:** Uses AG2's `ConversableAgent` with `AutoPattern` and `a_initiate_group_chat` for structured agent communication.

## Prerequisites

Before running this example, ensure you have:

- Python 3.11 or higher
- NeMo Agent Toolkit installed (see [Install Guide](../../../docs/source/get-started/installation.md))
- NVIDIA API key for NIM access

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md) to create the development environment and install NeMo Agent Toolkit.

### Install this Workflow

From the root directory of the NeMo Agent Toolkit repository, run the following commands:

```bash
# Install the demo workflow and its dependencies (this also installs the core toolkit and required plugins)
uv pip install -e examples/frameworks/nat_ag2_demo

# Required to run the current_datetime MCP tool used in the example workflow
uv pip install -e examples/getting_started/simple_calculator

# Optional: Install Phoenix for observability and tracing
uv pip install -e '.[phoenix]'
uv pip install arize-phoenix

uv pip install matplotlib
```

### Export Required Environment Variables

If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/get-started/installation.md#obtain-api-keys) instructions to obtain API keys.

For NVIDIA NIM, set the following environment variable:

```bash
export NVIDIA_API_KEY="YOUR-NVIDIA-API-KEY-HERE"
```

## Run the Workflow

### Set up the MCP Server

This example uses the MCP client abstraction provided by NeMo Agent Toolkit to connect to an MCP server. The MCP connection is configured in the workflow YAML file, and the toolkit automatically wraps the MCP tools for use with AG2 agents.

In a separate terminal, or in the background, run the MCP server with this command:

```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml --tool_names current_datetime
```

> [!NOTE]
> If the MCP server is not started as a background task (using the `&` operator), you will need to open a new terminal session, activate the uv environment, and export NVIDIA_API_KEY again.

Then, run the workflow with the CLI provided by the toolkit:

```bash
nat run --config_file examples/frameworks/nat_ag2_demo/configs/config.yml --input "What is the current traffic on the 405 South?"
```

### Expected Output

```console
% nat run --config_file examples/frameworks/nat_ag2_demo/configs/config.yml --input "What is the current traffic on the 405 South?"
2026-01-16 11:30:54 - INFO     - nat.cli.commands.start:192 - Starting NAT from config file: 'examples/frameworks/nat_ag2_demo/configs/config.yml'
...

--------------------------------------------------
Workflow Result:
["The current traffic conditions on the 405 South are as follows:\n\n* Segment: Mulholland Drive to LAX\n* Traffic Conditions: Light\n\nIt appears that traffic is relatively clear on the 405 South.\n\nAPPROVE"]
```

## Async Workflow

The `config-async.yml` config demonstrates the same traffic workflow using a dedicated async workflow type (`ag2_async_team`). This config is functionally identical to the default but uses a separate workflow registration to clearly distinguish the async execution path:

```bash
nat run --config_file examples/frameworks/nat_ag2_demo/configs/config-async.yml \
  --input "What is the current traffic on the 405 South?"
```

Both `config.yml` and `config-async.yml` use AG2's `a_initiate_group_chat` for async execution. The async config exists as an explicit example of the async pattern for reference.

## Research Team Example

The `config-research.yml` config demonstrates a different agent pattern — a researcher and writer agent collaborate to produce a structured research summary:

```bash
nat run --config_file examples/frameworks/nat_ag2_demo/configs/config-research.yml \
  --input "What are the latest advances in quantum computing?"
```

This config uses the `ag2_research_team` workflow type directly as the workflow, with prompts configured inline.

## Observability with Phoenix

This section demonstrates how to enable distributed tracing using Phoenix to monitor and analyze the AG2 workflow execution.

### Start Phoenix Server

In a separate terminal, start Phoenix:

```bash
phoenix serve
```

Phoenix runs on `http://localhost:6006` with the tracing endpoint at `http://localhost:6006/v1/traces`.

### Run with Tracing Enabled

With Phoenix running, execute the workflow using the evaluation config which has tracing enabled:

```bash
nat run --config_file examples/frameworks/nat_ag2_demo/configs/config-eval.yml \
  --input "What is the current traffic on the 10 West?"
```

### View Traces in Phoenix

Open your browser to `http://localhost:6006` to explore traces in the Phoenix UI. You can see:

- **Agent execution flow**: Track the conversation between TrafficAgent and FinalResponseAgent
- **Tool invocations**: Monitor calls to `traffic_status_tool` and `current_datetime`
- **LLM interactions**: View prompts, completions, and token usage
- **Timing metrics**: Analyze latency across different workflow components

## Evaluate the Workflow

NeMo Agent Toolkit provides a comprehensive evaluation framework to assess your workflow's performance against a test dataset.

### Evaluation Dataset

The evaluation dataset contains three test cases with different Los Angeles highways:

| ID | Highway | Direction | Description |
|----|---------|-----------|-------------|
| 1 | 405 | South | Major freeway connecting San Fernando Valley to LAX |
| 2 | 10 | West | Santa Monica Freeway from Downtown LA to Santa Monica |
| 3 | 110 | North | Harbor Freeway from Long Beach to Pasadena |

The dataset is located at `examples/frameworks/nat_ag2_demo/data/toy_data.json`.

Traffic status varies by time period:

- **Morning Rush (7-9 AM):** Inbound routes (405-South, 110-South, 10-East, 210-East) are heavy
- **Evening Rush (4-7 PM):** Outbound routes (405-North, 110-North, 10-West, 210-West) are heavy
- **Off-Peak:** All routes are light

### Run the Evaluation

Ensure both the MCP server and Phoenix are running, then execute the evaluation:

```bash
# Terminal 1: Start MCP server (if not already running)
# nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml --tool_names current_datetime

# Terminal 2: Start Phoenix server (if not already running)
# phoenix serve

# Terminal 3: Run evaluation
nat eval --config_file examples/frameworks/nat_ag2_demo/configs/config-eval.yml
```

The evaluation runs the workflow against all three test cases and evaluates results using:

- **Answer `Accuracy`**: Measures how accurately the agent answers the questions
- **Response `Groundedness`**: Evaluates whether responses are grounded in the tool outputs
- **Trajectory `Accuracy`**: Assesses the agent's decision-making path and tool usage

### Understanding Evaluation Results

The `nat eval` command produces several output files in `.tmp/nat/examples/frameworks/nat_ag2_demo/traffic_eval/`:

- **`workflow_output.json`**: Raw outputs from the workflow for each input
- **Evaluator-specific files**: Each configured evaluator generates its own output file with scores and reasoning

## Architecture

The AG2 workflow consists of two main agents:

1. **TrafficAgent**: Retrieves traffic information using tools
   - Uses the `current_datetime` MCP tool to get the current time
   - Uses the `traffic_status_tool` to get traffic conditions for LA highways based on the hour
   - Responds with "DONE" when the task is completed

2. **FinalResponseAgent**: Formats and presents the final response
   - Consolidates information from the TrafficAgent
   - Provides clear, concise answers to user queries
   - Terminates the conversation with "APPROVE"

The agents communicate through AG2's `AutoPattern` with `a_initiate_group_chat`. A `ConversableAgent` with `human_input_mode="NEVER"` serves as both the user initiator and tool executor — tool calls from TrafficAgent are routed to it for execution, keeping the group chat flow self-contained.

### Async Execution

All AG2 demo workflows use native async execution:

- **`a_initiate_group_chat`** orchestrates agent turns without blocking the event loop
- **Tool functions** are async coroutines (`await fn.acall_invoke(...)`) executed natively by AG2's `a_execute_function`, which checks `is_coroutine_callable(func)` and awaits the result
- **Streaming tools** collect results into a single response since AG2 tools return a single value

This replaces the previous approach of running async tool calls through a `ThreadPoolExecutor`, eliminating unnecessary thread overhead.

### Tool Integration

This example demonstrates the unified approach to tool integration provided by NeMo Agent Toolkit:

- **Local tools** (like `traffic_status_tool`) are defined as NAT functions and provide time-aware traffic data for Los Angeles highways
- **MCP tools** (like `current_datetime`) are configured in YAML using the `mcp_client` function group provided by the toolkit

Both types of tools are passed to AG2 agents through the `builder.get_tools()` method, which automatically wraps them for the AG2 framework. This eliminates the need for framework-specific MCP integration code and provides a consistent interface across all supported frameworks (AG2, AutoGen, LangChain, Semantic Kernel, and others).

## Available Configs

| Config | Workflow Type | Description |
|--------|--------------|-------------|
| `config.yml` | `ag2_team` | Default traffic workflow with async group chat |
| `config-async.yml` | `ag2_async_team` | Explicit async variant of the traffic workflow |
| `config-research.yml` | `ag2_research_team` | Research team with researcher + writer agents |
| `config-eval.yml` | `ag2_team` | Traffic workflow with Phoenix tracing and evaluation |

