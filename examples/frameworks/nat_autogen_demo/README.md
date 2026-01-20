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

# AutoGen Framework Example

A quick example using Microsoft's AutoGen framework showcasing a multi-agent weather and time information system where agents collaborate through AutoGen's conversation system to provide accurate weather and time data for specified cities.

## Table of Contents

- [AutoGen Framework Example](#autogen-framework-example)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Prerequisites](#prerequisites)
  - [Installation and Setup](#installation-and-setup)
    - [Install this Workflow](#install-this-workflow)
    - [Export Required Environment Variables](#export-required-environment-variables)
  - [Run the Workflow](#run-the-workflow)
    - [Set up the MCP Server](#set-up-the-mcp-server)
    - [Expected Output](#expected-output)
  - [Observability with Phoenix](#observability-with-phoenix)
    - [Install Phoenix Dependencies](#install-phoenix-dependencies)
    - [Start Phoenix Server](#start-phoenix-server)
    - [Run with Tracing Enabled](#run-with-tracing-enabled)
    - [View Traces in Phoenix](#view-traces-in-phoenix)
  - [Evaluate the Workflow](#evaluate-the-workflow)
    - [Evaluation Dataset](#evaluation-dataset)
    - [Run the Evaluation](#run-the-evaluation)
    - [Understanding Evaluation Results](#understanding-evaluation-results)
  - [Architecture](#architecture)
    - [Tool Integration](#tool-integration)

## Key Features

- **AutoGen Framework Integration:** Demonstrates the NVIDIA NeMo Agent toolkit support for Microsoft's AutoGen framework alongside other frameworks like LangChain/LangGraph and Semantic Kernel.
- **Multi-Agent Collaboration:** Shows two specialized agents working together - a WeatherAndTimeAgent for data retrieval and a FinalResponseAgent for response formatting.
- **Unified Tool Integration:** Uses the unified abstraction provided by the toolkit to integrate both local tools (weather updates) and MCP tools (time service) without framework-specific code. MCP servers are hosted using the native MCP server included in the toolkit and integrated with AutoGen as a function.
- **Round-Robin Group Chat:** Uses AutoGen's RoundRobinGroupChat for structured agent communication with termination conditions.

## Prerequisites

Before running this example, ensure you have:

- Python 3.11 or higher
- NeMo Agent toolkit installed (see [Install Guide](../../../docs/source/get-started/installation.md))
- NVIDIA API key for NIM access

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow

From the root directory of the NeMo Agent toolkit repository, run the following commands:

```bash
# Required to run the current_datetime MCP tool used in the example workflow
uv pip install -e examples/getting_started/simple_calculator

uv pip install -e ".[mcp]"

uv pip install -e examples/frameworks/nat_autogen_demo

uv pip install matplotlib
```

### Export Required Environment Variables

If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/get-started/installation.md#obtain-api-keys) instructions to obtain API keys.

For NVIDIA NIM, export the following:

- `NVIDIA_API_KEY`

## Run the Workflow

### Set up the MCP Server

This example uses the MCP client abstraction provided by NeMo Agent toolkit to connect to an MCP server. The MCP connection is configured in the workflow YAML file, and the toolkit automatically wraps the MCP tools for use with AutoGen agents. This approach provides a consistent interface across all supported frameworks.

In a separate terminal, or in the background, run the MCP server with this command:

```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml --tool_names current_datetime
```

Then, run the workflow with the CLI provided by the toolkit:

```bash
nat run --config_file examples/frameworks/nat_autogen_demo/configs/config.yml --input "What is the weather and time in New York today?"
```

### Expected Output

```console
2025-10-07 14:34:28,122 - nat.cli.commands.start - INFO - Starting NAT from config file: 'examples/frameworks/nat_autogen_demo/configs/config.yml'
2025-10-07 14:34:30,285 - mcp.client.streamable_http - INFO - Received session ID: 652a05b6646c4ddb945cf2adf0b3ec18
Received session ID: 652a05b6646c4ddb945cf2adf0b3ec18
2025-10-07 14:34:30,287 - mcp.client.streamable_http - INFO - Negotiated protocol version: 2025-06-18
Negotiated protocol version: 2025-06-18

Configuration Summary:
--------------------
Workflow Type: autogen_team
Number of Functions: 1
Number of Function Groups: 0
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Object Stores: 0
Number of Retrievers: 0
Number of TTC Strategies: 0
Number of Authentication Providers: 0

2025-10-07 14:34:30,301 - nat.observability.exporter_manager - INFO - Started exporter 'otelcollector'
2025-10-07 14:34:46,704 - nat.front_ends.console.console_front_end_plugin - INFO -
.
.
.
<snipped for brevity>
.
.
.
--------------------------------------------------
Workflow Result:
['New York weather: Sunny, around 25°C (77°F).\nCurrent local time in New York: 5:34 PM EDT (UTC−4) on October 7, 2025.\n\nAPPROVE']
--------------------------------------------------

```

## Observability with Phoenix

This section demonstrates how to enable distributed tracing using Phoenix to monitor and analyze the AutoGen workflow execution.

### Install Phoenix Dependencies

Phoenix requires the NeMo Agent toolkit Phoenix plugin and the Phoenix server. Install them with:

```bash
# Install NAT Phoenix plugin for tracing integration
uv pip install "nvidia-nat[phoenix]"

# Install Phoenix server
uv pip install arize-phoenix
```

### Start Phoenix Server

Phoenix provides local tracing capabilities for development and testing. In a separate terminal, start Phoenix:

```bash
phoenix serve
```

Phoenix runs on `http://localhost:6006` with the tracing endpoint at `http://localhost:6006/v1/traces`.

### Run with Tracing Enabled

With Phoenix running, execute the workflow using the evaluation config which has tracing enabled:

```bash
nat run --config_file examples/frameworks/nat_autogen_demo/configs/config-eval.yml \
  --input "What is the weather and time in New York?"
```

### View Traces in Phoenix

Open your browser to `http://localhost:6006` to explore traces in the Phoenix UI. You can see:

- **Agent execution flow**: Track the conversation between WeatherAndTimeAgent and FinalResponseAgent
- **Tool invocations**: Monitor calls to `weather_update_tool` and `current_datetime`
- **LLM interactions**: View prompts, completions, and token usage
- **Timing metrics**: Analyze latency across different workflow components

## Evaluate the Workflow

NeMo Agent toolkit provides a comprehensive evaluation framework to assess your workflow's performance against a test dataset.

### Evaluation Dataset

The evaluation dataset contains three test cases with different cities:

| ID | City | Description |
|----|------|-------------|
| 1 | New York | Weather and time in Eastern Time zone |
| 2 | London | Weather and time in British Time zone |
| 3 | Tokyo | Weather and time in Japan Standard Time zone |

The dataset is located at `examples/frameworks/nat_autogen_demo/data/eval_dataset.json`.

### Run the Evaluation

Ensure both the MCP server and Phoenix are running, then execute the evaluation:

```bash
# Terminal 1: Start MCP server (if not already running)
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml --tool_names current_datetime

# Terminal 2: Start Phoenix server (if not already running)
phoenix serve

# Terminal 3: Run evaluation
nat eval --config_file examples/frameworks/nat_autogen_demo/configs/config-eval.yml
```

The evaluation runs the workflow against all three test cases and evaluates results using:

- **Answer Accuracy**: Measures how accurately the agent answers the questions
- **Response groundedness**: Evaluates whether responses are grounded in the tool outputs
- **Trajectory Accuracy**: Assesses the agent's decision-making path and tool usage

### Understanding Evaluation Results

The `nat eval` command produces several output files in `.tmp/nat/examples/frameworks/nat_autogen_demo/eval/`:

- **`workflow_output.json`**: Raw outputs from the workflow for each input
- **Evaluator-specific files**: Each configured evaluator generates its own output file with scores and reasoning

Example output:

```console
2025-10-07 15:00:00,000 - nat.eval - INFO - Running evaluation with 3 test cases...
2025-10-07 15:00:30,000 - nat.eval - INFO - Evaluation complete

Results Summary:
----------------
accuracy: 0.85
groundedness: 0.90
trajectory_accuracy: 0.88

Detailed results saved to: .tmp/nat/examples/frameworks/nat_autogen_demo/eval/
```

Each evaluator provides:

- An **average score** across all dataset entries (0-1 scale, where 1 is perfect)
- **Individual scores** for each entry with detailed reasoning
- **Performance metrics** to help identify areas for improvement

View detailed traces for each evaluation run in Phoenix at `http://localhost:6006`.

## Architecture

The AutoGen workflow consists of two main agents:

1. **WeatherAndTimeAgent**: Retrieves weather and time information using tools
   - Uses the `weather_update_tool` for current weather conditions
   - Uses the `mcp_time` tool group for accurate time information (configured through the MCP client provided by the toolkit)
   - Responds with "DONE" when task is completed

2. **FinalResponseAgent**: Formats and presents the final response
   - Consolidates information from other agents
   - Provides clear, concise answers to user queries
   - Terminates the conversation with "APPROVE"

The agents communicate through AutoGen's RoundRobinGroupChat system, which manages the conversation flow and ensures proper termination when the task is complete.

### Tool Integration

This example demonstrates the unified approach to tool integration provided by NeMo Agent toolkit:

- **Local tools** (like `weather_update_tool`) are defined as functions in the toolkit
- **MCP tools** (like `mcp_time`) are configured in YAML using the `mcp_client` function group provided by the toolkit

Both types of tools are passed to AutoGen agents through the `builder.get_tools()` method included in the toolkit, which automatically wraps them for the target framework. This eliminates the need for framework-specific MCP integration code and provides a consistent interface across all supported frameworks (AutoGen, LangChain, Semantic Kernel, and others).
