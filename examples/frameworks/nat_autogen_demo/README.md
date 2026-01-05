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

- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Export Required Environment Variables](#export-required-environment-variables)
- [Run the Workflow](#run-the-workflow)
  - [Set up the MCP Server](#set-up-the-mcp-server)
  - [Expected Output](#expected-output)
- [Architecture](#architecture)

## Key Features

- **AutoGen Framework Integration:** Demonstrates the NVIDIA NeMo Agent toolkit support for Microsoft's AutoGen framework alongside other frameworks like LangChain/LangGraph and Semantic Kernel.
- **Multi-Agent Collaboration:** Shows two specialized agents working together - a WeatherAndTimeAgent for data retrieval and a FinalResponseAgent for response formatting.
- **Unified Tool Integration:** Uses the toolkit's unified abstraction to integrate both local tools (weather updates) and MCP tools (time service) without framework-specific code. MCP servers are hosted using the toolkit's native MCP server and integrated with AutoGen as a function.
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
```

### Export Required Environment Variables

If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/get-started/installation.md#obtain-api-keys) instructions to obtain API keys.

For NVIDIA NIM, export the following:

- `NVIDIA_API_KEY`

## Run the Workflow

### Set up the MCP Server

This example uses NeMo Agent toolkit's MCP client abstraction to connect to an MCP server. The MCP connection is configured in the workflow's YAML file, and the toolkit automatically wraps the MCP tools for use with AutoGen agents. This approach provides a consistent interface across all supported frameworks.

In a separate terminal, or in the background, run the MCP server with this command:

```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml --tool_names current_datetime
```

Then, run the workflow with the NAT CLI:

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

## Architecture

The AutoGen workflow consists of two main agents:

1. **WeatherAndTimeAgent**: Retrieves weather and time information using tools
   - Uses the `weather_update_tool` for current weather conditions
   - Uses the `mcp_time` tool group for accurate time information (configured through NAT's MCP client)
   - Responds with "DONE" when task is completed

2. **FinalResponseAgent**: Formats and presents the final response
   - Consolidates information from other agents
   - Provides clear, concise answers to user queries
   - Terminates the conversation with "APPROVE"

The agents communicate through AutoGen's RoundRobinGroupChat system, which manages the conversation flow and ensures proper termination when the task is complete.

### Tool Integration

This example demonstrates NeMo Agent toolkit's unified approach to tool integration:

- **Local tools** (like `weather_update_tool`) are defined as NAT functions
- **MCP tools** (like `mcp_time`) are configured in YAML using NAT's `mcp_client` function group

Both types of tools are passed to AutoGen agents through NAT's `builder.get_tools()` method, which automatically wraps them for the target framework. This eliminates the need for framework-specific MCP integration code and provides a consistent interface across all supported frameworks (AutoGen, LangChain, Semantic Kernel, and others).
