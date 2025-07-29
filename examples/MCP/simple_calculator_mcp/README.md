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

# Simple Calculator - Model Context Protocol (MCP)

This example demonstrates how to integrate the NVIDIA NeMo Agent toolkit with Model Context Protocol (MCP) servers. You'll learn to use remote tools through MCP and publish Agent toolkit functions as MCP services.

## Table of Contents

- [Key Features](#key-features)
- [What is MCP?](#what-is-mcp)
- [What You'll Learn](#what-youll-learn)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
- [Run the Workflow](#run-the-workflow)
  - [AIQ Toolkit as an MCP Client](#aiq-toolkit-as-an-mcp-client)
  - [AIQ Toolkit as an MCP Server](#aiq-toolkit-as-an-mcp-server)
- [Configuration Examples](#configuration-examples)

## Key Features

- **MCP Client Integration:** Demonstrates how to use NeMo Agent toolkit as an MCP client to connect to remote MCP servers and access distributed tools like date/time services and advanced mathematical operations.
- **MCP Server Publishing:** Shows how to publish NeMo Agent toolkit functions as MCP services using the `aiq mcp` command, making calculator tools available to other AI systems through the standardized MCP protocol.
- **Distributed AI Tool Networks:** Enables building networks of interconnected AI tools where different capabilities can be hosted on separate systems and accessed remotely through MCP.
- **Cross-System Interoperability:** Demonstrates integration with the broader MCP ecosystem, allowing NeMo Agent toolkit workflows to both consume and provide tools in a standardized manner.
- **Remote Tool Access:** Shows how to securely connect to external data sources and tools through the MCP protocol while maintaining security and access control.

## What is MCP?

Model Context Protocol (MCP) is a standard protocol that enables AI applications to securely connect to external data sources and tools. It allows you to:

- **Access remote tools**: Use functions hosted on different systems
- **Share capabilities**: Publish your tools for other AI systems to use
- **Build distributed systems**: Create networks of interconnected AI tools
- **Maintain security**: Control access to remote capabilities

## What You'll Learn

- Connect to external MCP servers as a client
- Publish Agent toolkit functions as MCP services
- Build distributed AI tool networks
- Integrate with the broader MCP ecosystem

## Prerequisites

1. **Agent toolkit**: Ensure you have the Agent toolkit installed. If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.
2. **Base workflow**: This example builds upon the Getting Started [Simple Calculator](../../getting_started/simple_calculator/) example. Make sure you are familiar with the example before proceeding.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow

Install this example:

```bash
uv pip install -e examples/MCP/simple_calculator_mcp
```

## Run the Workflow

### AIQ Toolkit as an MCP Client
You can run the simple calculator workflow using Remote MCP tools. In this case, the workflow acts as a MCP client and connects to the MCP server running on the specified URL. Details are provided in the [MCP Client Guide](../../../docs/source/workflows/mcp/mcp-client.md).

### AIQ Toolkit as an MCP Server
You can publish the simple calculator tools via MCP using the `aiq mcp` command. Details are provided in the [MCP Server Guide](../../../docs/source/workflows/mcp/mcp-server.md).

## Configuration Examples

| Configuration File | MCP Server Type | Available Tools |
|-------------------|-----------------|-----------------|
| `config-mcp-date.yml` | Date Server | Current time, date formatting |
| `config-mcp-math.yml` | Math Server | Advanced mathematical operations |
| `config-combined.yml` | Multiple Servers | Combined demonstration |

**Note**: To run the `config-combined.yml` example successfully, you need to keep both MCP servers running:
- **Docker container MCP server**: Follow the setup instructions in `deploy_external_mcp/README.md` to start the containerized time server on port 8080
- **NeMo Agent Toolkit MCP server**: Use `aiq mcp --config_file examples/getting_started/simple_calculator/configs/config.yml` to serve calculator tools on port 9901

Both servers must be active simultaneously for the combined workflow to access all the required tools.
