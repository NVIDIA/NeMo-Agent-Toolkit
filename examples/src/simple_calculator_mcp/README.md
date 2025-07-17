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

Install the basic Simple Calculator example first:

```bash
uv pip install -e examples/basic/functions/simple_calculator
```

## Installation

```bash
uv pip install -e examples/intermediate/MCP/simple_calculator_mcp
```

## Usage

### AIQ Toolkit as an MCP Client
You can run the simple calculator workflow using Remote MCP tools. In this case, the workflow acts as a MCP client and connects to the MCP server running on the specified URL. Details are provided in the [MCP Client Guide](../../../../docs/source/workflows/mcp/mcp-client.md).

### AIQ Toolkit as an MCP Server
You can publish the simple calculator tools via MCP using the `aiq mcp` command. Details are provided in the [MCP Server Guide](../../../../docs/source/workflows/mcp/mcp-server.md).

## Configuration Examples

| Configuration File | MCP Server Type | Available Tools |
|-------------------|-----------------|-----------------|
| `config-mcp-date.yml` | Date Server | Current time, date formatting |
| `config-mcp-math.yml` | Math Server | Advanced mathematical operations |
| `demo_config_mcp.yml` | Multiple Servers | Combined demonstration |
