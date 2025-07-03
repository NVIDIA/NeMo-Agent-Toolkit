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

### Using Agent Toolkit as MCP Client

Connect to external MCP servers to access remote tools:

#### Date Server Example
Access date and time functions from a remote MCP server:

```bash
aiq run --config_file examples/intermediate/MCP/simple_calculator_mcp/configs/config-mcp-date.yml --input "What day is it today and what's 2 + 3?"
```

#### Math Server Example
Use advanced mathematical operations from a remote server:

```bash
aiq run --config_file examples/intermediate/MCP/simple_calculator_mcp/configs/config-mcp-math.yml --input "Calculate the square root of 144"
```

#### Combined Example
Connect to multiple MCP servers simultaneously:

```bash
aiq run --config_file examples/intermediate/MCP/simple_calculator_mcp/configs/demo_config_mcp.yml --input "What time is it and what's 5 * 7?"
```

### Using Agent Toolkit as MCP Server

Publish your Agent toolkit functions as MCP services for other applications:

```bash
# Start Agent toolkit as MCP server
aiq mcp --config_file examples/basic/functions/simple_calculator/configs/config.yml

# Your calculator tools are now available via MCP protocol
# Other MCP clients can connect and use these tools
```

### External MCP Deployment

Deploy MCP servers independently using Docker:

```bash
# Navigate to deployment directory
cd examples/basic/functions/simple_calculator/deploy_external_mcp

# Build and run MCP server container
docker build -t simple-calculator-mcp .
docker run -p 8080:8080 simple-calculator-mcp
```

## Configuration Examples

| Configuration File | MCP Server Type | Available Tools |
|-------------------|-----------------|-----------------|
| `config-mcp-date.yml` | Date Server | Current time, date formatting |
| `config-mcp-math.yml` | Math Server | Advanced mathematical operations |
| `demo_config_mcp.yml` | Multiple Servers | Combined demonstration |

## Key Benefits

- **Standardization**: Use a common protocol for tool sharing
- **Interoperability**: Connect different AI systems seamlessly
- **Scalability**: Distribute tools across your infrastructure
- **Security**: Control access to remote capabilities
- **Flexibility**: Mix and match tools from various sources

## MCP Operations

The Agent toolkit supports these MCP operations:

- **Tool Discovery**: List available remote tools
- **Tool Execution**: Call remote functions with parameters
- **Resource Access**: Retrieve remote resources and data
- **Session Management**: Maintain stateful connections

## Learn More

- [MCP Client Guide](../../../../docs/source/workflows/mcp/mcp-client.md) - Using Agent toolkit as MCP client
- [MCP Server Guide](../../../../docs/source/workflows/mcp/mcp-server.md) - Publishing Agent toolkit tools via MCP
