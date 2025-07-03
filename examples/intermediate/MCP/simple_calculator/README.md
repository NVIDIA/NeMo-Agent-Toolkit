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

This example demonstrates **Model Context Protocol (MCP) integration** using the Simple Calculator workflow. Learn how to connect AIQ toolkit with external MCP servers and publish AIQ tools as MCP services.

## 🎯 What You'll Learn

- **MCP Client Usage**: Connect to external MCP servers for remote tool access
- **MCP Server Setup**: Publish AIQ toolkit functions as MCP services
- **Protocol Integration**: Seamless interoperability with MCP ecosystem
- **Remote Tool Access**: Use tools hosted on different systems
- **Service Architecture**: Build distributed AI tool networks

## 🔗 Prerequisites

This example builds upon the [basic Simple Calculator](../../../basic/functions/simple_calculator/). Install it first:

```bash
uv pip install -e examples/basic/functions/simple_calculator
```

## 📦 Installation

```bash
uv pip install -e examples/intermediate/MCP/simple_calculator
```

## 🚀 Usage

### AIQ as MCP Client

Connect to external MCP servers to use remote tools:

#### Date MCP Server
```bash
aiq run --config_file examples/intermediate/MCP/simple_calculator/configs/config-mcp-date.yml --input "What day is it today and what's 2 + 3?"
```

#### Math MCP Server
```bash
aiq run --config_file examples/intermediate/MCP/simple_calculator/configs/config-mcp-math.yml --input "Calculate the square root of 144"
```

#### Combined Demo
```bash
aiq run --config_file examples/intermediate/MCP/simple_calculator/configs/demo_config_mcp.yml --input "What time is it and what's 5 * 7?"
```

### AIQ as MCP Server

Publish AIQ toolkit tools as MCP services:

```bash
# Start AIQ as MCP server
aiq mcp --config_file examples/basic/functions/simple_calculator/configs/config.yml

# Connect from MCP client applications
# The calculator tools are now available via MCP protocol
```

### External MCP Deployment

Deploy MCP servers independently:

```bash
# Navigate to deployment directory
cd examples/basic/functions/simple_calculator/deploy_external_mcp

# Build and run MCP server container
docker build -t simple-calculator-mcp .
docker run -p 8080:8080 simple-calculator-mcp
```

## 🔍 Key Features Demonstrated

- **Remote Tool Access**: Use tools hosted on different systems
- **Protocol Standardization**: Interoperable tool sharing
- **Distributed Architecture**: Scale tools across infrastructure
- **Service Discovery**: Dynamic tool registration and discovery
- **Cross-Platform Integration**: Connect different AI frameworks

## 📊 Available Configurations

| config File | MCP Server | Tools Available |
|-------------|------------|-----------------|
| `config-mcp-date.yml` | Date Server | Current time, date formatting |
| `config-mcp-math.yml` | Math Server | Advanced mathematical operations |
| `demo_config_mcp.yml` | Multiple | Combined demonstration |

## 🏗️ MCP Architecture

```
┌─────────────────┐    MCP Protocol    ┌─────────────────┐
│   AIQ Toolkit   │ ◄─────────────── │   MCP Server    │
│   (Client)      │                   │   (Tools)       │
└─────────────────┘                   └─────────────────┘
        │                                       │
        │              ┌─────────────────┐     │
        └─────────────►│   MCP Server    │─────┘
                       │   (AIQ Tools)   │
                       └─────────────────┘
```

## 🛠️ MCP Protocol Benefits

- **Standardization**: Common protocol for tool sharing
- **Interoperability**: Works across different AI systems
- **Scalability**: Distribute tools across networks
- **Security**: Controlled access to remote capabilities
- **Flexibility**: Mix and match tools from various sources

## 📋 Supported MCP Operations

- **Tool Discovery**: List available remote tools
- **Tool Execution**: Call remote functions with parameters
- **Resource Access**: Retrieve remote resources and data
- **Session Management**: Maintain stateful connections

This example showcases how AIQ toolkit seamlessly integrates with the MCP ecosystem for distributed AI tool architectures.
