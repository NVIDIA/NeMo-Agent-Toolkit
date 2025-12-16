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

This example demonstrates how to integrate the NVIDIA NeMo Agent toolkit with [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/modelcontextprotocol) servers. You'll learn to use remote tools through MCP and publish Agent toolkit functions as MCP services.

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

### NeMo Agent toolkit as an MCP Client
You can run the simple calculator workflow using Remote MCP tools. In this case, the workflow acts as a MCP client and connects to the MCP server running on the specified URL. Details are provided in the [MCP Client Guide](../../../docs/source/workflows/mcp/mcp-client.md).

### NeMo Agent toolkit as an MCP Server
You can publish the simple calculator tools via MCP using the `nat mcp serve` command. Details are provided in the [MCP Server Guide](../../../docs/source/workflows/mcp/mcp-server.md).


### MCP Client Configuration
NeMo Agent toolkit enables workflows to use MCP tools as functions. The library handles the MCP server connection, tool discovery, and function registration. This allows the workflow to use MCP tools as regular functions.

Tools served by remote MCP servers can be leveraged as NeMo Agent toolkit functions in one of two ways:
- `mcp_client`: A flexible configuration using function groups, that allows you to connect to a MCP server, dynamically discover the tools it serves, and register them as NeMo Agent toolkit functions. `config-mcp-client.yml` example demonstrates how to use the `mcp_client` function group with both local and remote MCP servers.
- `mcp_tool_wrapper`: A simple configuration that allows you to wrap a single MCP tool as a NeMo Agent toolkit function. `config-mcp-tool-wrapper.yml` example demonstrates how to use the `mcp_tool_wrapper` function group with a remote MCP server.

### Running the example
The `config-mcp-client.yml` example demonstrates how to use the `mcp_client` function group with both local and remote MCP servers. This configuration shows how to use multiple MCP servers with different transports in the same workflow.

`examples/MCP/simple_calculator_mcp/configs/config-mcp-client.yml`:
```yaml
function_groups:
  mcp_time:
    _type: mcp_client
    server:
      transport: stdio
      command: "python"
      args: ["-m", "mcp_server_time", "--local-timezone=America/Los_Angeles"]
  mcp_math:
    _type: mcp_client
    server:
      transport: streamable-http
      url: "http://localhost:9901/mcp"

workflow:
  _type: react_agent
  tool_names:
    - mcp_time
    - mcp_math
```

This configuration creates two function groups:
- `mcp_time`: Connects to a local MCP server using stdio transport to get current date and time
- `mcp_math`: Connects to a remote MCP server using streamable-http transport to access calculator tools

To run this example:

1. Start the remote MCP server:
```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml
```
This starts an MCP server on port 9901 with endpoint `/mcp` and uses `streamable-http` transport. See the [MCP Server](../../../docs/source/workflows/mcp/mcp-server.md) documentation for more information.

2. Run the workflow:
```bash
nat run --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-client.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```
