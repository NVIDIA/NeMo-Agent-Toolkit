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

# NeMo Agent Toolkit as an MCP Client

Model Context Protocol (MCP) is an open protocol developed by Anthropic that standardizes how applications provide context to LLMs. You can read more about MCP [here](https://modelcontextprotocol.io/introduction).

You can use NeMo Agent toolkit as an MCP Host with one or more MCP Clients serving tools from remote MCP servers.

This guide will cover how to use NeMo Agent toolkit as an MCP Host with one or more MCP Clients. For more information on how to use NeMo Agent toolkit as an MCP Server, please refer to the [MCP Server](./mcp-server.md) documentation.

## Installation

MCP client functionality requires the `nvidia-nat-mcp` package. Install it with:

```bash
uv pip install nvidia-nat[mcp]
```

## MCP Client
NeMo Agent toolkit enables workflows to use MCP tools as functions. The library handles the MCP server connection, tool discovery, and function registration. This allow the workflow to use MCP tools as regular functions.

## Usage
Tools served by remote MCP servers can be leveraged as NeMo Agent toolkit functions in one of two ways:
- `mcp_client`: A flexible configuration using function groups, that allows you to connect to a MCP server, dynamically discover the tools it serves, and register them as NeMo Agent toolkit functions.
- `mcp_tool_wrapper`: A simple configuration that allows you wrap a single MCP tool as a NeMo Agent toolkit function.

### `mcp_client` Configuration
```yaml
function_groups:
  mcp_tools:
    _type: mcp_client
    server:
      transport: streamable-http
      url: "http://localhost:9901/mcp"
    include:
      - tool_a
      - tool_b
    tool_overrides:
      tool_a:
        alias: "tool_a_alias"
        description: "Tool A description"

workflows:
  _type: react_agent
  tool_names:
    - mcp_tools
```
You can use `mcp_client` function group to connect to a MCP server, dynamically discover the tools it serves, and register them as NeMo Agent toolkit functions.

The function group supports filtering via the `include` and `exclude` parameters. You can also optionally override the tool name and description defined by the MCP server via the `tool_overrides` parameter.

The function group can be directly referenced in the workflow configuration and provides all accessible tools from the MCP server to the workflow. Multiple function groups can be used in the same workflow to access tools from multiple MCP servers.

### `mcp_tool_wrapper` Configuration
```yaml
functions:
  mcp_tool_a:
    _type: mcp_tool_wrapper
    url: "http://localhost:9901/mcp"
    mcp_tool_name: tool_a
  mcp_tool_b:
    _type: mcp_tool_wrapper
    url: "http://localhost:9901/mcp"
    mcp_tool_name: tool_b

workflows:
  _type: react_agent
  tool_names:
    - mcp_tool_a
    - mcp_tool_b
```
You can use `mcp_tool_wrapper` to wrap a single MCP tool as a NeMo Agent toolkit function. You need to specify the server URL and the tool name. This configuration needs to be repeated for each tool you want to wrap.

## Transport
The MCP client can connect to MCP servers using different transport types. The choice of transport should match the server's configuration.

### Transport Types

- **`streamable-http`** (default): Modern HTTP-based transport, recommended for new deployments
- **`sse`**: Server-Sent Events transport, maintained for backwards compatibility
- **`stdio`**: Standard input/output transport for local process communication

### Streamable-HTTP Mode Configuration
For streamable-http mode, you only need to specify the server URL:

```yaml
functions:
  mcp_client:
    _type: mcp_client
    server:
      transport: streamable-http
      url: "http://localhost:8080/mcp"
```

### SSE Mode Configuration
SSE mode is supported for backwards compatibility with existing systems. It is recommended to use `streamable-http` mode instead.

```yaml
function_groups:
  mcp_tools:
    _type: mcp_client
    server:
      transport: sse
      url: "http://localhost:8080/sse"
```

### STDIO Mode Configuration
For STDIO mode, you need to specify the command to run and any additional arguments or environment variables:

```yaml
function_groups:
  github_mcp:
    _type: mcp_client
    server:
      transport: stdio
      command: "docker"
      args: [
        "run",
        "-i",
        "--rm",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: "${input:github_token}"
```

## Example
The simple calculator workflow can be configured to use local and remote MCP tools. Sample configuration is provided in the `config-mcp-date-stdio.yml` file.

`examples/MCP/simple_calculator_mcp/configs/config-mcp-date-stdio.yml`:
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
```

To run the simple calculator workflow using local and remote MCP tools, follow these steps:
1. Start the example remote MCP server.
```bash
nat mcp --config_file examples/getting_started/simple_calculator/configs/config.yml
```
This starts an MCP server on port 9901 with endpoint `/mcp` and uses `streamable-http` transport. This MCP server serves the calculator tools. See the [MCP Server](./mcp-server.md) documentation for more information.

2. Run the workflow using the `nat run` command.
```bash
nat run --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-date-stdio.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```
This configuration uses:
- a local MCP server to get the current date and time using stdio transport
- a remote MCP server to access the calculator tools using streamable-http transport.

## Displaying MCP Tools
The `nat info mcp` command can be used to list the tools served by an MCP server.
```bash
nat info mcp --url http://localhost:9901/mcp
```

Sample output:
```text
calculator_multiply
calculator_inequality
current_datetime
calculator_divide
calculator_subtract
react_agent
```

To get more detailed information about a specific tool, you can use the `--tool` flag.
```bash
nat info mcp --url http://localhost:9901/mcp --tool calculator_multiply
```
Sample output:
```text
Tool: calculator_multiply
Description: This is a mathematical tool used to multiply two numbers together. It takes 2 numbers as an input and computes their numeric product as the output.
Input Schema:
{
  "properties": {
    "text": {
      "description": "",
      "title": "Text",
      "type": "string"
    }
  },
  "required": [
    "text"
  ],
  "title": "CalculatorMultiplyInputSchema",
  "type": "object"
}
------------------------------------------------------------
```
