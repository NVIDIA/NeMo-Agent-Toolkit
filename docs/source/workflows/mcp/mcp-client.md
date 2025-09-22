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

You can create a workflow that uses MCP tools as functions. In this case the workflow acts as a MCP host and creates MCP clients to connect to MCP servers and use their tools as functions.

This guide will cover how to use a NeMo Agent toolkit workflow as a MCP host with one or more MCP clients. For more information on how to use NeMo Agent toolkit as an MCP Server, please refer to the [MCP Server](./mcp-server.md) documentation.

## Installation

MCP client functionality requires the `nvidia-nat-mcp` package. Install it with:

```bash
uv pip install nvidia-nat[mcp]
```

## MCP Client Configuration
NeMo Agent toolkit enables workflows to use MCP tools as functions. The library handles the MCP server connection, tool discovery, and function registration. This allow the workflow to use MCP tools as regular functions.

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
You can use the `mcp_client` function group to connect to an MCP server, dynamically discover the tools it serves, and register them as NeMo Agent toolkit functions.

The function group supports filtering via the `include` and `exclude` parameters. You can also optionally override the tool name and description defined by the MCP server via the `tool_overrides` parameter.

The function group can be directly referenced in the workflow configuration and provides all accessible tools from the MCP server to the workflow. Multiple function groups can be used in the same workflow to access tools from multiple MCP servers. See [Function Groups](../function-groups.md) for more information about function group capabilities.

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
You can use `mcp_tool_wrapper` to wrap a single MCP tool as a NeMo Agent toolkit function. Specify the server URL and the tool name for each tool you want to wrap. This approach requires a separate configuration entry for each individual tool.

## Transport Configuration
The `mcp_client` function group can connect to MCP servers using different transport types. Choose the transport that matches your MCP server's configuration to ensure proper communication.

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
The following example demonstrates how to use the `mcp_client` function group with both local and remote MCP servers. This configuration shows how to use multiple MCP servers with different transports in the same workflow.

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

workflows:
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
This starts an MCP server on port 9901 with endpoint `/mcp` and uses `streamable-http` transport. See the [MCP Server](./mcp-server.md) documentation for more information.

2. Run the workflow:
```bash
nat run --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-date-stdio.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```

## Displaying MCP Tools

Use the `nat mcp client` commands to inspect and call tools available from an MCP server before configuring your workflow. This is useful for discovering available tools and understanding their input schemas.

### List All Tools

To list all tools served by an MCP server:

```bash
# For streamable-http transport (default)
nat mcp client tool list --url http://localhost:9901/mcp

# For stdio transport
nat mcp client tool list --transport stdio --command "python" --args "-m mcp_server_time"

# For sse transport
nat mcp client tool list --url http://localhost:9901/sse --transport sse
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

### Get Tool Details

To get detailed information about a specific tool, use the `--tool` flag:

```bash
nat mcp client tool list --url http://localhost:9901/mcp --tool calculator_multiply
```

### Call a Tool

To call a tool and see its output:

```bash
# Pass arguments as JSON
nat mcp client tool call calculator_multiply \
  --url http://localhost:9901/mcp \
  --json-args '{"a": 2, "b": 3}'
```

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

### Troubleshooting

If you encounter connection issues:
- Verify the MCP server is running and accessible via the `nat mcp client ping` command
- Check that the transport type matches the server configuration
- Ensure the URL or command is correct
- Check network connectivity for remote servers
