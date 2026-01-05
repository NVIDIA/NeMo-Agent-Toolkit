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

# NVIDIA NeMo Agent Toolkit as an MCP Client

Model Context Protocol (MCP) is an open protocol developed by Anthropic that standardizes how applications provide context to [LLMs](./llms/index.md). You can read more about MCP [here](https://modelcontextprotocol.io/introduction).

You can create a [workflow](./about-building-workflows.md) that uses MCP [tools](./functions-and-function-groups/functions.md#agents-and-tools) as [functions](./functions-and-function-groups/functions.md). In this case, the workflow acts as an MCP host and creates MCP clients to connect to MCP servers and use their tools as functions.

This guide covers how to use a NeMo Agent toolkit workflow as an MCP host with one or more MCP clients. For more information on how to use the NeMo Agent toolkit as an MCP server, refer to [MCP Server](../run-workflows/mcp-server.md).

## Installation

MCP client functionality requires the `nvidia-nat-mcp` package. Install it with:

```bash
uv pip install "nvidia-nat[mcp]"
```
## Accessing Protected MCP Servers
NeMo Agent toolkit can access protected MCP servers through the MCP client auth provider. For more information, refer to [MCP Authentication](../components/auth/mcp-auth/index.md).

## MCP Client Configuration
NeMo Agent toolkit enables workflows to use MCP tools as functions. The library handles the MCP server connection, tool discovery, and function registration. This allows the workflow to use MCP tools as regular functions.

Tools served by remote MCP servers can be used as NeMo Agent toolkit functions using `mcp_client`, a flexible configuration using [function groups](./functions-and-function-groups/function-groups.md) that allows you to connect to an MCP server, dynamically discover the tools it serves, and register them as NeMo Agent toolkit functions.

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

workflow:
  _type: react_agent
  tool_names:
    - mcp_tools
```
You can use the `mcp_client` function group to connect to an MCP server, dynamically discover the tools it serves, and register them as NeMo Agent toolkit functions.

The function group supports filtering using the `include` and `exclude` parameters. You can also optionally override the tool name and description defined by the MCP server using the `tool_overrides` parameter.

The function group can be directly referenced in the workflow configuration and provides all accessible tools from the MCP server to the workflow. Multiple function groups can be used in the same workflow to access tools from multiple MCP servers. Refer to [Function Groups](./functions-and-function-groups/function-groups.md) for more information about function group capabilities.

A tool within a function group can also be referenced by its name using the following syntax: `<function_group_name>.<tool_name>`.

:::{note}
This requires that the tool name is explicitly listed under the optional `include` list of the function group configuration.

See [function group accessibility](./functions-and-function-groups/function-groups.md#understanding-function-accessibility) for more details.
:::
Example:
```yaml
workflows:
  _type: react_agent
  tool_names:
    - mcp_tools.tool_a
```

An additional case to note is when a function group is served by an MCP server, the tools within the function group must still be accessed by their full name. This is the same as the prior case, but there is an important difference. Consider the following example:
```yaml
workflow:
  _type: react_agent
  tool_names:
    - mcp_tools__calculator__add
```

`mcp_tools` is the name of the function group, and `calculator__add` is the name of the tool within the function group. This is because the tools are added to the function group as functions, and the function group is then added to the workflow as a tool.

#### Configuration Options

The `mcp_client` function group supports the following configuration options:

**Note**: You can get the complete list of configuration options and their schemas by running:
```bash
nat info components -t function_group -q mcp_client
```

##### Server Configuration

- `server.transport`: Transport type (`stdio`, `sse`, or `streamable-http`). Refer to [Transport Configuration](#transport-configuration) for details.
- `server.url`: URL of the MCP server (required for `sse` and `streamable-http` transports)
- `server.command`: Command to run for `stdio` transport, such as `python` or `docker`
- `server.args`: Arguments for the stdio command
- `server.env`: Environment variables for the stdio process
- `server.auth_provider`: Reference to [authentication provider](../components/auth/api-authentication.md) for protected MCP servers (only supported with `streamable-http` transport)

##### Timeout Configuration

- `tool_call_timeout`: Timeout for MCP tool calls. Defaults to `60` seconds
- `auth_flow_timeout`: Timeout for interactive authentication flow. Defaults to `300` seconds

##### Reconnection Configuration

- `reconnect_enabled`: Whether to enable reconnecting to the MCP server if the connection is lost. Defaults to `true`.
- `reconnect_max_attempts`: Maximum number of reconnect attempts. Defaults to `2`.
- `reconnect_initial_backoff`: Initial backoff time for reconnect attempts. Defaults to `0.5` seconds.
- `reconnect_max_backoff`: Maximum backoff time for reconnect attempts. Defaults to `50.0` seconds.

##### Session Management Configuration

- `max_sessions`: Maximum number of concurrent session clients. Defaults to `100`.
- `session_idle_timeout`: Time after which inactive sessions are cleaned up. Defaults to `1 hour`.

##### Tool Customization

- `tool_overrides`: Optional overrides for tool names and descriptions. Each entry can specify:
  - `alias`: Override the tool name (function name in the workflow)
  - `description`: Override the tool description

Example with all options:

```yaml
function_groups:
  mcp_tools:
    _type: mcp_client
    include:
      - calculator__add
      - calculator__multiply
    server:
      transport: streamable-http
      url: "http://localhost:9901/mcp"
      auth_provider: "mcp_oauth2"  # Optional authentication
    tool_call_timeout: 60  # 1 minute for tool calls
    auth_flow_timeout: 300  # 5 minutes for auth flow
    reconnect_enabled: true
    reconnect_max_attempts: 3
    reconnect_initial_backoff: 1.0
    reconnect_max_backoff: 60.0
    max_sessions: 50  # Maximum concurrent sessions
    session_idle_timeout: 7200  # 2 hours (in seconds)
    tool_overrides:
      calculator__add:
        alias: "add_numbers"
        description: "Add two numbers together"
      calculator__multiply:
        description: "Multiply two numbers"  # Keeps original name
```

## Transport Configuration
The `mcp_client` function group can connect to MCP servers using different transport types. Choose the transport that matches your MCP server's configuration to ensure proper communication.

### Transport Types

- **`streamable-http`** (default): Modern HTTP-based transport, recommended for new deployments
- **`SSE`**: Server-Sent Events transport, maintained for backwards compatibility
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
SSE mode is supported for backward compatibility with existing systems. It is recommended to use `streamable-http` mode instead.

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
This starts an MCP server on port 9901 with endpoint `/mcp` and uses `streamable-http` transport. Refer to [MCP Server](../run-workflows/mcp-server.md) for more information.

2. Run the workflow:
```bash
nat run --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-client.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```

## Displaying MCP Tools using the CLI

Use the `nat mcp client` commands to inspect and call tools available from an MCP server before configuring your workflow. This is useful for discovering available tools and understanding their input schemas.

### List All Tools

To list all tools served by an MCP server:

```bash
# For streamable-http transport (default)
nat mcp client tool list --url http://localhost:9901/mcp

# For stdio transport
nat mcp client tool list --transport stdio --command "python" --args "-m mcp_server_time"

# For SSE transport
nat mcp client tool list --url http://localhost:9901/sse --transport sse
```
For SSE transport, ensure the MCP server starts with the `--transport sse` flag. The transport type on the client and server needs to match for MCP communication to work. The default transport type is `streamable-http`.

Sample output:
```text
calculator__add
calculator__multiply
calculator__subtract
calculator__divide
calculator__compare
current_datetime
react_agent
```

### Get Tool Details

To get detailed information about a specific tool, use the `--tool` flag:

```bash
nat mcp client tool list --url http://localhost:9901/mcp --tool calculator__multiply
```

Sample output:
```text
Tool: calculator__multiply
Description: Multiply two or more numbers together
Input Schema:
{
  "properties": {
    "numbers": {
      "description": "",
      "items": {
        "type": "number"
      },
      "title": "Numbers",
      "type": "array"
    }
  },
  "required": [
    "numbers"
  ],
  "title": "Calculator__MultiplyInputSchema",
  "type": "object"
}
```
### Call a Tool

To call a tool and get its output:

```console
# Pass arguments as JSON
$ nat mcp client tool call calculator__multiply \
  --url http://localhost:9901/mcp \
  --json-args '{"numbers": [1, 3, 6, 10]}'
180.0
```

### Using Protected MCP Servers

To use a protected MCP server, you need to provide the `--auth` flag:
```bash
nat mcp client tool list --url http://example.com/mcp --auth
```
This will use the `mcp_oauth2` authentication provider to authenticate the user. For more information, refer to [MCP Authentication](../components/auth/mcp-auth/index.md).

## List MCP Client Tools using the HTTP endpoint
This is useful when you want to inspect the tools configured on the client side and whether each tool is available on the connected server.

When you serve a workflow that includes an `mcp_client` function group, the NeMo Agent toolkit exposes an HTTP endpoint to inspect the tools configured on the client side and whether each tool is available on the connected server.

### Steps

1. Start the MCP server:
   ```bash
   nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml
   ```

2. Start the workflow (MCP client) with FastAPI:
   ```bash
   nat serve --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-client.yml
   ```

3. Call the endpoint and pretty-print the response:
   ```bash
   curl -s http://localhost:8000/mcp/client/tool/list | jq
   ```

### Endpoint

- Path: `/mcp/client/tool/list`
- Method: `GET`
- Purpose: Returns tools configured in each `mcp_client` function group, indicates whether each tool is available on the connected MCP server, and includes metadata about the function group and HTTP session.

### Sample Output

```json
{
  "mcp_clients": [
    {
      "function_group": "mcp_time",
      "server": "stdio:python",
      "transport": "stdio",
      "session_healthy": true,
      "protected": false,
      "tools": [
        {
          "name": "convert_time",
          "description": "Convert time between timezones",
          "server": "stdio:python",
          "available": true
        },
        {
          "name": "get_current_time_mcp_tool",
          "description": "Returns the current date and time",
          "server": "stdio:python",
          "available": true
        }
      ],
      "total_tools": 2,
      "available_tools": 2
    },
    {
      "function_group": "mcp_math",
      "server": "streamable-http:http://localhost:9901/mcp",
      "transport": "streamable-http",
      "session_healthy": true,
      "protected": false,
      "tools": [
        {
          "name": "calculator__add",
          "description": "Add two or more numbers together",
          "server": "streamable-http:http://localhost:9901/mcp",
          "available": true
        },
        {
          "name": "calculator__compare",
          "description": "Compare two numbers",
          "server": "streamable-http:http://localhost:9901/mcp",
          "available": true
        },
        {
          "name": "calculator__divide",
          "description": "Divide one number by another",
          "server": "streamable-http:http://localhost:9901/mcp",
          "available": true
        },
        {
          "name": "calculator__multiply",
          "description": "Multiply two or more numbers together",
          "server": "streamable-http:http://localhost:9901/mcp",
          "available": true
        },
        {
          "name": "calculator__subtract",
          "description": "Subtract one number from another",
          "server": "streamable-http:http://localhost:9901/mcp",
          "available": true
        }
      ],
      "total_tools": 5,
      "available_tools": 5
    }
  ]
}
```

## MCP Inspection via UI
You can inspect the MCP tools available on the client side using the UI.

### Steps

1. Start the MCP server:
   ```bash
   nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml
   ```

2. Start the workflow (MCP client) with FastAPI:
   ```bash
   nat serve --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-client.yml
   ```

3. Launch the UI by following the instructions in the [Launching the UI](../run-workflows/launching-ui.md) documentation.

4. Click on the MCP tab in the side panel to inspect the MCP tools available on the client side.

### Sample Output
![MCP Side Panel](../_static/mcp_side_panel.png)

![MCP Tools](../_static/mcp_tools.png)


### Troubleshooting

If you encounter connection issues:
- Verify the MCP server is running and accessible using the example `nat mcp client ping` command:
  ```bash
  nat mcp client ping --url http://localhost:9901/mcp
  ```
- Check that the transport type matches the server configuration
- Ensure the URL or command is correct
- Check network connectivity for remote servers
