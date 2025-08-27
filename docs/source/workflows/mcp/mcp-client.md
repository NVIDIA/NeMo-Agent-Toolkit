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

You can use NeMo Agent toolkit as an MCP Client to connect to and use tools served by remote MCP servers.

This guide will cover how to use NeMo Agent toolkit as an MCP Client. For more information on how to use NeMo Agent toolkit as an MCP Server, please refer to the [MCP Server](./mcp-server.md) documentation.

## MCP Client Configuration

The MCP client can connect to MCP servers using different transport types. The choice of transport should match the server's configuration.

### Transport Types

- **`streamable-http`** (default): Modern HTTP-based transport, recommended for new deployments
- **`sse`**: Server-Sent Events transport, maintained for backwards compatibility
- **`stdio`**: Standard input/output transport for local process communication

## Usage
Tools served by remote MCP servers can be leveraged as NeMo Agent toolkit functions through configuration of an `mcp_tool_wrapper` or `mcp_client`.
- `mcp_tool_wrapper` is a simple configuration that allows you to connect to a MCP server and wrap a single tool as a NeMo Agent toolkit function.
- `mcp_client` is a more flexible configuration that allows you to connect to a MCP server, dynamically discover the tools it serves, and register them as NeMo Agent toolkit functions. Support for `mcp_client` is experimental.

### `mcp_tool_wrapper` Configuration
```python
class MCPToolConfig(FunctionBaseConfig, name="mcp_tool_wrapper"):
    """
    Function which connects to a Model Context Protocol (MCP) server and wraps the selected tool as a NeMo Agent toolkit
    function.
    """
    # Add your custom configuration parameters here
    url: HttpUrl | None = Field(default=None, description="The URL of the MCP server (for streamable-http or sse modes)")
    mcp_tool_name: str = Field(description="The name of the tool served by the MCP Server that you want to use")
    transport: Literal["sse", "stdio", "streamable-http"] = Field(default="streamable-http", description="The type of transport to use (default: streamable-http, backwards compatible with sse)")
    command: str | None = Field(default=None, description="The command to run for stdio mode (e.g. 'mcp-server')")
    args: list[str] | None = Field(default=None, description="Additional arguments for the stdio command")
    env: dict[str, str] | None = Field(default=None, description="Environment variables to set for the stdio process")
    description: str | None = Field(
        default=None,
        description="""
        Description for the tool that will override the description provided by the MCP server. Should only be used if
        the description provided by the server is poor or nonexistent
        """)
    return_exception: bool = Field(default=True,
                                   description="""
        If true, the tool will return the exception message if the tool call fails.
        If false, raise the exception.
        """)

```
In addition to the URL of the server, the configuration also takes as a parameter the name of the MCP tool you want to use as a NeMo Agent toolkit function. This is required because MCP servers can serve multiple tools, and for this wrapper we want to maintain a one-to-one relationship between NeMo Agent toolkit functions and MCP tools. This means that if you want to include multiple tools from an MCP server you will configure multiple `mcp_tool_wrappers`.

### 🧪 `mcp_client` Configuration (Experimental)

```python
class MCPServerConfig(BaseModel):
    """
    Server connection details for MCP client.
    Supports stdio, sse, and streamable-http transports.
    streamable-http is the recommended default for HTTP-based connections.
    """
    transport: Literal["stdio", "sse", "streamable-http"] = Field(
        ..., description="Transport type to connect to the MCP server (stdio, sse, or streamable-http)")
    url: HttpUrl | None = Field(default=None,
                                description="URL of the MCP server (for sse or streamable-http transport)")
    command: str | None = Field(default=None,
                                description="Command to run for stdio transport (e.g. 'python' or 'docker')")
    args: list[str] | None = Field(default=None, description="Arguments for the stdio command")
    env: dict[str, str] | None = Field(default=None, description="Environment variables for the stdio process")

class MCPClientConfig(FunctionBaseConfig, name="mcp_client"):
    """
    Configuration for connecting to an MCP server as a client and exposing selected tools.
    """
    server: MCPServerConfig = Field(..., description="Server connection details (transport, url/command, etc.)")
    tool_filter: dict[str, ToolOverrideConfig] | list[str] | None = Field(
        default=None,
        description="""Filter or map tools to expose from the server (list or dict).
        Can be:
        - A list of tool names to expose: ['tool1', 'tool2']
        - A dict mapping tool names to override configs:
          {'tool1': {'alias': 'new_name', 'description': 'New desc'}}
          {'tool2': {'description': 'Override description only'}}  # alias defaults to 'tool2'
        """)
```

`mcp_client` is a more flexible configuration that allows you to connect to a MCP server, dynamically discover the tools it serves, and register them as NeMo Agent toolkit functions. `mcp_client` can be used instead of `mcp_tool_wrapper` if you want to dynamically discover tools or if your transport type is `stdio`. `mcp_client` also supports filtering and overriding tool names and descriptions.

### Streamable-HTTP Mode Configuration
For streamable-http mode, you only need to specify the server URL and the tool name:

```yaml
functions:
  mcp_tool_a:
    _type: mcp_tool_wrapper
    transport: streamable-http
    url: "http://localhost:8080/mcp"
    mcp_tool_name: tool_a
  mcp_tool_b:
    _type: mcp_tool_wrapper
    transport: streamable-http
    url: "http://localhost:8080/mcp"
    mcp_tool_name: tool_b
```
You can use `mcp_client` instead of `mcp_tool_wrapper` if you want to dynamically discover tools:

```yaml
functions:
  mcp_client:
    _type: mcp_client
    server:
      transport: streamable-http
      url: "http://localhost:8080/mcp"
```

### SSE Mode Configuration
For SSE mode, you only need to specify the server URL and the tool name:

```yaml
functions:
  mcp_tool_a:
    _type: mcp_tool_wrapper
    transport: sse
    url: "http://localhost:8080/sse"
    mcp_tool_name: tool_a
  mcp_tool_b:
    _type: mcp_tool_wrapper
    transport: sse
    url: "http://localhost:8080/sse"
    mcp_tool_name: tool_b
```
SSE mode is supported for backwards compatibility with existing systems.

### 🧪 STDIO Mode Configuration (Experimental)
For STDIO mode, you need to specify the command to run and any additional arguments or environment variables:

```yaml
functions:
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
STDIO mode support is experimental. Note that you should use `mcp_client` instead of `mcp_tool_wrapper` as the function type for `stdio` mode. `mcp_client` allows you to connect to a MCP server, dynamically discover the tools it serves, and register them as NeMo Agent toolkit functions. See `examples/MCP/simple_calculator_mcp/configs/config-mcp-date-stdio.yml` for a complete example.

Once configured, a Pydantic input schema will be generated based on the input schema provided by the MCP server. This input schema is included with the configured function and is accessible by any agent or function calling the configured `mcp_tool_wrapper` function. The `mcp_tool_wrapper` function can accept the following type of arguments as long as they satisfy the input schema:
 * a validated instance of it's input schema
 * a string that represents a valid JSON
 * A python dictionary
 * Keyword arguments


## Example
The simple calculator workflow can be configured to use remote MCP tools. Sample configuration is provided in the `config-mcp-date.yml` file.

`examples/MCP/simple_calculator_mcp/configs/config-mcp-date.yml`:
```yaml
functions:
  mcp_time_tool:
    _type: mcp_tool_wrapper
    url: "http://localhost:8080/sse"
    mcp_tool_name: get_current_time
    description: "Returns the current date and time from the MCP server"
```

To run the simple calculator workflow using remote MCP tools, follow these steps:
1. Start the remote MCP server, `mcp-server-time`, by following the instructions in the `examples/MCP/simple_calculator_mcp/deploy_external_mcp/README.md` file. Check that the server is running by running the following command:
```bash
docker ps --filter "name=mcp-proxy-nat-time"
```
Sample output:
```
CONTAINER ID   IMAGE                      COMMAND                  CREATED      STATUS        PORTS                                       NAMES
4279653533ec   time_service-time_server   "mcp-proxy --pass-en…"   9 days ago   Up 41 hours   0.0.0.0:8080->8080/tcp, :::8080->8080/tcp   mcp-proxy-nat-time
```

2. Run the workflow using the `nat run` command.
```bash
nat run --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-date.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```
This will use the `mcp_time_tool` function to get the current hour of the day from the MCP server.

### 🧪 Using STDIO Mode (Experimental)
Alternatively, you can run the same example using stdio mode with the `config-mcp-date-stdio.yml` configuration:

```yaml
functions:
  mcp_time:
    _type: mcp_client
    server:
      transport: stdio
      command: "python"
      args: ["-m", "mcp_server_time", "--local-timezone=America/Los_Angeles"]
```

This configuration launches the MCP server directly as a `subprocess` instead of connecting to a running server. It dynamically discovers the tools served by the MCP server and registers them as NeMo Agent toolkit functions. Run it with:
```bash
nat run --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-date-stdio.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```
Ensure that MCP server time package is installed in your environment before running the workflow.
```bash
uv pip install mcp-server-time
```

## Displaying MCP Tools
The `nat info mcp` command can be used to list the tools served by an MCP server.
```bash
nat info mcp --url http://localhost:8080/sse
```

Sample output:
```
get_current_time
convert_time
```

To get more detailed information about a specific tool, you can use the `--tool` flag.
```bash
nat info mcp --url http://localhost:8080/sse --tool get_current_time
```
Sample output:
```
Tool: get_current_time
Description: Get current time in a specific timezones
Input Schema:
{
  "properties": {
    "timezone": {
      "description": "IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use 'UTC' as local timezone if no timezone provided by the user.",
      "title": "Timezone",
      "type": "string"
    }
  },
  "required": [
    "timezone"
  ],
  "title": "GetCurrentTimeInputSchema",
  "type": "object"
}
------------------------------------------------------------
```
