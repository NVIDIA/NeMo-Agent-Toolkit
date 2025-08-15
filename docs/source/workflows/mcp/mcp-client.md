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

## Usage
Tools served by remote MCP servers can be leveraged as NeMo Agent toolkit functions through configuration of an `mcp_tool_wrapper`.

```python
class MCPToolConfig(FunctionBaseConfig, name="mcp_tool_wrapper"):
    """
    Function which connects to a Model Context Protocol (MCP) server and wraps the selected tool as a NeMo Agent toolkit
    function.
    """
    # Add your custom configuration parameters here
    url: HttpUrl = Field(description="The URL of the MCP server")
    mcp_tool_name: str = Field(description="The name of the tool served by the MCP Server that you want to use")
    description: str | None = Field(default=None,
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

For example:

```yaml
functions:
  mcp_tool_a:
    _type: mcp_tool_wrapper
    url: "http://localhost:8080/sse"
    mcp_tool_name: tool_a
  mcp_tool_b:
    _type: mcp_tool_wrapper
    url: "http://localhost:8080/sse"
    mcp_tool_name: tool_b
  mcp_tool_c:
    _type: mcp_tool_wrapper
    url: "http://localhost:8080/sse"
    mcp_tool_name: tool_c
```

The optional configuration parameters (`description` and `return_exception`) provide additional control over the tool behavior. The `description` parameter should only be used if the description provided by the MCP server is not sufficient, or if there is no description provided by the server. The `return_exception` parameter controls whether exceptions are returned as messages or raised directly.

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
