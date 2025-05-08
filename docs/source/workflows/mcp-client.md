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
# Model Context Protocol Integration
Model Context Protocol (MCP) is an open protocol developed by Anthropic that standardizes how applications provide context to LLMs. You can read more about MCP [here](https://modelcontextprotocol.io/introduction). AIQ Toolkit implements an MCP Client Tool which allows AIQ Toolkit workflows and functions to connect to and use tools served by remote MCP servers using server sent events.

## Usage
Tools served by remote MCP servers can be leveraged as AIQ Toolkit functions through configuration of an `mcp_tool_wrapper`.

```python
class MCPToolConfig(FunctionBaseConfig, name="mcp_tool_wrapper"):
    """
    Function which connects to a Model Context Protocol (MCP) server and wraps the selected tool as an AIQ function.
    """
    # Add your custom configuration parameters here
    url: HttpUrl = Field(description="The URL of the MCP server")
    mcp_tool_name: str = Field(description="The name of the tool served by the MCP Server that you want to use")
    description: str | None = Field(
        default=None,
        description="""
        Description for the tool that will override the description provided by the MCP server. Should only be used if
        the description provided by the server is poor or nonexistent
        """
    )
```
In addition to the URL of the server, the configuration also takes as a parameter the name of the MCP tool you want to use as an AIQ Toolkit function. This is required because MCP servers can serve multiple tools, and for this wrapper we want to maintain a one-to-one relationship between AIQ Toolkit functions and MCP tools. This means that if you want to include multiple tools from an MCP server you will configure multiple `mcp_tool_wrappers`.

For example:

```yaml
functions:
  mcp_tool_a:
    _type: mcp_tool_wrapper
    url: "http://0.0.0.0:8080/sse"
    mcp_tool_name: tool_a
  mcp_tool_b:
    _type: mcp_tool_wrapper
    url: "http://0.0.0.0:8080/sse"
    mcp_tool_name: tool_b
  mcp_tool_c:
    _type: mcp_tool_wrapper
    url: "http://0.0.0.0:8080/sse"
    mcp_tool_name: tool_c
```

The final configuration parameter (the `description`) is optional, and should only be used if the description provided by the MCP server is not sufficient, or if there is no description provided by the server.

Once configured, a Pydantic input schema will be generated based on the input schema provided by the MCP server. This input schema is included with the configured function and is accessible by any agent or function calling the configured `mcp_tool_wrapper` function. The `mcp_tool_wrapper` function can accept the following type of arguments as long as they satisfy the input schema:
 * a validated instance of it's input schema
 * a string that represents a valid JSON
 * A python dictionary
 * Keyword arguments
