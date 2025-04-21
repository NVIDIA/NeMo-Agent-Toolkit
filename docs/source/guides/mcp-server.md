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

# AgentIQ MCP Front-End

Model Context Protocol (MCP) is an open protocol developed by Anthropic that standardizes how applications provide context to LLMs. You can read more about MCP [here](https://modelcontextprotocol.io/introduction). 

The MCP front-end in AgentIQ allows you to expose your workflow functions as MCP-compatible tools that can be used by any MCP client. This enables seamless integration between AgentIQ workflows and other MCP-compatible systems.

## Using the MCP Front-End

The MCP front-end is invoked using the `aiq mcp` command (which is an alias for `aiq start mcp`). This command starts an MCP server that exposes the functions from your workflow as MCP tools.

### Configuration

The MCP front-end can be configured with the following options:

```
--config_file FILE         A JSON/YAML file with the workflow configuration (required)
--override <TEXT TEXT>...  Override config values using dot notation
--name TEXT                Name of the MCP server
--host TEXT                Host to bind the server to
--port INTEGER             Port to bind the server to
--debug BOOLEAN            Enable debug mode
--log_level TEXT           Log level for the MCP server
--tool_names TEXT          Comma-separated list of tool names to expose
```

### Example Usage

To start an MCP server exposing a specific tool from your workflow:

```bash
aiq mcp --config_file examples/agents/mixture_of_agents/configs/config.yml --tool_names math_agent
```

This will:
1. Load the workflow configuration from the specified file
2. Start an MCP server on the default host (localhost) and port (9901)
3. Expose only the `math_agent` function from the workflow as an MCP tool

### Accessing the MCP Tools

Once the MCP server is running, any MCP-compatible client can connect to it and use the exposed tools. For example, an MCP client could connect to `http://localhost:9901/sse` and use the `math_agent`.

## Integration with MCP Clients

The AgentIQ MCP front-end implements the Model Context Protocol specification, making it compatible with any MCP client. This allows for seamless integration with various systems that support MCP, including:

1. MCP-compatible LLM frameworks
2. Other agent frameworks that support MCP
3. Custom applications that implement the MCP client specification
