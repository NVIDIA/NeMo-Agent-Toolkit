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

# NeMo Agent Toolkit as an MCP Server

Model Context Protocol (MCP) is an open protocol developed by Anthropic that standardizes how applications provide context to LLMs. You can read more about MCP [here](https://modelcontextprotocol.io/introduction).

This guide will cover how to use NeMo Agent toolkit as an MCP Server to publish tools using MCP. For more information on how to use NeMo Agent toolkit as an MCP Client, refer to the [MCP Client](./mcp-client.md) documentation.

## MCP Server Usage

The `nat mcp` command can be used to start an MCP server that publishes the functions from your workflow as MCP tools.

To start an MCP server publishing all tools from your workflow, run the following command:

```bash
nat mcp --config_file examples/getting_started/simple_calculator/configs/config.yml
```

This will load the workflow configuration from the specified file, start an MCP server on the default host (localhost) and port (9901), and publish all tools from the workflow as MCP tools. The MCP server is available at `http://localhost:9901/mcp` using streamable-http transport.

You can also use the `sse` (Server-Sent Events) transport for backwards compatibility via the `--transport` flag for example:
```bash
nat mcp --config_file examples/getting_started/simple_calculator/configs/config.yml --transport sse
```
With this configuration, the MCP server is available at `http://localhost:9901/sse` using SSE transport.

You can optionally specify the server settings using the following flags:
```bash
nat mcp --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --host 0.0.0.0 \
  --port 9901 \
  --name "My MCP Server"
```

### Filtering MCP Tools
You can specify a filter to only publish a subset of tools from the workflow.

```bash
nat mcp --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --tool_names calculator_multiply \
  --tool_names calculator_divide \
  --tool_names calculator_subtract \
  --tool_names calculator_inequality
```

## Displaying MCP Tools published by an MCP server

To list the tools published by the MCP server you can use the `nat info mcp` command. This command acts as a MCP client and connects to the MCP server running on the specified URL (defaults to `http://localhost:9901/mcp` for streamable-http, with backwards compatibility for `http://localhost:9901/sse`).

```bash
nat info mcp
```

Sample output:
```
calculator_multiply
calculator_inequality
calculator_divide
calculator_subtract
```

To get more information about a specific tool, use the `--detail` flag or the `--tool` flag followed by the tool name.

```bash
nat info mcp --tool calculator_multiply
```

Sample output:
```
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
## Integration with MCP Clients

The NeMo Agent toolkit MCP front-end implements the Model Context Protocol specification, making it compatible with any MCP client. This allows for seamless integration with various systems that support MCP, including:

- MCP-compatible LLM frameworks
- Other agent frameworks that support MCP
- Custom applications including NeMo Agent toolkit applications that implement the MCP client specification

### Example
In this example, we will use NeMo Agent toolkit as both a MCP client and a MCP server.

1. Start the MCP server by following the instructions in the [MCP Server Usage](#mcp-server-usage) section. NeMo Agent toolkit will act as an MCP server and publish the calculator tools as MCP tools.
2. Run the simple calculator workflow with the `config-mcp-math.yml` config file. NeMo Agent toolkit will act as an MCP client and connect to the MCP server started in the previous step to access the remote tools.
```bash
nat run --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-math.yml --input "Is 2 times 2 greater than the current hour?"
```

The functions in `config-mcp-math.yml` are configured to use the calculator tools published by the MCP server running on `http://localhost:9901/mcp` using streamable-http transport.
`examples/MCP/simple_calculator_mcp/configs/config-mcp-math.yml`:
```yaml
functions:
  calculator_multiply:
    _type: mcp_tool_wrapper
    url: "http://localhost:9901/mcp"
    transport: "streamable-http"
    mcp_tool_name: calculator_multiply
    description: "Returns the product of two numbers"
  calculator_inequality:
    _type: mcp_tool_wrapper
    url: "http://localhost:9901/mcp"
    transport: "streamable-http"
    mcp_tool_name: calculator_inequality
    description: "Returns the inequality of two numbers"
  calculator_divide:
    _type: mcp_tool_wrapper
    url: "http://localhost:9901/mcp"
    transport: "streamable-http"
    mcp_tool_name: calculator_divide
    description: "Returns the quotient of two numbers"
  current_datetime:
    _type: current_datetime
  calculator_subtract:
    _type: mcp_tool_wrapper
    url: "http://localhost:9901/mcp"
    transport: "streamable-http"
    mcp_tool_name: calculator_subtract
    description: "Returns the difference of two numbers"
```
In this example, the `calculator_multiply`, `calculator_inequality`, `calculator_divide`, and `calculator_subtract` tools are remote MCP tools. The `current_datetime` tool is a local NeMo Agent toolkit tool.


## Verifying MCP Server Health
You can verify the health of the MCP using the `/health` route or the `nat info mcp ping` command.

### Using the `/health` route
The MCP server exposes a `/health` route that can be used to verify the health of the MCP server.

```bash
curl -s http://localhost:9901/health | jq
```

Sample output:
```json
{
  "status": "healthy",
  "error": null,
  "server_name": "NAT MCP"
}
```

### Using the `nat info mcp ping` command
You can also test if an MCP server is responsive and healthy using the `nat info mcp ping` command:
```bash
nat info mcp ping --url http://localhost:9901/mcp
```

Sample output:
```
Server at http://localhost:9901/mcp is healthy (response time: 4.35ms)
```
This is useful for health checks and monitoring.
