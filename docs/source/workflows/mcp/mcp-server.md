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

This guide will cover how to use NeMo Agent toolkit as an MCP Server to publish tools using MCP. For more information on how to use NeMo Agent toolkit as an MCP Host with one or more MCP Clients, refer to the [MCP Client](./mcp-client.md) documentation.

## MCP Server Usage

The `nat mcp serve` command can be used to start an MCP server that publishes the functions from your workflow as MCP tools.

To start an MCP server publishing all tools from your workflow, run the following command:

```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml
```

This will load the workflow configuration from the specified file, start an MCP server on the default host (localhost) and port (9901), and publish all tools from the workflow as MCP tools. The MCP server is available at `http://localhost:9901/mcp` using streamable-http transport.

You can also use the `sse` (Server-Sent Events) transport for backwards compatibility through the `--transport` flag, for example:
```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml --transport sse
```
With this configuration, the MCP server is available at `http://localhost:9901/sse` using SSE transport.

:::{warning}
**SSE Transport Security Limitations**: The SSE transport does not support authentication. For production deployments, use `streamable-http` transport with authentication configured. SSE should only be used for local development on localhost or behind an authenticating reverse proxy.
:::

You can optionally specify the server settings using the following flags:
```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --host 0.0.0.0 \
  --port 9901 \
  --name "My MCP Server"
```

### Filtering MCP Tools
You can specify a filter to only publish a subset of tools from the workflow.

```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --tool_names calculator_multiply \
  --tool_names calculator_divide \
  --tool_names calculator_subtract \
  --tool_names calculator_inequality
```

## Displaying MCP Tools published by an MCP server

To list the tools published by the MCP server you can use the `nat mcp client tool list` command. This command acts as an MCP client and connects to the MCP server running on the specified URL (defaults to `http://localhost:9901/mcp` for streamable-http, with backwards compatibility for `http://localhost:9901/sse`).

**Note:** The `nat mcp client` commands require the `nvidia-nat-mcp` package. If you encounter an error about missing MCP client functionality, install it with `uv pip install "nvidia-nat[mcp]"`.

```bash
nat mcp client tool list
```

Sample output:
```
calculator_multiply
calculator_inequality
calculator_divide
calculator_subtract
```

### Debug route for listing tools (no MCP client required)
You can also inspect the tools exposed by the MCP server without an MCP client by using the debug route:

```bash
curl -s http://localhost:9901/debug/tools/list | jq
```

This returns a JSON list of tools with names and descriptions.

You can request one or more specific tools by name. The `name` parameter accepts repeated values or a comma‑separated list. When `name` is provided, detailed schemas are returned by default:

```bash
# Single tool (detailed by default)
curl -s "http://localhost:9901/debug/tools/list?name=calculator_multiply" | jq

# Multiple tools (detailed by default)
curl -s "http://localhost:9901/debug/tools/list?name=calculator_multiply&name=calculator_divide" | jq

# Comma-separated list (equivalent)
curl -s "http://localhost:9901/debug/tools/list?name=calculator_multiply,calculator_divide" | jq
```

The response includes the tool's name, description, and its input schema by default. For tools that accept a chat‑style input, the schema is simplified as a single `query` string parameter to match the exposed MCP interface.

You can control the amount of detail using the `detail` query parameter:

- When requesting specific tool(s) with `name`, detailed schema is returned by default. Pass `detail=false` to suppress schemas:

```bash
curl -s "http://localhost:9901/debug/tools/list?name=calculator_multiply&detail=false" | jq
```

- When listing all tools (without `name`), the default output is simplified. Pass `detail=true` to include schemas for each tool:

```bash
curl -s "http://localhost:9901/debug/tools/list?detail=true" | jq
```

To get more information about a specific tool, use the `--detail` flag or the `--tool` flag followed by the tool name.

```bash
nat mcp client tool list --tool calculator_multiply
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
2. Run the simple calculator workflow with the `config-mcp-client.yml` config file. NeMo Agent toolkit will act as an MCP client and connect to the MCP server started in the previous step to access the remote tools.
```bash
nat run --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-client.yml --input "Is 2 times 2 greater than the current hour?"
```

## Verifying MCP Server Health
You can verify the health of the MCP using the `/health` route or the `nat mcp client ping` command.

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

### Using the `nat mcp client ping` command
You can also test if an MCP server is responsive and healthy using the `nat mcp client ping` command:
```bash
nat mcp client ping --url http://localhost:9901/mcp
```

Sample output:
```
Server at http://localhost:9901/mcp is healthy (response time: 4.35ms)
```
This is useful for health checks and monitoring.

## Security Considerations

### Authentication Limitations
- The `nat mcp serve` command currently starts an MCP server without built-in authentication. Server-side authentication is planned for a future release.
- NeMo Agent toolkit workflows can still connect to protected third-party MCP servers through the MCP client auth provider. See the [MCP Authentication](./mcp-auth.md) documentation for more information.

### Local Development
For local development, you can use `localhost` or `127.0.0.1` as the host (default). This limits access to your local machine only.

### Production Deployment
For production environments:
- Run `nat mcp serve` behind a trusted network or an authenticating reverse proxy with HTTPS (OAuth2, JWT, or mTLS)
- Do not expose the server directly to the public Internet
- Do not bind to non-localhost addresses (such as `0.0.0.0` or public IP addresses) without authentication

If you bind the MCP server to a non-localhost address without configuring authentication, the server will log a warning. This configuration exposes your server to unauthorized access.
