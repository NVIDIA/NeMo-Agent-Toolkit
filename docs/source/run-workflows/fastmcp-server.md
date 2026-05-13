<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NVIDIA NeMo Agent Toolkit as an MCP Server using FastMCP

Model Context Protocol (MCP) is an open protocol developed by Anthropic that standardizes how applications provide context to [LLMs](../build-workflows/llms/index.md). This guide covers how to run NVIDIA NeMo Agent Toolkit workflows as an MCP server using the FastMCP server runtime.

## Decision

NeMo Agent Toolkit supports two MCP server runtimes. Both publish the workflow and its tools as MCP tools. Choose the runtime that matches your deployment stack and MCP server policy of the organization:

- Use `nat mcp serve` for the [MCP SDK server runtime](https://github.com/modelcontextprotocol/python-sdk).
- Use `nat fastmcp server run` for the [FastMCP server runtime](https://github.com/jlowin/fastmcp).
- For the MCP SDK server guide, see [NeMo Agent Toolkit as an MCP Server](./mcp-server.md).
- MCP client commands and configuration require the MCP SDK package (`nvidia-nat-mcp`).


:::{warning}
The `nvidia-nat-fastmcp` package depends on the beta release of FastMCP3 and is not recommended for production use. This warning will be removed when FastMCP3 is generally available.
:::

## Installation

Install the `nvidia-nat-fastmcp` package:

```bash
uv pip install nvidia-nat-fastmcp
```

For MCP client commands and configuration, install the `nvidia-nat-mcp` package:

```bash
uv pip install nvidia-nat-mcp
```

## FastMCP Server Usage

Use `nat fastmcp server run` to start an MCP server using the FastMCP server runtime and publish workflow tools.

```bash
nat fastmcp server run --config_file examples/getting_started/simple_calculator/configs/config.yml
```

This starts an MCP server using the FastMCP server runtime on the default host (`localhost`) and port (`9902`) and publishes all workflow tools at `http://localhost:9902/mcp` using streamable-http transport.

You can also specify server settings with CLI flags:

```bash
nat fastmcp server run --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --host 0.0.0.0 \
  --port 9902 \
  --name "My FastMCP Server"
```

### Using Developer Mode

Use `nat fastmcp server dev` to restart the server when files change. This is useful when you iterate on workflow code or configuration.

```bash
nat fastmcp server dev --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --watch-path examples/getting_started/simple_calculator/src
```

By default, developer mode ignores common noisy files such as `*.log`, `*.tmp`, and `*.temp`.
To further control which changes trigger reloads, use include and exclude globs:

- `--reload-include-glob` narrows reloads to matching paths.
- `--reload-exclude-glob` removes matches from that set.
- When include globs are provided, they take precedence over default excludes.

```bash
nat fastmcp server dev --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --watch-path examples/getting_started/simple_calculator/src \
  --reload-include-glob "*.py" \
  --reload-include-glob "*.yml" \
  --reload-exclude-glob "*.log"
```

### Generating MCP Client Configuration Snippets

Use `nat fastmcp server install` to generate MCP client configuration snippets for a FastMCP server. This command does not modify your environment.

```bash
nat fastmcp server install cursor --url http://localhost:9902/mcp
```
Sample output:
```json
{
  "mcpServers": {
    "mcp_server": {
      "transport": "streamable-http",
      "url": "http://localhost:9902/mcp"
    }
  }
}
```

To generate a MCP client configuration YAML snippet for a workflow configuration:

```bash
nat fastmcp server install nat-workflow --url http://localhost:9902/mcp --name mcp_math
```
Sample output:
```yaml
function_groups:
  mcp_math:
    _type: per_user_mcp_client
    server:
      transport: streamable-http
      url: http://localhost:9902/mcp
```

For a full command reference, see [Command Line Interface](../reference/cli.md).

### Filtering FastMCP Tools

You can publish a subset of tools using the `--tool_names` flag:

```bash
nat fastmcp server run --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --tool_names calculator__multiply \
  --tool_names calculator__divide
```

### Mounting at Custom Paths

To mount the server at a custom base path, set `base_path` in the configuration file:

```yaml
general:
  front_end:
    _type: fastmcp
    name: "my_fastmcp_server"
    base_path: "/api/v1"
```

With this configuration, the MCP server is accessible at `http://localhost:9902/api/v1/mcp`.

## Inspecting and Running MCP Tools Published by a FastMCP Server

Use `nat mcp client` to inspect and run tools exposed by an MCP server using the FastMCP server runtime.

**Note:** The `nat mcp client` commands require the `nvidia-nat-mcp` package. If you encounter an error about missing MCP client functionality, install it with `uv pip install "nvidia-nat[mcp]"`.

### List all tools

```console
$ nat mcp client tool list --url http://localhost:9902/mcp
calculator__divide
calculator__compare
calculator__subtract
calculator__add
calculator__multiply
```

### List a tool with schema

```console
$ nat mcp client tool list --url http://localhost:9902/mcp --tool calculator__multiply --detail
Tool: calculator__multiply
Description: Multiply two or more numbers together.
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

### Call a tool with JSON arguments

```console
nat mcp client tool call calculator__multiply \
  --url http://localhost:9902/mcp \
  --json-args '{"numbers": [1, 3, 6, 10]}'
180.0
```

### Using the `/debug/tools/list` route (no MCP client required)

```console
curl -s http://localhost:9902/debug/tools/list | jq
```

## Integration with MCP Clients

The MCP server started with the FastMCP server runtime implements the Model Context Protocol specification, so it works with MCP clients. You can run a workflow that connects to the MCP server by pointing an MCP client function group at `http://localhost:9902/mcp`.

Example:

```bash
nat run --config_file examples/MCP/simple_calculator_fastmcp/configs/config-mcp-client.yml \
  --input "Is 2 times 2 greater than the current hour?"
```

## Authentication

MCP servers started with the FastMCP server runtime can validate bearer tokens using OAuth2 token introspection. Configure `server_auth` in your front end config with the introspection endpoint and client credentials.

See the protected example for a full setup:

- `examples/MCP/simple_calculator_fastmcp_protected`

## Verifying FastMCP Server Health

You can verify server health using the `/health` route or `nat mcp client ping`:

```console
curl -s http://localhost:9902/health | jq
```

```console
nat mcp client ping --url http://localhost:9902/mcp
```

## Related Examples

- `examples/MCP/simple_calculator_fastmcp/`: FastMCP calculator example
- `examples/MCP/simple_calculator_fastmcp_protected/`: Protected FastMCP calculator example
