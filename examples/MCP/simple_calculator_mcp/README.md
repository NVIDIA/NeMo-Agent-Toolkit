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

# Simple Calculator - Model Context Protocol (MCP)

**Complexity:** 🟢 Beginner

This example demonstrates how to integrate the NVIDIA NeMo Agent Toolkit with [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/modelcontextprotocol) servers. You'll learn to use remote tools through MCP and publish Agent toolkit functions as MCP services.

This example uses **shared workflow** mode, allowing multiple users to interact concurrently using the same unprotected MCP calculator tools. This is useful for development and testing purposes.

For production use see the [Simple Calculator MCP Protected](../simple_calculator_mcp_protected/) example, which demonstrates how to set up an OAuth2-protected MCP server and securely access it in a per-user workflow.

## Prerequisites

Ensure the following prerequisites are met before running the simply calculator workflow.

- **Agent toolkit**: Ensure you have the Agent toolkit installed. If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.
- **Base workflow**: This example builds upon the Getting Started [Simple Calculator](../../getting_started/simple_calculator/) example. Make sure you are familiar with the example before proceeding.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install this Workflow

Install this example:

```bash
uv pip install -e examples/MCP/simple_calculator_mcp
```

## Run the Workflow

### NeMo Agent Toolkit as an MCP Client
You can run the simple calculator workflow using Remote MCP tools. In this case, the workflow acts as a MCP client and connects to the MCP server running on the specified URL. Details are provided in the [MCP Client Guide](../../../docs/source/build-workflows/mcp-client.md).

### NeMo Agent Toolkit as an MCP Server
You can publish the simple calculator tools using MCP using the `nat mcp serve` command. Details are provided in the [MCP Server Guide](../../../docs/source/run-workflows/mcp-server.md).


### MCP Client Configuration
NeMo Agent Toolkit enables workflows to use MCP tools as functions. The library handles the MCP server connection, tool discovery, and function registration. This allows the workflow to use MCP tools as regular functions.

Tools served by remote MCP servers can be leveraged as NeMo Agent Toolkit functions using `mcp_client`, a flexible configuration using function groups that allows you to connect to an MCP server, dynamically discover the tools it serves, and register them as NeMo Agent Toolkit functions. The `config-mcp-client.yml` example demonstrates how to use the `mcp_client` function group with both local and remote MCP servers.

### Running the example
The `config-mcp-client.yml` example demonstrates how to use the `mcp_client` function group with both local and remote MCP servers. This configuration shows how to use multiple MCP servers with different transports in the same workflow.

`examples/MCP/simple_calculator_mcp/configs/config-mcp-client.yml`:
```yaml
functions:
  current_timezone:
    _type: current_timezone

function_groups:
  mcp_time:
    _type: mcp_client
    server:
      transport: stdio
      command: "python"
      args: ["-m", "mcp_server_time", "--local-timezone=America/Los_Angeles"]
    tool_overrides:
      get_current_time:
        alias: get_current_time_mcp_tool
        description: "Returns the current date and time. REQUIRED: You must call the current_timezone tool first and pass its result as the timezone argument. Do not use your own or an assumed timezone; only use the value returned by current_timezone."

  mcp_math:
    _type: mcp_client
    server:
      transport: streamable-http
      url: "http://localhost:9901/mcp"
    include:
      - calculator__add
      - calculator__subtract
      - calculator__multiply
      - calculator__divide
      - calculator__compare

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 1024

workflow:
  _type: react_agent
  tool_names:
    - current_timezone
    - mcp_time
    - mcp_math
```

This configuration creates two function groups:
- `mcp_time`: Connects to a local MCP server using stdio transport to get current date and time. The timezone is always assumed to be America/Los_Angeles
- `mcp_math`: Connects to a remote MCP server using streamable-http transport to access calculator tools

To run this example:

1. Start the remote MCP server:
```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml
```
This starts an MCP server on port 9901 with endpoint `/mcp` and uses `streamable-http` transport. Refer to [MCP Server](../../../docs/source/run-workflows/mcp-server.md) for more information.

2. Run the workflow:
```bash
nat run --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-client.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```

# Per-user workflow
The `config-per-user-mcp-client.yml` example demonstrates how to use the `per_user_mcp_client` function group with a per-user workflow.
Per-user workflows are useful when:
1. You need to lazy instantiate MCP sessions for each user on first input.
2. Need complete isolation of workflow instances for each user.

`examples/MCP/simple_calculator_mcp/configs/config-per-user-mcp-client.yml`:
```yaml
function_groups:
  mcp_math:
    _type: per_user_mcp_client
    server:
      transport: streamable-http
      url: "http://localhost:9901/mcp"
    include:
      - calculator__add
      - calculator__subtract
      - calculator__multiply
      - calculator__divide
```

To run this example:

1. Start the remote MCP server:
```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml
```

2. Run the workflow:
```bash
nat serve --config_file examples/MCP/simple_calculator_mcp/configs/config-per-user-mcp-client.yml
```

3. Send a message to the workflow for users "Alice" and "Hatter":
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d '{"messages": [{"role": "user", "content": "Is the product of 2 * 4 greater than the current hour of the day?"}]}'
```
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=Hatter" \
  -d '{"messages": [{"role": "user", "content": "Is the product of 2 * 4 greater than the current hour of the day?"}]}'
```

4. You can also list the tools available to each user:
```bash
curl -s "http://localhost:8000/mcp/client/tool/list/per_user?user_id=alice" | jq
```

```bash
curl -s "http://localhost:8000/mcp/client/tool/list/per_user?user_id=Hatter" | jq
```
