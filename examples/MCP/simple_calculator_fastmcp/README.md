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

# Simple Calculator - FastMCP

**Complexity:** 🟢 Beginner

This example demonstrates how to run the NVIDIA NeMo Agent Toolkit as an MCP server using the FastMCP server runtime and use those tools from a Model Context Protocol (MCP) client workflow.

This example mirrors the `simple_calculator_mcp` workflow, but it uses the FastMCP server command and defaults to port `9902`. The FastMCP server integration comes from `nvidia-nat-fastmcp`, and the MCP client commands and configuration use `nvidia-nat-mcp`.

## Prerequisites

- **Agent toolkit**: Ensure you have the NVIDIA NeMo Agent Toolkit installed. If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install the toolkit.
- **Base workflow**: This example builds upon the Getting Started [Simple Calculator](../../getting_started/simple_calculator/) example. Make sure you are familiar with the example before proceeding.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install the toolkit.

### Install this Workflow

Install this example:

```bash
uv pip install -e examples/MCP/simple_calculator_fastmcp
```

## Run the Workflow

1. Start the MCP server using the FastMCP server runtime:
<!-- path-check-skip-begin -->
```bash
nat fastmcp server run --config_file examples/getting_started/simple_calculator/configs/config.yml
```
<!-- path-check-skip-end -->
This starts an MCP server on port `9902` with endpoint `/mcp` and uses `streamable-http` transport.

2. Inspect the tools available on the MCP server using the MCP client:

```bash
nat mcp client tool list --url http://localhost:9902/mcp
```

Sample output:

```text
calculator__add
calculator__subtract
calculator__multiply
calculator__divide
calculator__compare
```

3. Run the workflow:

If you installed this example using `uv pip install -e examples/MCP/simple_calculator_fastmcp`, the `mcp-server-time` dependency is already available. If you did not install the example package, install it manually:

```bash
uv pip install mcp-server-time
```

```bash
nat run --config_file examples/MCP/simple_calculator_fastmcp/configs/config-mcp-client.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```

The client configuration is in `examples/MCP/simple_calculator_fastmcp/configs/config-mcp-client.yml`.

## Expose Selected Tools

To expose only specific tools from the workflow, use `--tool_names` when starting the server:

<!-- path-check-skip-begin -->
```bash
nat fastmcp server run --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --tool_names calculator__multiply \
  --tool_names calculator__divide
```
<!-- path-check-skip-end -->

## Related Examples
- `examples/MCP/simple_calculator_fastmcp_protected/`: Protected FastMCP calculator example

## References

- [FastMCP Server](../../../docs/source/run-workflows/fastmcp-server.md) - Learn about running the FastMCP server runtime
- [MCP Client](../../../docs/source/build-workflows/mcp-client.md) - Learn about using the MCP client to interact with the MCP server
