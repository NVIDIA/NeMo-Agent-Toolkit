# nvidia-nat-mcp

Subpackage for MCP client integration in NeMo Agent toolkit.

This package provides MCP (Model Context Protocol) client functionality, allowing NeMo Agent toolkit workflows to connect to external MCP servers and use their tools as functions.

## Features

- Connect to MCP servers via stdio, SSE, or streamable-http transports
- Wrap individual MCP tools as NeMo Agent toolkit functions
- Connect to MCP servers and dynamically discover available tools
- Comprehensive error handling and exception management

## Installation

```bash
pip install nvidia-nat-mcp
```

## Usage

This package is automatically loaded when installed. You can use MCP client functions in your workflows:

- `mcp_tool_wrapper`: Wrap a single tool from an MCP server
- `mcp_client`: Connect to an MCP server and discover multiple tools

For more information, see the [MCP Client documentation](https://docs.nvidia.com/nat/workflows/mcp/mcp-client.html).
