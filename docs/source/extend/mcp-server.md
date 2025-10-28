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

# Adding a Custom MCP Server Worker

:::{note}
We recommend reading the [MCP Server Guide](../workflows/mcp/mcp-server.md) before proceeding with this documentation, to understand how MCP servers work in NVIDIA NeMo Agent toolkit.
:::

The NVIDIA NeMo Agent toolkit provides a default MCP server worker that publishes your workflow functions as MCP tools. However, you may need to customize the server behavior for enterprise requirements such as authentication, custom endpoints, or telemetry. This guide shows you how to create custom MCP server workers that extend the default implementation.

## When to Create a Custom Worker

Create a custom MCP worker when you need to:
- **Add authentication/authorization**: OAuth, API keys, JWT tokens, or custom auth flows
- **Integrate custom transport protocols**: WebSocket, gRPC, or other communication methods
- **Add logging and telemetry**: Custom logging, metrics collection, or distributed tracing
- **Modify server behavior**: Custom middleware, error handling, or protocol extensions
- **Integrate with enterprise systems**: SSO, audit logging, or compliance requirements

## Creating and Registering a Custom MCP Worker

To extend the NeMo Agent toolkit with custom MCP workers, you need to create a worker class that inherits from {py:class}`~nat.front_ends.mcp.mcp_front_end_plugin_worker.MCPFrontEndPluginWorker` and override the methods you want to customize.

This section provides a step-by-step guide to create and register a custom MCP worker with the NeMo Agent toolkit. A custom status endpoint worker is used as an example to demonstrate the process.

## Step 1: Implement the Worker Class

Create a new Python file for your worker implementation. The following example shows a minimal worker that adds a custom status endpoint to the MCP server.

Each worker is instantiated once when `nat mcp serve` runs. The `create_mcp_server()` method executes during initialization, and `add_routes()` runs after the workflow is built.

<!-- path-check-skip-next-line -->
`src/my_package/custom_worker.py`:
```python
import logging

from mcp.server.fastmcp import FastMCP

from nat.builder.workflow_builder import WorkflowBuilder
from nat.front_ends.mcp.mcp_front_end_plugin_worker import MCPFrontEndPluginWorker

logger = logging.getLogger(__name__)


class CustomStatusWorker(MCPFrontEndPluginWorker):
    """MCP worker that adds a custom status endpoint."""

    async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
        """Register tools and add custom server behavior.

        This method calls the parent implementation to get all default behavior,
        then adds custom routes.

        Args:
            mcp: The FastMCP server instance
            builder: The workflow builder containing functions to expose
        """
        # Get all default routes and tool registration
        await super().add_routes(mcp, builder)

        # Add a custom status endpoint
        @mcp.custom_route("/custom/status", methods=["GET"])
        async def custom_status(_request):
            """Custom status endpoint with additional server information."""
            from starlette.responses import JSONResponse

            logger.info("Custom status endpoint called")
            return JSONResponse({
                "status": "ok",
                "server": mcp.name,
                "custom_worker": "CustomStatusWorker"
            })
```

**Key components**:
- **Inheritance**: Extend {py:class}`~nat.front_ends.mcp.mcp_front_end_plugin_worker.MCPFrontEndPluginWorker`
- **`super().add_routes()`**: Calls parent to get standard tool registration and default routes
- **`@mcp.custom_route()`**: Adds custom HTTP endpoints to the server
- **Clean inheritance**: Use standard Python `super()` pattern to extend behavior

## Step 2: Use the Worker in Your Workflow

Configure your workflow to use the custom worker by specifying the fully qualified class name in the `runner_class` field.

<!-- path-check-skip-next-line -->
`custom_mcp_server_workflow.yml`:
```yaml
general:
  front_end:
    _type: mcp
    runner_class: "my_package.custom_worker.CustomStatusWorker"
    name: "my_custom_server"
    host: "localhost"
    port: 9000


llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.3-70b-instruct

functions:
  search:
    _type: tavily_internet_search

workflow:
  _type: react_agent
  llm_name: nim_llm
  tool_names: [search]
```

## Step 3: Run and Test Your Server

Start your server using the NeMo Agent toolkit CLI:

```bash
nat mcp serve --config_file custom_mcp_server_workflow.yml
```

**Expected output**:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:9000 (Press CTRL+C to quit)
```

**Test the server** with the MCP client:

```bash
# List available tools
nat mcp client tool list --url http://localhost:9000/mcp

# Call a tool
nat mcp client tool call search \
  --url http://localhost:9000/mcp \
  --json-args '{"question": "When is the next GTC event?"}'

# Test the custom status endpoint
curl http://localhost:9000/custom/status
```

**Expected response from custom endpoint**:
```json
{
  "status": "ok",
  "server": "my_custom_server",
  "custom_worker": "CustomStatusWorker"
}
```

## Understanding Inheritance and Extension

### Using `super().add_routes()`

When extending {py:class}`~nat.front_ends.mcp.mcp_front_end_plugin_worker.MCPFrontEndPluginWorker`, call `super().add_routes()` to get all default functionality:

- **Health endpoint**: `/health` for server status checks
- **Workflow building**: Processes your workflow configuration
- **Function-to-tool conversion**: Registers NeMo Agent toolkit functions as MCP tools
- **Debug endpoints**: Additional routes for development

Most workers call `super().add_routes()` first to ensure all standard NeMo Agent toolkit tools are registered, then add custom features:

```python
async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
    # Get all default behavior from parent
    await super().add_routes(mcp, builder)

    # Add your custom features
    @mcp.custom_route("/my/endpoint", methods=["GET"])
    async def my_endpoint(_request):
        return JSONResponse({"custom": "data"})
```

### Overriding `create_mcp_server()`

Override `create_mcp_server()` when you need to use a different MCP server implementation:

```python
async def create_mcp_server(self) -> FastMCP:
    from my_custom_mcp import CustomFastMCP

    return CustomFastMCP(
        name=self.front_end_config.name,
        host=self.front_end_config.host,
        port=self.front_end_config.port,
        # Custom parameters
        auth_provider=self.get_auth_provider(),
    )
```

**Authentication ownership**: When you override `create_mcp_server()`, your worker controls authentication. If you need custom auth (JWT, OAuth2, API keys), configure it inside `create_mcp_server()`. Any front-end config auth settings are optional hints and may be ignored by your worker.

### Accessing Configuration

Your worker has access to configuration through instance variables:

- **`self.front_end_config`**: MCP server configuration
  - `name`: Server name
  - `host`: Server host address
  - `port`: Server port number
  - `debug`: Debug mode flag

- **`self.full_config`**: Complete NeMo Agent toolkit configuration
  - `general`: General settings including front end config
  - `llms`: LLM configurations
  - `functions`: Function configurations
  - `workflow`: Workflow configuration

**Example using configuration**:

```python
async def create_mcp_server(self) -> FastMCP:
    # Access server name from config
    server_name = self.front_end_config.name

    # Customize based on debug mode
    if self.front_end_config.debug:
        logger.info(f"Creating debug server: {server_name}")

    return FastMCP(
        name=server_name,
        host=self.front_end_config.host,
        port=self.front_end_config.port,
        debug=self.front_end_config.debug,
    )
```

## Summary

This guide provides a step-by-step process to create custom MCP server workers in the NeMo Agent toolkit. The custom status worker demonstrates how to:

1. Extend {py:class}`~nat.front_ends.mcp.mcp_front_end_plugin_worker.MCPFrontEndPluginWorker`
2. Override `add_routes()` and use `super()` to get default behavior
3. Override `create_mcp_server()` to use a different server implementation. When doing so, implement your own authentication and authorization logic within that server.

Custom workers enable enterprise features like authentication, telemetry, and integration with existing infrastructure without modifying NeMo Agent toolkit core code.
