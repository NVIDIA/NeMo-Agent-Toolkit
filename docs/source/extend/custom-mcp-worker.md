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

The NeMo Agent toolkit's MCP frontend supports custom server implementations through a plugin system. This guide shows you how to create custom MCP server workers that extend the default server behavior.

## When to Create a Custom Worker

Create a custom MCP worker when you need to:
- **Add authentication/authorization**: OAuth, API keys, JWT tokens, or custom auth flows
- **Integrate custom transport protocols**: WebSocket, gRPC, or other communication methods
- **Add logging and telemetry**: Request/response logging, metrics collection, or distributed tracing
- **Modify server behavior**: Custom middleware, error handling, or protocol extensions
- **Integrate with enterprise systems**: SSO, audit logging, or compliance requirements

## Creating and Registering a Custom MCP Worker

To extend the NeMo Agent toolkit with custom MCP workers, you need to create a worker class that inherits from {py:class}`~nat.front_ends.mcp.mcp_front_end_plugin_worker.McpServerWorker` and implement two required methods.

This section provides a step-by-step guide to create and register a custom MCP worker with the NeMo Agent toolkit. A request logging worker is used as an example to demonstrate the process.

## Step 1: Implement the Worker Class

This step defines how your worker creates and configures the MCP server before adding any custom routes or middleware.

Create a new Python file for your worker implementation. The following example shows a minimal worker that adds request logging middleware.

Each worker is instantiated once when `nat mcp serve` runs. The `create_mcp_server()` method executes during initialization, and `add_routes()` runs after the workflow is built.

<!-- path-check-skip-next-line -->
`src/my_package/logging_worker.py`:
```python
import logging
import time

from mcp.server.fastmcp import FastMCP

from nat.builder.workflow_builder import WorkflowBuilder
from nat.front_ends.mcp.mcp_front_end_plugin_worker import McpServerWorker

logger = logging.getLogger(__name__)


class LoggingMCPWorker(McpServerWorker):
    """MCP worker that adds request/response logging."""

    async def create_mcp_server(self) -> FastMCP:
        """Create and configure the MCP server.

        This method is called once during server initialization.
        Return a FastMCP instance or any subclass with custom behavior.

        Returns:
            FastMCP: The configured server instance
        """
        return FastMCP(
            name=self.front_end_config.name,
            host=self.front_end_config.host,
            port=self.front_end_config.port,
            debug=self.front_end_config.debug,
        )

    async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
        """Register tools and add custom server behavior.

        This method is called after the server is created.
        Use _default_add_routes() to get standard tool registration,
        then add your custom features.

        Args:
            mcp: The FastMCP server instance
            builder: The workflow builder containing functions to expose
        """
        # Register NeMo Agent toolkit functions as MCP tools (standard behavior)
        await self._default_add_routes(mcp, builder)

        # Add custom middleware for request/response logging
        @mcp.app.middleware("http")
        async def log_requests(request, call_next):
            start_time = time.time()

            logger.info(f"Request: {request.method} {request.url.path}")
            # call_next is part of the Starlette middleware stack used by FastMCP. Each request passes through this
            # coroutine chain before returning a response.
            response = await call_next(request)

            duration = time.time() - start_time
            logger.info(f"Response: {response.status_code} ({duration:.2f}s)")
            return response
```

**Key components**:
- **Inheritance**: Extend {py:class}`~nat.front_ends.mcp.mcp_front_end_plugin_worker.McpServerWorker`
- **`create_mcp_server()`**: Create and return the MCP server instance
- **`add_routes()`**: Register tools and add custom features
- **`_default_add_routes()`**: Helper that provides standard tool registration

## Step 2: Use the Worker in Your Workflow

Configure your workflow to use the custom worker by specifying the fully qualified class name in the `runner_class` field.

<!-- path-check-skip-next-line -->
`configs/my_workflow.yml`:
```yaml
general:
  front_end:
    _type: mcp
    runner_class: "my_package.logging_worker.LoggingMCPWorker"
    name: "my_logging_server"
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
nat mcp serve --config_file configs/my_workflow.yml
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
```

**Observe the logs** showing your custom middleware in action:
```
INFO:my_package.logging_worker:Request: POST /tools/call
INFO:my_package.logging_worker:Response: 200 (0.45s)
```

## Understanding Helper Methods

### `_default_add_routes()`

The {py:meth}`~nat.front_ends.mcp.mcp_front_end_plugin_worker.McpServerWorker._default_add_routes` method provides standard tool registration functionality:

- **Health endpoint**: `/health` for server status checks
- **Workflow building**: Processes your workflow configuration
- **Function-to-tool conversion**: Registers NeMo Agent toolkit functions as MCP tools
- **Debug endpoints**: Additional routes for development

Most workers call `_default_add_routes()` first to ensure all standard NeMo Agent toolkit tools are registered, then extend or modify the behavior as needed. You can skip this call if you intend to handle all route registration manually.

```python
async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
    # Get standard behavior
    await self._default_add_routes(mcp, builder)

    # Add your custom features
    self._add_custom_middleware(mcp)
    self._register_custom_routes(mcp)
```

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

This guide provides a step-by-step process to create custom MCP server workers in the NeMo Agent toolkit. The request logging worker demonstrates how to:

1. Extend {py:class}`~nat.front_ends.mcp.mcp_front_end_plugin_worker.McpServerWorker`
2. Implement `create_mcp_server()` and `add_routes()` methods
3. Use `_default_add_routes()` for standard tool registration
4. Add custom middleware for logging, monitoring, or other features
5. Configure and test the custom worker in your workflows

Custom workers enable enterprise features like authentication, telemetry, and integration with existing infrastructure without modifying NeMo Agent toolkit core code.
