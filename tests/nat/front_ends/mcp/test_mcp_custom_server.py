# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test MCP plugin system for loading custom workers."""

import logging
import os

import pytest
from mcp.server.fastmcp import FastMCP

from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig
from nat.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin
from nat.front_ends.mcp.mcp_front_end_plugin_worker import MCPFrontEndPluginWorker
from nat.front_ends.mcp.mcp_front_end_plugin_worker import McpServerWorker
from nat.test.functions import EchoFunctionConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Test Fixtures: Dummy MCP Worker Plugins
# ============================================================================


class DummyMCPWorker(McpServerWorker):
    """Minimal test plugin that creates a server with a custom name prefix."""

    async def create_mcp_server(self) -> FastMCP:
        """Create a basic FastMCP server with 'DUMMY-' name prefix."""
        return FastMCP(
            name=f"DUMMY-{self.front_end_config.name}",
            host=self.front_end_config.host,
            port=self.front_end_config.port,
            debug=True,  # Always debug for testing
        )

    async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
        """Use default route registration from base worker."""
        # Delegate to default implementation
        default_worker = MCPFrontEndPluginWorker(self.full_config)
        await default_worker.add_routes(mcp, builder)


class CustomMCPWorker(McpServerWorker):
    """Test plugin that uses environment variables for configuration."""

    async def create_mcp_server(self) -> FastMCP:
        """Create server with name from CUSTOM_MCP_NAME env var."""
        custom_name = os.getenv("CUSTOM_MCP_NAME", self.front_end_config.name)

        logger.info("CustomMCPWorker: Creating server with name '%s'", custom_name)

        return FastMCP(
            name=custom_name,
            host=self.front_end_config.host,
            port=self.front_end_config.port,
            debug=self.front_end_config.debug,
        )

    async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
        """Add routes with logging."""
        logger.info("CustomMCPWorker: Adding routes to %s", mcp.name)

        # Use default route registration
        default_worker = MCPFrontEndPluginWorker(self.full_config)
        await default_worker.add_routes(mcp, builder)

        logger.info("CustomMCPWorker: Routes added successfully to %s", mcp.name)


class LoggingMCPWorker(McpServerWorker):
    """MCP worker that implements the logging example from custom-mcp-worker.md documentation."""

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
        # Register NAT functions as MCP tools (standard behavior)
        await self._default_add_routes(mcp, builder)

        # Add custom middleware for request/response logging
        @mcp.app.middleware("http")
        async def log_requests(request, call_next):
            import logging
            import time

            logger = logging.getLogger(__name__)
            start_time = time.time()

            logger.info(f"Request: {request.method} {request.url.path}")
            response = await call_next(request)

            duration = time.time() - start_time
            logger.info(f"Response: {response.status_code} ({duration:.2f}s)")
            return response


# ============================================================================
# Test Fixtures: Config Setup
# ============================================================================


@pytest.fixture
def base_config():
    """Create base config for testing."""
    echo_function_config = EchoFunctionConfig()

    mcp_front_end_config = MCPFrontEndConfig(name="Test Server", host="localhost", port=9999)

    return Config(general=GeneralConfig(front_end=mcp_front_end_config),
                  workflow=echo_function_config,
                  functions={"echo": echo_function_config})


# ============================================================================
# Tests: Plugin System
# ============================================================================


def test_default_worker_loads(base_config):
    """Test that default worker loads when no runner_class specified."""
    plugin = MCPFrontEndPlugin(full_config=base_config)

    worker = plugin._get_worker_instance()

    assert worker is not None
    assert isinstance(worker, MCPFrontEndPluginWorker)
    assert worker.__class__.__name__ == "MCPFrontEndPluginWorker"


def test_default_worker_class_name(base_config):
    """Test that default worker class name is returned correctly."""
    plugin = MCPFrontEndPlugin(full_config=base_config)

    class_name = plugin.get_worker_class_name()

    assert class_name == "nat.front_ends.mcp.mcp_front_end_plugin_worker.MCPFrontEndPluginWorker"


def test_custom_worker_loads_via_importlib(base_config):
    """Test loading custom worker via importlib from this test module."""
    # Use __name__ pattern like eval tests do to reference workers in this file
    base_config.general.front_end.runner_class = f"{__name__}.DummyMCPWorker"

    plugin = MCPFrontEndPlugin(full_config=base_config)
    worker = plugin._get_worker_instance()

    assert worker is not None
    assert worker.__class__.__name__ == "DummyMCPWorker"
    assert hasattr(worker, 'create_mcp_server')
    assert hasattr(worker, 'add_routes')


def test_custom_worker_class_name(base_config):
    """Test that custom worker class name is returned from config."""
    base_config.general.front_end.runner_class = f"{__name__}.CustomMCPWorker"

    plugin = MCPFrontEndPlugin(full_config=base_config)
    class_name = plugin.get_worker_class_name()

    assert class_name == f"{__name__}.CustomMCPWorker"


def test_invalid_worker_module_raises_error(base_config):
    """Test that invalid worker module raises ImportError."""
    base_config.general.front_end.runner_class = "nonexistent.module.Worker"

    plugin = MCPFrontEndPlugin(full_config=base_config)

    with pytest.raises(ModuleNotFoundError):
        plugin._get_worker_instance()


def test_invalid_worker_class_raises_error(base_config):
    """Test that invalid worker class raises AttributeError."""
    base_config.general.front_end.runner_class = f"{__name__}.NonExistentWorker"

    plugin = MCPFrontEndPlugin(full_config=base_config)

    with pytest.raises(AttributeError):
        plugin._get_worker_instance()


async def test_default_worker_creates_server(base_config):
    """Test that default worker creates FastMCP server correctly."""
    worker = MCPFrontEndPluginWorker(base_config)

    mcp = await worker.create_mcp_server()

    assert mcp is not None
    assert isinstance(mcp, FastMCP)
    assert mcp.name == "Test Server"


async def test_dummy_worker_creates_custom_server(base_config):
    """Test that dummy worker creates server with custom name."""
    worker = DummyMCPWorker(base_config)

    mcp = await worker.create_mcp_server()

    assert mcp is not None
    assert mcp.name == "DUMMY-Test Server"
    assert mcp.name.startswith("DUMMY-")


async def test_custom_worker_uses_env_var(base_config):
    """Test that custom worker uses environment variable for name."""
    # Set environment variable
    os.environ["CUSTOM_MCP_NAME"] = "MyCustomServer"

    try:
        worker = CustomMCPWorker(base_config)
        mcp = await worker.create_mcp_server()

        assert mcp is not None
        assert mcp.name == "MyCustomServer"
    finally:
        # Clean up
        os.environ.pop("CUSTOM_MCP_NAME", None)


async def test_custom_worker_fallback_to_config(base_config):
    """Test that custom worker falls back to config name when env var not set."""
    # Make sure env var is not set
    os.environ.pop("CUSTOM_MCP_NAME", None)

    worker = CustomMCPWorker(base_config)
    mcp = await worker.create_mcp_server()

    assert mcp is not None
    assert mcp.name == "Test Server"


async def test_logging_worker_creates_server(base_config):
    """Test that the logging worker can create a server."""
    worker = LoggingMCPWorker(base_config)
    mcp = await worker.create_mcp_server()

    assert mcp is not None
    assert mcp.name == "Test Server"
    assert mcp.host == "localhost"
    assert mcp.port == 9999


async def test_logging_worker_adds_routes(base_config):
    """Test that the logging worker can add routes with middleware."""
    worker = LoggingMCPWorker(base_config)
    mcp = await worker.create_mcp_server()
    builder = WorkflowBuilder(general_config=base_config.general)

    # Add routes (including middleware)
    await worker.add_routes(mcp, builder)

    # Verify middleware was registered (middleware is stored in app.user_middleware)
    assert len(mcp.app.user_middleware) > 0, "Middleware should be registered"

    # Verify that tools were registered via default route registration
    # This is indirect - we check that the workflow was configured
    assert builder._workflow is not None, "Workflow should be configured after add_routes"


async def test_logging_worker_middleware_logs_requests(base_config, caplog):
    """Test that the logging worker middleware logs requests."""
    import logging
    from unittest.mock import MagicMock

    caplog.set_level(logging.INFO)

    worker = LoggingMCPWorker(base_config)
    mcp = await worker.create_mcp_server()
    builder = WorkflowBuilder(general_config=base_config.general)

    # Add routes (including middleware)
    await worker.add_routes(mcp, builder)

    # Simulate a request through the middleware
    # Get the middleware function
    middleware = mcp.app.user_middleware[0]
    middleware_func = middleware.cls

    # Create mock request and response
    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request.url.path = "/test"

    mock_response = MagicMock()
    mock_response.status_code = 200

    # Create a mock call_next that returns the response
    async def mock_call_next(request):
        return mock_response

    # Call the middleware
    response = await middleware_func(mock_request, mock_call_next)

    # Verify the middleware returned the response
    assert response == mock_response

    # Verify logging occurred
    assert any("Request: GET /test" in record.message for record in caplog.records), \
        "Request should be logged"
    assert any("Response: 200" in record.message for record in caplog.records), \
        "Response should be logged"
