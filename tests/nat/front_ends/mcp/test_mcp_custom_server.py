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
from nat.front_ends.mcp.mcp_front_end_plugin_worker import MCPFrontEndPluginWorkerBase
from nat.test.functions import EchoFunctionConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Test Fixtures: Dummy MCP Worker Plugins
# ============================================================================


class DummyMCPWorker(MCPFrontEndPluginWorkerBase):
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


class CustomMCPWorker(MCPFrontEndPluginWorkerBase):
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
