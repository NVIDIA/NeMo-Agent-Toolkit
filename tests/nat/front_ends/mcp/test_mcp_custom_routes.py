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

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from mcp.server.fastmcp import FastMCP

from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig
from nat.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin
from nat.front_ends.mcp.mcp_front_end_plugin_worker import MCPFrontEndPluginWorker
from nat.utils.type_utils import override


class CustomMCPWorker(MCPFrontEndPluginWorker):
    """Custom MCP worker that adds additional routes."""

    @override
    async def add_routes(self, mcp, builder: WorkflowBuilder):
        """Add default routes plus custom routes."""
        # Add all the default routes first
        await super().add_routes(mcp, builder)

        # Add custom routes here
        @mcp.custom_route("/custom", methods=["GET"])
        async def custom_route(_request):
            """Custom route for testing."""
            from starlette.responses import JSONResponse
            return JSONResponse({"message": "This is a custom MCP route"})

        @mcp.custom_route("/api/status", methods=["GET"])
        async def api_status(_request):
            """API status endpoint."""
            from starlette.responses import JSONResponse
            return JSONResponse({"status": "ok", "server_name": mcp.name, "custom_worker": True})


@pytest.fixture
def mcp_nat_config() -> Config:
    """Fixture to provide a minimal NAT configuration."""
    general_config = GeneralConfig(front_end=MCPFrontEndConfig(name="Test MCP", host="localhost", port=9902))
    return Config(general=general_config)


async def test_custom_mcp_worker(mcp_nat_config: Config):
    """Test that custom MCP worker can add routes without breaking functionality."""
    worker = CustomMCPWorker(mcp_nat_config)
    mcp = FastMCP("Test Server")

    # Mock out the function registration since we're only testing custom routes
    mock_builder = Mock(spec=WorkflowBuilder)

    # Create a minimal mock workflow with functions
    mock_workflow = Mock()
    mock_workflow.functions = {"test_function": Mock()}  # Simple dict with one mock function
    mock_workflow.config.workflow.type = "test_workflow"
    mock_builder.build.return_value = mock_workflow

    # Mock the register_function_with_mcp so we skip function registration entirely
    with patch('nat.front_ends.mcp.tool_converter.register_function_with_mcp'):
        # Test that the worker can add routes
        await worker.add_routes(mcp, mock_builder)

    # Test that the custom routes are added
    custom_routes = [route for route in mcp._custom_starlette_routes if route.path == "/custom"]
    api_status_routes = [route for route in mcp._custom_starlette_routes if route.path == "/api/status"]

    # Test that the default health route is added
    health_routes = [route for route in mcp._custom_starlette_routes if route.path == "/health"]

    assert len(custom_routes) > 0, "Custom route /custom should be added"
    assert len(api_status_routes) > 0, "Custom route /api/status should be added"
    assert len(health_routes) > 0, "Health route /health should be added"


def test_runner_class_configuration(mcp_nat_config: Config):
    """Test that the runner_class configuration works correctly."""
    # Test with no runner_class (should use default)
    plugin_default = MCPFrontEndPlugin(mcp_nat_config)
    assert "MCPFrontEndPluginWorker" in plugin_default.get_worker_class_name()

    # Test with custom runner_class (should return the custom class name)
    custom_nat_config = Config(general=GeneralConfig(front_end=MCPFrontEndConfig(
        runner_class="nat.front_ends.mcp.test_mcp_custom_routes.CustomMCPWorker")))

    plugin_custom = MCPFrontEndPlugin(custom_nat_config)
    assert "CustomMCPWorker" in plugin_custom.get_worker_class_name()
