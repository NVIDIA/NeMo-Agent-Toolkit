# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from pytest import fixture

from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.plugins.mcp.server.front_end_config import MCPFrontEndConfig
from nat.plugins.mcp.server.front_end_plugin_worker import MCPFrontEndPluginWorker
from nat.utils.type_utils import override


class CustomRootLevelRoutesWorker(MCPFrontEndPluginWorker):
    """Custom MCP worker that adds root-level routes to wrapper app."""

    @override
    async def add_root_level_routes(self, wrapper_app: FastAPI, mcp: FastMCP) -> None:
        """Add OAuth discovery and health check routes at root level."""

        @wrapper_app.get("/.well-known/oauth-protected-resource")
        async def oauth_discovery():
            """OAuth 2.0 Protected Resource Metadata endpoint."""
            from starlette.responses import JSONResponse

            return JSONResponse({
                "resource": f"http://{self.front_end_config.host}:{self.front_end_config.port}",
                "authorization_servers": ["https://auth.example.com"],
            })

        @wrapper_app.get("/health")
        async def root_health():
            """Root-level health check endpoint."""
            from starlette.responses import JSONResponse

            return JSONResponse({"status": "healthy", "server": mcp.name, "location": "root"})


@fixture
def mcp_config_with_base_path() -> Config:
    """Fixture providing NAT configuration with base_path set."""
    general_config = GeneralConfig(
        front_end=MCPFrontEndConfig(name="Test MCP", host="localhost", port=9903, base_path="/api/test"))
    return Config(general=general_config)


@fixture
def mcp_config_without_base_path() -> Config:
    """Fixture providing NAT configuration without base_path."""
    general_config = GeneralConfig(front_end=MCPFrontEndConfig(name="Test MCP", host="localhost", port=9903))
    return Config(general=general_config)


async def test_default_add_root_level_routes_does_nothing(mcp_config_with_base_path: Config):
    """Test that default implementation of add_root_level_routes() does nothing."""
    worker = MCPFrontEndPluginWorker(mcp_config_with_base_path)
    wrapper_app = FastAPI()
    mcp = FastMCP("Test Server")

    # Save initial route count (FastAPI adds default docs routes)
    initial_route_count = len(wrapper_app.routes)

    # Should not raise any errors and should not add any routes
    await worker.add_root_level_routes(wrapper_app, mcp)

    # Verify no additional routes were added
    assert len(wrapper_app.routes) == initial_route_count


async def test_custom_worker_adds_root_level_routes(mcp_config_with_base_path: Config):
    """Test that custom worker can override add_root_level_routes() to add routes."""
    from starlette.testclient import TestClient

    worker = CustomRootLevelRoutesWorker(mcp_config_with_base_path)
    wrapper_app = FastAPI()
    mcp = FastMCP("Test Server")

    # Call the method
    await worker.add_root_level_routes(wrapper_app, mcp)

    # Verify routes were added by testing they respond
    client = TestClient(wrapper_app)

    response = client.get("/.well-known/oauth-protected-resource")
    assert response.status_code == 200

    response = client.get("/health")
    assert response.status_code == 200


async def test_root_level_routes_are_accessible(mcp_config_with_base_path: Config):
    """Test that root-level routes respond correctly."""
    from starlette.testclient import TestClient

    worker = CustomRootLevelRoutesWorker(mcp_config_with_base_path)
    wrapper_app = FastAPI()
    mcp = FastMCP("Test Server")

    # Add root-level routes
    await worker.add_root_level_routes(wrapper_app, mcp)

    # Create test client
    client = TestClient(wrapper_app)

    # Test OAuth discovery endpoint
    response = client.get("/.well-known/oauth-protected-resource")
    assert response.status_code == 200
    data = response.json()
    assert "resource" in data
    assert "authorization_servers" in data
    assert data["authorization_servers"] == ["https://auth.example.com"]

    # Test root health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["server"] == "Test Server"
    assert data["location"] == "root"


async def test_add_root_level_routes_called_in_run_with_mount():
    """Test that add_root_level_routes() is called when base_path is configured."""
    from nat.plugins.mcp.server.front_end_plugin import MCPFrontEndPlugin

    # Create config with base_path
    config = Config(general=GeneralConfig(
        front_end=MCPFrontEndConfig(name="Test", base_path="/api/test", transport="streamable-http")))

    plugin = MCPFrontEndPlugin(config)

    # Mock the worker instance
    mock_worker = Mock(spec=MCPFrontEndPluginWorker)
    mock_worker.add_root_level_routes = AsyncMock()
    mock_worker.create_mcp_server = AsyncMock()
    mock_worker.add_routes = AsyncMock()

    # Mock MCP server
    mock_mcp = Mock(spec=FastMCP)
    mock_mcp.name = "Test Server"
    mock_mcp.streamable_http_app = Mock(return_value=Mock())
    mock_mcp.session_manager.run = Mock()
    mock_worker.create_mcp_server.return_value = mock_mcp

    # Test _run_with_mount which should call add_root_level_routes
    with patch.object(plugin, '_get_worker_instance', return_value=mock_worker):
        with patch('uvicorn.Server') as mock_server_class:
            # Mock the server
            mock_server = Mock()
            mock_server.serve = AsyncMock()
            mock_server_class.return_value = mock_server

            # Mock the session manager context
            with patch('contextlib.AsyncExitStack') as mock_exit_stack:
                mock_stack = AsyncMock()
                mock_exit_stack.return_value.__aenter__ = AsyncMock(return_value=mock_stack)
                mock_exit_stack.return_value.__aexit__ = AsyncMock()

                # Run the method
                await plugin._run_with_mount(mock_mcp)

                # Verify add_root_level_routes was called
                mock_worker.add_root_level_routes.assert_called_once()

                # Verify it was called with FastAPI app and mcp server
                call_args = mock_worker.add_root_level_routes.call_args
                assert isinstance(call_args[0][0], FastAPI)  # wrapper_app
                assert call_args[0][1] == mock_mcp  # mcp server


async def test_root_level_routes_not_interfere_with_mcp_routes(mcp_config_with_base_path: Config):
    """Test that root-level routes don't interfere with MCP server routes."""
    from starlette.testclient import TestClient

    worker = CustomRootLevelRoutesWorker(mcp_config_with_base_path)
    wrapper_app = FastAPI()
    mcp = FastMCP("Test Server")

    # Add a custom route to MCP server
    @mcp.custom_route("/mcp-health", methods=["GET"])
    async def mcp_health(_request):
        from starlette.responses import JSONResponse

        return JSONResponse({"status": "mcp-healthy"})

    # Mount MCP server at base_path
    wrapper_app.mount("/api/test", mcp.streamable_http_app())

    # Add root-level routes
    await worker.add_root_level_routes(wrapper_app, mcp)

    # Create test client
    client = TestClient(wrapper_app)

    # Test root-level route works
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["location"] == "root"

    # Test MCP route works at mounted path
    response = client.get("/api/test/mcp-health")
    assert response.status_code == 200
    assert response.json()["status"] == "mcp-healthy"
