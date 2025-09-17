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

import logging
import typing

from nat.builder.front_end import FrontEndBase
from nat.builder.workflow_builder import WorkflowBuilder
from nat.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig
from nat.front_ends.mcp.mcp_front_end_plugin_worker import MCPFrontEndPluginWorkerBase

logger = logging.getLogger(__name__)


class MCPFrontEndPlugin(FrontEndBase[MCPFrontEndConfig]):
    """MCP front end plugin implementation."""

    def get_worker_class(self) -> type[MCPFrontEndPluginWorkerBase]:
        """Get the worker class for handling MCP routes."""
        from nat.front_ends.mcp.mcp_front_end_plugin_worker import MCPFrontEndPluginWorker

        return MCPFrontEndPluginWorker

    @typing.final
    def get_worker_class_name(self) -> str:
        """Get the worker class name from configuration or default."""
        if self.front_end_config.runner_class:
            return self.front_end_config.runner_class

        worker_class = self.get_worker_class()
        return f"{worker_class.__module__}.{worker_class.__qualname__}"

    def _get_worker_instance(self) -> MCPFrontEndPluginWorkerBase:
        """Get an instance of the worker class."""
        # Import the worker class dynamically if specified in config
        if self.front_end_config.runner_class:
            module_name, class_name = self.front_end_config.runner_class.rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_name)
            worker_class = getattr(module, class_name)
        else:
            worker_class = self.get_worker_class()

        return worker_class(self.full_config)

    def _create_token_verifier(self):
        """Create a token verifier stub that logs and always returns success."""
        if not self.front_end_config.require_auth:
            return None

        from mcp.server.auth.provider import AccessToken
        from mcp.server.auth.provider import TokenVerifier

        class StubTokenVerifier(TokenVerifier):

            def __init__(self, server_url: str, scopes: list[str]):
                self.server_url = server_url
                self.scopes = scopes

            async def verify_token(self, token: str) -> AccessToken | None:
                logger.info("STUB: Token verification requested for token")
                logger.info("STUB: Server URL %s", self.server_url)
                logger.info("STUB: Required scopes: %s", self.scopes)
                logger.info("STUB: Returning successful validation (stub implementation)")

                # Always return a successful AccessToken for testing
                return AccessToken(token=token, client_id="stub_client", scopes=self.scopes, resource=self.server_url)

        server_url = f"http://{self.front_end_config.host}:{self.front_end_config.port}"
        return StubTokenVerifier(server_url, self.front_end_config.scopes)

    def _get_server_url(self) -> str:
        """Get the server URL."""
        return f"http://{self.front_end_config.host}:{self.front_end_config.port}"

    async def run(self) -> None:
        """Run the MCP server."""
        # Import FastMCP
        from mcp.server.fastmcp import FastMCP

        # Create auth settings if auth is required
        auth_settings = None
        if self.front_end_config.require_auth:
            from mcp.server.auth.settings import AuthSettings
            from pydantic import AnyHttpUrl

            if not self.front_end_config.auth_server_url:
                raise ValueError("auth_server_url is required when require_auth is True")

            auth_settings = AuthSettings(issuer_url=AnyHttpUrl(self.front_end_config.auth_server_url),
                                         required_scopes=self.front_end_config.scopes,
                                         resource_server_url=self._get_server_url())

        # Create an MCP server with the configured parameters
        mcp = FastMCP(self.front_end_config.name,
                      host=self.front_end_config.host,
                      port=self.front_end_config.port,
                      debug=self.front_end_config.debug,
                      log_level=self.front_end_config.log_level,
                      token_verifier=self._create_token_verifier(),
                      auth=auth_settings)

        # Get the worker instance and set up routes
        worker = self._get_worker_instance()

        # Build the workflow and add routes using the worker
        async with WorkflowBuilder.from_config(config=self.full_config) as builder:
            # Add routes through the worker (includes health endpoint and function registration)
            await worker.add_routes(mcp, builder)

            # Start the MCP server with configurable transport
            # streamable-http is the default, but users can choose sse if preferred
            if self.front_end_config.transport == "sse":
                logger.info("Starting MCP server with SSE endpoint at /sse")
                await mcp.run_sse_async()
            else:  # streamable-http
                logger.info("Starting MCP server with streamable-http endpoint at /mcp/")
                await mcp.run_streamable_http_async()
