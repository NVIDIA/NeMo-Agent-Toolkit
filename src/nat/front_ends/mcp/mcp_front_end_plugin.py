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

from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
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

    async def _create_token_verifier(self, token_verifier_config: OAuth2ResourceServerConfig):
        """Create a token verifier based on configuration."""
        from nat.front_ends.mcp.introspection_token_verifier import IntrospectionTokenVerifier

        if not self.front_end_config.server_auth:
            return None

        return IntrospectionTokenVerifier(token_verifier_config)

    async def run(self) -> None:
        """Run the MCP server."""
        # Import FastMCP
        from mcp.server.fastmcp import FastMCP

        # Create auth settings and token verifier if auth is required
        auth_settings = None
        token_verifier = None

        # Build the workflow and add routes using the worker
        async with WorkflowBuilder.from_config(config=self.full_config) as builder:

            if self.front_end_config.server_auth:
                from mcp.server.auth.settings import AuthSettings
                from pydantic import AnyHttpUrl

                server_url = f"http://{self.front_end_config.host}:{self.front_end_config.port}"

                auth_settings = AuthSettings(issuer_url=AnyHttpUrl(self.front_end_config.server_auth.issuer_url),
                                             required_scopes=self.front_end_config.server_auth.scopes,
                                             resource_server_url=AnyHttpUrl(server_url))

                token_verifier = await self._create_token_verifier(self.front_end_config.server_auth)

            # Create an MCP server with the configured parameters
            mcp = FastMCP(name=self.front_end_config.name,
                          host=self.front_end_config.host,
                          port=self.front_end_config.port,
                          debug=self.front_end_config.debug,
                          auth=auth_settings,
                          token_verifier=token_verifier)

            # Get the worker instance and set up routes
            worker = self._get_worker_instance()

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
