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

import logging
import typing

from nat.builder.front_end import FrontEndBase
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.mcp.server.front_end_config import MCPFrontEndConfig
from nat.plugins.mcp.server.front_end_plugin_worker import MCPFrontEndPluginWorkerBase

if typing.TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)


class MCPFrontEndPlugin(FrontEndBase[MCPFrontEndConfig]):
    """MCP front end plugin implementation."""

    def get_worker_class(self) -> type[MCPFrontEndPluginWorkerBase]:
        """Get the worker class for handling MCP routes."""
        from nat.plugins.mcp.server.front_end_plugin_worker import MCPFrontEndPluginWorker

        return MCPFrontEndPluginWorker

    @typing.final
    def get_worker_class_name(self) -> str:
        """Get the worker class name from configuration or default."""
        if self.front_end_config.runner_class:
            return self.front_end_config.runner_class

        worker_class = self.get_worker_class()
        return f"{worker_class.__module__}.{worker_class.__qualname__}"

    def _get_worker_instance(self):
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

    async def run(self) -> None:
        """Run the MCP server."""
        # Build the workflow and add routes using the worker
        async with WorkflowBuilder.from_config(config=self.full_config) as builder:

            # Get the worker instance
            worker = self._get_worker_instance()

            # Let the worker create the MCP server (allows plugins to customize)
            mcp = await worker.create_mcp_server()

            # Add routes through the worker (includes health endpoint and function registration)
            await worker.add_routes(mcp, builder)

            # Phase A: use FastMCP 3 defaults for transport/base_path
            try:
                if self.front_end_config.base_path:
                    logger.warning("Phase A ignores base_path=%s; using FastMCP 3 defaults.",
                                   self.front_end_config.base_path)
                if self.front_end_config.transport != "streamable-http":
                    logger.warning("Phase A ignores transport=%s; using FastMCP 3 defaults.",
                                   self.front_end_config.transport)
                full_url = f"http://{self.front_end_config.host}:{self.front_end_config.port}/mcp"
                logger.info("MCP server URL: %s", full_url)
                await mcp.run_async(
                    transport="streamable-http",
                    host=self.front_end_config.host,
                    port=self.front_end_config.port,
                    path="/mcp",
                    log_level=self.front_end_config.log_level.lower(),
                )
            except KeyboardInterrupt:
                logger.info("MCP server shutdown requested (Ctrl+C). Shutting down gracefully.")

    async def _run_with_mount(self, mcp: "FastMCP") -> None:
        """Run MCP server mounted at configured base_path using FastAPI wrapper.

        Args:
            mcp: The FastMCP server instance to mount
        """
        import uvicorn
        from fastapi import FastAPI

        mcp_app = mcp.http_app(transport="streamable-http", path="/mcp")

        # Create a FastAPI wrapper app with MCP app lifespan management
        app = FastAPI(
            title=self.front_end_config.name,
            description="MCP server mounted at custom base path",
            lifespan=mcp_app.lifespan,
        )

        # Mount the MCP server's ASGI app at the configured base_path
        app.mount(self.front_end_config.base_path, mcp_app)

        # Allow plugins to add routes to the wrapper app (e.g., OAuth discovery endpoints)
        worker = self._get_worker_instance()
        await worker.add_root_level_routes(app, mcp)

        # Configure and start uvicorn server
        config = uvicorn.Config(
            app,
            host=self.front_end_config.host,
            port=self.front_end_config.port,
            log_level=self.front_end_config.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()
