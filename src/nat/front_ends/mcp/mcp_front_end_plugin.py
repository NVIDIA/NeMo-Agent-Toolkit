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

if typing.TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

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

            # Start the MCP server with configurable transport
            # streamable-http is the default, but users can choose sse if preferred
            try:
                # If base_path is configured, mount server at sub-path using FastAPI wrapper
                if self.front_end_config.base_path:
                    if self.front_end_config.transport == "sse":
                        logger.warning(
                            "base_path is configured but SSE transport does not support mounting at sub-paths. "
                            "Use streamable-http transport for base_path support.")
                        logger.info("Starting MCP server with SSE endpoint at /sse")
                        await mcp.run_sse_async()
                    else:
                        full_url = f"http://{self.front_end_config.host}:{self.front_end_config.port}{self.front_end_config.base_path}/mcp"
                        logger.info(
                            "Mounting MCP server at %s/mcp on %s:%s",
                            self.front_end_config.base_path,
                            self.front_end_config.host,
                            self.front_end_config.port,
                        )
                        logger.info("MCP server URL: %s", full_url)
                        await self._run_with_mount(mcp)
                # Standard behavior - run at root path
                elif self.front_end_config.transport == "sse":
                    logger.info("Starting MCP server with SSE endpoint at /sse")
                    await mcp.run_sse_async()
                else:  # streamable-http
                    full_url = f"http://{self.front_end_config.host}:{self.front_end_config.port}/mcp"
                    logger.info("MCP server URL: %s", full_url)
                    await mcp.run_streamable_http_async()
            except KeyboardInterrupt:
                logger.info("MCP server shutdown requested (Ctrl+C). Shutting down gracefully.")

    async def _run_with_mount(self, mcp: "FastMCP") -> None:
        """Run MCP server mounted at configured base_path using FastAPI wrapper.

        Args:
            mcp: The FastMCP server instance to mount
        """
        import contextlib

        import uvicorn
        from fastapi import FastAPI

        @contextlib.asynccontextmanager
        async def lifespan(_app: FastAPI):
            """Manage MCP server session lifecycle."""
            logger.info("Starting MCP server session manager...")
            async with contextlib.AsyncExitStack() as stack:
                try:
                    # Initialize the MCP server's session manager
                    await stack.enter_async_context(mcp.session_manager.run())
                    logger.info("MCP server session manager started successfully")
                    yield
                except Exception as e:
                    logger.error("Failed to start MCP server session manager: %s", e)
                    raise
            logger.info("MCP server session manager stopped")

        # Create a FastAPI wrapper app with lifespan management
        app = FastAPI(
            title=self.front_end_config.name,
            description="MCP server mounted at custom base path",
            lifespan=lifespan,
        )

        # Mount the MCP server's ASGI app at the configured base_path
        app.mount(self.front_end_config.base_path, mcp.streamable_http_app())

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
