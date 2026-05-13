# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""FastMCP front end plugin implementation."""

import logging
import typing

from nat.builder.front_end import FrontEndBase
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.fastmcp.server.front_end_config import FastMCPFrontEndConfig
from nat.plugins.fastmcp.server.front_end_plugin_worker import FastMCPFrontEndPluginWorkerBase

if typing.TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)


class FastMCPFrontEndPlugin(FrontEndBase[FastMCPFrontEndConfig]):
    """FastMCP front end plugin implementation."""

    def get_worker_class(self) -> type[FastMCPFrontEndPluginWorkerBase]:
        """Get the worker class for handling FastMCP routes."""
        from nat.plugins.fastmcp.server.front_end_plugin_worker import FastMCPFrontEndPluginWorker

        return FastMCPFrontEndPluginWorker

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
        """Run the FastMCP server."""
        async with WorkflowBuilder.from_config(config=self.full_config) as builder:

            # Get the worker instance
            worker = self._get_worker_instance()

            # Let the worker create the FastMCP server (allows plugins to customize)
            mcp = await worker.create_mcp_server()

            # Add routes through the worker (includes health endpoint and function registration)
            await worker.add_routes(mcp, builder)

            try:
                if self.front_end_config.base_path:
                    if self.front_end_config.transport == "sse":
                        logger.warning(
                            "base_path is configured but SSE transport does not support mounting at sub-paths. "
                            "Use streamable-http transport for base_path support.")
                        logger.info("Starting FastMCP server with SSE endpoint at /sse")
                        await mcp.run_async(transport="sse",
                                            host=self.front_end_config.host,
                                            port=self.front_end_config.port,
                                            log_level=self.front_end_config.log_level.lower())
                    else:
                        full_url = f"http://{self.front_end_config.host}:{self.front_end_config.port}{self.front_end_config.base_path}/mcp"
                        logger.info(
                            "Mounting FastMCP server at %s/mcp on %s:%s",
                            self.front_end_config.base_path,
                            self.front_end_config.host,
                            self.front_end_config.port,
                        )
                        logger.info("FastMCP server URL: %s", full_url)
                        await self._run_with_mount(mcp, worker)
                elif self.front_end_config.transport == "sse":
                    logger.info("Starting FastMCP server with SSE endpoint at /sse")
                    await mcp.run_async(transport="sse",
                                        host=self.front_end_config.host,
                                        port=self.front_end_config.port,
                                        log_level=self.front_end_config.log_level.lower())
                else:
                    full_url = f"http://{self.front_end_config.host}:{self.front_end_config.port}/mcp"
                    logger.info("FastMCP server URL: %s", full_url)
                    await mcp.run_async(transport="streamable-http",
                                        host=self.front_end_config.host,
                                        port=self.front_end_config.port,
                                        path="/mcp",
                                        log_level=self.front_end_config.log_level.lower())
            except KeyboardInterrupt:
                logger.info("FastMCP server shutdown requested (Ctrl+C). Shutting down gracefully.")

    async def _run_with_mount(self, mcp: "FastMCP", worker: FastMCPFrontEndPluginWorkerBase) -> None:
        """Run FastMCP server mounted at configured base_path using FastAPI wrapper.

        Args:
            mcp: The FastMCP server instance to mount.
            worker: The FastMCP worker instance.
        """
        import uvicorn
        from fastapi import FastAPI

        # Create FastAPI wrapper app with FastMCP lifecycle management
        mcp_app = mcp.http_app(transport="streamable-http", path="/mcp")
        app = FastAPI(
            title=self.front_end_config.name,
            description="FastMCP server mounted at custom base path",
            lifespan=mcp_app.lifespan,
        )

        # Mount the FastMCP server's ASGI app at the configured base_path
        app.mount(self.front_end_config.base_path, mcp_app)

        # Allow plugins to add routes to the wrapper app (e.g., OAuth discovery endpoints)
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
