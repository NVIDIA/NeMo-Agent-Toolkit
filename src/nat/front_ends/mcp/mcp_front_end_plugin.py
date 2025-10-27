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
                if self.front_end_config.transport == "sse":
                    logger.info("Starting MCP server with SSE endpoint at /sse")
                    await mcp.run_sse_async()
                else:  # streamable-http
                    logger.info("Starting MCP server with streamable-http endpoint at /mcp/")
                    await mcp.run_streamable_http_async()
            except KeyboardInterrupt:
                logger.info("MCP server shutdown requested (Ctrl+C). Shutting down gracefully.")
