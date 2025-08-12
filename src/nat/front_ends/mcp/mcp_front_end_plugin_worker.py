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
from abc import ABC
from abc import abstractmethod

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request

from nat.builder.function import Function
from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import AIQConfig
from nat.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig

logger = logging.getLogger(__name__)


class MCPFrontEndPluginWorkerBase(ABC):
    """Base class for MCP front end plugin workers."""

    def __init__(self, config: AIQConfig):
        """Initialize the MCP worker with configuration.

        Args:
            config: The full AIQ configuration
        """
        self.full_config = config
        self.front_end_config: MCPFrontEndConfig = config.general.front_end

    def _setup_health_endpoint(self, mcp: FastMCP):
        """Set up the HTTP health endpoint that exercises MCP ping handler."""

        @mcp.custom_route("/health", methods=["GET"])
        async def health_check(_request: Request):
            """HTTP health check using server's internal ping handler"""
            from starlette.responses import JSONResponse

            try:
                from mcp.types import PingRequest

                # Create a ping request
                ping_request = PingRequest(method="ping")

                # Call the ping handler directly (same one that responds to MCP pings)
                await mcp._mcp_server.request_handlers[PingRequest](ping_request)

                return JSONResponse({
                    "status": "healthy",
                    "error": None,
                    "server_name": mcp.name,
                })

            except Exception as e:
                return JSONResponse({
                    "status": "unhealthy",
                    "error": str(e),
                    "server_name": mcp.name,
                },
                                    status_code=503)

    @abstractmethod
    async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
        """Add routes to the MCP server.

        Args:
            mcp: The FastMCP server instance
            builder (WorkflowBuilder): The workflow builder instance
        """
        pass

    def _get_all_functions(self, workflow: Workflow) -> dict[str, Function]:
        """Get all functions from the workflow.

        Args:
            workflow: The AIQ workflow.

        Returns:
            Dict mapping function names to Function objects.
        """
        functions: dict[str, Function] = {}

        # Extract all functions from the workflow
        for function_name, function in workflow.functions.items():
            functions[function_name] = function

        functions[workflow.config.workflow.type] = workflow

        return functions


class MCPFrontEndPluginWorker(MCPFrontEndPluginWorkerBase):
    """Default MCP front end plugin worker implementation."""

    async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
        """Add default routes to the MCP server.

        Args:
            mcp: The FastMCP server instance
            builder (WorkflowBuilder): The workflow builder instance
        """
        from nat.front_ends.mcp.tool_converter import register_function_with_mcp

        # Set up the health endpoint
        self._setup_health_endpoint(mcp)

        # Build the workflow and register all functions with MCP
        workflow = builder.build()

        # Get all functions from the workflow
        functions = self._get_all_functions(workflow)

        # Filter functions based on tool_names if provided
        if self.front_end_config.tool_names:
            logger.info("Filtering functions based on tool_names: %s", self.front_end_config.tool_names)
            filtered_functions: dict[str, Function] = {}
            for function_name, function in functions.items():
                if function_name in self.front_end_config.tool_names:
                    filtered_functions[function_name] = function
                else:
                    logger.debug("Skipping function %s as it's not in tool_names", function_name)
            functions = filtered_functions

        # Register each function with MCP
        for function_name, function in functions.items():
            register_function_with_mcp(mcp, function_name, function)

        # Add a simple fallback function if no functions were found
        if not functions:
            raise RuntimeError("No functions found in workflow. Please check your configuration.")
