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

from aiq.builder.front_end import FrontEndBase
from aiq.builder.function import Function
from aiq.builder.workflow import Workflow
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig

logger = logging.getLogger(__name__)


class MCPFrontEndPlugin(FrontEndBase[MCPFrontEndConfig]):
    """MCP front end plugin implementation."""

    async def run(self) -> None:
        """Run the MCP server."""
        # Import FastMCP
        from mcp.server.fastmcp import FastMCP

        from aiq.front_ends.mcp.tool_converter import register_function_with_mcp

        # Create an MCP server with the configured parameters
        mcp = FastMCP(
            self.front_end_config.name,
            host=self.front_end_config.host,
            port=self.front_end_config.port,
            debug=self.front_end_config.debug,
            log_level=self.front_end_config.log_level,
        )

        # Build the workflow and register all functions with MCP
        async with WorkflowBuilder.from_config(config=self.full_config) as builder:
            # Build the workflow
            workflow = builder.build()

            # Get all functions from the workflow
            functions = self._get_all_functions(workflow)

            # Filter functions based on tool_names if provided
            if self.front_end_config.tool_names:
                logger.info(f"Filtering functions based on tool_names: {self.front_end_config.tool_names}")
                filtered_functions: dict[str, Function] = {}
                for function_name, function in functions.items():
                    if function_name in self.front_end_config.tool_names:
                        filtered_functions[function_name] = function
                    else:
                        logger.debug(f"Skipping function {function_name} as it's not in tool_names")
                functions = filtered_functions

            # Register each function with MCP
            for function_name, function in functions.items():
                register_function_with_mcp(mcp, function_name, function)

            # Add a simple fallback function if no functions were found
            if not functions:
                raise RuntimeError("No functions found in workflow. Please check your configuration.")

            # Start the MCP server
            await mcp.run_sse_async()

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
