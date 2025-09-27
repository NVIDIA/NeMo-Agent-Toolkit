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
from collections.abc import Mapping
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.exceptions import HTTPException
from starlette.requests import Request

from nat.builder.function import Function
from nat.builder.function_base import FunctionBase
from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig

logger = logging.getLogger(__name__)


class MCPFrontEndPluginWorkerBase(ABC):
    """Base class for MCP front end plugin workers."""

    def __init__(self, config: Config):
        """Initialize the MCP worker with configuration.

        Args:
            config: The full NAT configuration
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

    async def _get_all_functions(self, workflow: Workflow) -> dict[str, Function]:
        """Get all functions from the workflow.

        Args:
            workflow: The NAT workflow.

        Returns:
            Dict mapping function names to Function objects.
        """
        functions: dict[str, Function] = {}

        # Extract all functions from the workflow
        functions.update(workflow.functions)
        for function_group in workflow.function_groups.values():
            functions.update(await function_group.get_accessible_functions())

        if workflow.config.workflow.workflow_alias:
            functions[workflow.config.workflow.workflow_alias] = workflow
        else:
            functions[workflow.config.workflow.type] = workflow

        return functions

    def _setup_debug_endpoints(self, mcp: FastMCP, functions: Mapping[str, FunctionBase]) -> None:
        """Set up HTTP debug endpoints for introspecting tools and schemas.

        Exposes:
          - GET /debug/tools/list: List tools. Optional query param `name` (one or more, repeatable or comma separated)
            selects a subset and returns details for those tools.
        """

        @mcp.custom_route("/debug/tools/list", methods=["GET"])
        async def list_tools(request: Request):
            """HTTP list tools endpoint."""

            from starlette.responses import JSONResponse

            from nat.front_ends.mcp.tool_converter import get_function_description

            # Query params
            # Support repeated names and comma-separated lists
            names_param_list = set(request.query_params.getlist("name"))
            names: list[str] = []
            for raw in names_param_list:
                # if p.strip() is empty, it won't be included in the list!
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                names.extend(parts)
            detail_raw = request.query_params.get("detail")

            def _parse_detail_param(detail_param: str | None, has_names: bool) -> bool:
                if detail_param is None:
                    if has_names:
                        return True
                    return False
                v = detail_param.strip().lower()
                if v in ("0", "false", "no", "off"):
                    return False
                if v in ("1", "true", "yes", "on"):
                    return True
                # For invalid values, default based on whether names are present
                return has_names

            # Helper function to build the input schema info
            def _build_schema_info(fn: FunctionBase) -> dict[str, Any] | None:
                schema = getattr(fn, "input_schema", None)
                if schema is None:
                    return None

                # check if schema is a ChatRequest
                schema_name = getattr(schema, "__name__", "")
                schema_qualname = getattr(schema, "__qualname__", "")
                if "ChatRequest" in schema_name or "ChatRequest" in schema_qualname:
                    # Simplified interface used by MCP wrapper for ChatRequest
                    return {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string", "description": "User query string"
                            }
                        },
                        "required": ["query"],
                        "title": "ChatRequestQuery",
                    }

                # Pydantic models provide model_json_schema
                if schema is not None and hasattr(schema, "model_json_schema"):
                    return schema.model_json_schema()

                return None

            def _build_final_json(functions_to_include: Mapping[str, FunctionBase],
                                  include_schemas: bool = False) -> dict[str, Any]:
                tools = []
                for name, fn in functions_to_include.items():
                    list_entry: dict[str, Any] = {
                        "name": name, "description": get_function_description(fn), "is_workflow": hasattr(fn, "run")
                    }
                    if include_schemas:
                        list_entry["schema"] = _build_schema_info(fn)
                    tools.append(list_entry)

                return {
                    "count": len(tools),
                    "tools": tools,
                    "server_name": mcp.name,
                }

            if names:
                # Return selected tools
                try:
                    functions_to_include = {n: functions[n] for n in names}
                except KeyError as e:
                    raise HTTPException(status_code=404, detail=f"Tool \"{e.args[0]}\" not found.") from e
            else:
                functions_to_include = functions

            # Default for listing all: detail defaults to False unless explicitly set true
            return JSONResponse(
                _build_final_json(functions_to_include, _parse_detail_param(detail_raw, has_names=bool(names))))


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
        workflow = await builder.build()

        # Get all functions from the workflow
        functions = await self._get_all_functions(workflow)

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

        # Register each function with MCP, passing workflow context for observability
        for function_name, function in functions.items():
            register_function_with_mcp(mcp, function_name, function, workflow)

        # Add a simple fallback function if no functions were found
        if not functions:
            raise RuntimeError("No functions found in workflow. Please check your configuration.")

        # After registration, expose debug endpoints for tool/schema inspection
        self._setup_debug_endpoints(mcp, functions)
