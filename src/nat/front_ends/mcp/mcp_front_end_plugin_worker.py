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
from typing import TYPE_CHECKING
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.exceptions import HTTPException
from starlette.requests import Request

if TYPE_CHECKING:
    from fastapi import FastAPI

from nat.builder.function import Function
from nat.builder.function_base import FunctionBase
from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig
from nat.front_ends.mcp.memory_profiler import MemoryProfiler

logger = logging.getLogger(__name__)


class MCPFrontEndPluginWorkerBase(ABC):
    """Base class for MCP front end plugin workers.

    This abstract base class provides shared utilities and defines the contract
    for MCP worker implementations. Most users should inherit from
    MCPFrontEndPluginWorker instead of this class directly.
    """

    def __init__(self, config: Config):
        """Initialize the MCP worker with configuration.

        Args:
            config: The full NAT configuration
        """
        self.full_config = config
        self.front_end_config: MCPFrontEndConfig = config.general.front_end

        # Initialize memory profiler if enabled
        self.memory_profiler = MemoryProfiler(enabled=self.front_end_config.enable_memory_profiling,
                                              log_interval=self.front_end_config.memory_profile_interval,
                                              top_n=self.front_end_config.memory_profile_top_n,
                                              log_level=self.front_end_config.memory_profile_log_level)

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
    async def create_mcp_server(self) -> FastMCP:
        """Create and configure the MCP server instance.

        This is the main extension point. Plugins can return FastMCP or any subclass
        to customize server behavior (for example, add authentication, custom transports).

        Returns:
            FastMCP instance or a subclass with custom behavior
        """
        ...

    @abstractmethod
    async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
        """Add routes to the MCP server.

        Plugins must implement this method. Most plugins can call
        _default_add_routes() for standard behavior and then add
        custom enhancements.

        Args:
            mcp: The FastMCP server instance
            builder: The workflow builder instance
        """
        ...

    async def _default_add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
        """Default route registration logic - reusable by subclasses.

        This is a protected helper method that plugins can call to get
        standard route registration behavior. Plugins typically call this
        from their add_routes() implementation and then add custom features.

        This method:
        - Sets up the health endpoint
        - Builds the workflow and extracts all functions
        - Filters functions based on tool_names config
        - Registers each function as an MCP tool
        - Sets up debug endpoints for tool introspection

        Args:
            mcp: The FastMCP server instance
            builder: The workflow builder instance
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
                    # Treat current tool_names as function names, so check if the function name is in the list
                    filtered_functions[function_name] = function
                elif any(function_name.startswith(f"{group_name}.") for group_name in self.front_end_config.tool_names):
                    # Treat tool_names as function group names, so check if the function name starts with the group name
                    filtered_functions[function_name] = function
                else:
                    logger.debug("Skipping function %s as it's not in tool_names", function_name)
            functions = filtered_functions

        # Register each function with MCP, passing workflow context for observability
        for function_name, function in functions.items():
            register_function_with_mcp(mcp, function_name, function, workflow, self.memory_profiler)

        # Add a simple fallback function if no functions were found
        if not functions:
            raise RuntimeError("No functions found in workflow. Please check your configuration.")

        # After registration, expose debug endpoints for tool/schema inspection
        self._setup_debug_endpoints(mcp, functions)

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

    async def add_root_level_routes(self, wrapper_app: "FastAPI", mcp: FastMCP) -> None:
        """Add routes to the wrapper FastAPI app (optional extension point).

        This method is called when base_path is configured and a wrapper
        FastAPI app is created to mount the MCP server. Plugins can override
        this to add routes to the wrapper app at the root level, outside the
        mounted MCP server path.

        Common use cases:
        - OAuth discovery endpoints (e.g., /.well-known/oauth-protected-resource)
        - Health checks at root level
        - Static file serving
        - Custom authentication/authorization endpoints

        Default implementation does nothing, making this an optional extension point.

        Args:
            wrapper_app: The FastAPI wrapper application that mounts the MCP server
            mcp: The FastMCP server instance (already mounted at base_path)
        """
        pass  # Default: no additional root-level routes

    def _setup_debug_endpoints(self, mcp: FastMCP, functions: Mapping[str, FunctionBase]) -> None:
        """Set up HTTP debug endpoints for introspecting tools and schemas.

        Exposes:
          - GET /debug/tools/list: List tools. Optional query param `name` (one or more, repeatable or comma separated)
            selects a subset and returns details for those tools.
          - GET /debug/memory/stats: Get current memory profiling statistics (read-only)
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

        # Memory profiling endpoint (read-only)
        @mcp.custom_route("/debug/memory/stats", methods=["GET"])
        async def get_memory_stats(_request: Request):
            """Get current memory profiling statistics."""
            from starlette.responses import JSONResponse

            stats = self.memory_profiler.get_stats()
            return JSONResponse(stats)


class MCPFrontEndPluginWorker(MCPFrontEndPluginWorkerBase):
    """Default MCP server worker implementation.

    Inherit from this class to create custom MCP workers that extend or modify
    server behavior. Override create_mcp_server() to use a different server type,
    and override add_routes() to add custom functionality.

    Example:
        class CustomWorker(MCPFrontEndPluginWorker):
            async def create_mcp_server(self):
                # Return custom MCP server instance
                return MyCustomFastMCP(...)

            async def add_routes(self, mcp, builder):
                # Get default routes
                await super().add_routes(mcp, builder)
                # Add custom features
                self._add_my_custom_features(mcp)
    """

    async def create_mcp_server(self) -> FastMCP:
        """Create default MCP server with optional authentication.

        Returns:
            FastMCP instance configured with settings from NAT config
        """
        # Handle auth if configured
        auth_settings = None
        token_verifier = None

        if self.front_end_config.server_auth:
            from mcp.server.auth.settings import AuthSettings
            from pydantic import AnyHttpUrl

            server_url = f"http://{self.front_end_config.host}:{self.front_end_config.port}"
            auth_settings = AuthSettings(issuer_url=AnyHttpUrl(self.front_end_config.server_auth.issuer_url),
                                         required_scopes=self.front_end_config.server_auth.scopes,
                                         resource_server_url=AnyHttpUrl(server_url))

            # Create token verifier
            from nat.front_ends.mcp.introspection_token_verifier import IntrospectionTokenVerifier

            token_verifier = IntrospectionTokenVerifier(self.front_end_config.server_auth)

        return FastMCP(name=self.front_end_config.name,
                       host=self.front_end_config.host,
                       port=self.front_end_config.port,
                       debug=self.front_end_config.debug,
                       auth=auth_settings,
                       token_verifier=token_verifier)

    async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
        """Add default routes to the MCP server.

        Args:
            mcp: The FastMCP server instance
            builder: The workflow builder instance
        """
        # Use the default implementation from base class to add the tools to the MCP server
        await self._default_add_routes(mcp, builder)
