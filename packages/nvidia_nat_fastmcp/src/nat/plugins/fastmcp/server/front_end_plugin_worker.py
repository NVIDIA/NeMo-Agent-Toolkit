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
"""FastMCP front end worker implementation."""

import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any

from starlette.exceptions import HTTPException
from starlette.requests import Request

from fastmcp import FastMCP

if TYPE_CHECKING:
    from fastapi import FastAPI

from nat.builder.function import Function
from nat.builder.function_base import FunctionBase
from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.plugins.fastmcp.server.front_end_config import FastMCPFrontEndConfig
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class FastMCPFrontEndPluginWorkerBase(ABC):
    """Base class for FastMCP front end plugin workers."""

    def __init__(self, config: Config):
        """Initialize the FastMCP worker with configuration.

        Args:
            config: The full NeMo Agent Toolkit configuration.
        """
        self.full_config = config
        self.front_end_config: FastMCPFrontEndConfig = config.general.front_end

    def _setup_health_endpoint(self, mcp: FastMCP):
        """Set up the HTTP health endpoint that exercises FastMCP ping handler."""

        @mcp.custom_route("/health", methods=["GET"])
        async def health_check(_request: Request):
            """HTTP health check using server's internal ping handler."""
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

            except Exception:
                health_logger = getattr(mcp, "logger", None) or logging.getLogger(__name__)
                health_logger.exception("Health check failed while invoking PingRequest")
                return JSONResponse({
                    "status": "unhealthy",
                    "error": "internal server error",
                    "server_name": mcp.name,
                },
                                    status_code=503)

    @abstractmethod
    async def create_mcp_server(self) -> FastMCP:
        """Create and configure the FastMCP server instance.

        Returns:
            FastMCP instance or a subclass with custom behavior
        """
        ...

    @abstractmethod
    async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
        """Add routes to the FastMCP server.

        Args:
            mcp: The FastMCP server instance
            builder: The workflow builder instance
        """
        ...

    async def _default_add_routes(self, mcp: FastMCP, builder: WorkflowBuilder) -> None:
        """Default implementation for adding routes to FastMCP."""
        from nat.plugins.fastmcp.server.tool_converter import register_function_with_mcp

        # Set up the health endpoint
        self._setup_health_endpoint(mcp)

        # Build the default workflow
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
                elif any(function_name.startswith(f"{group_name}.") for group_name in self.front_end_config.tool_names):
                    filtered_functions[function_name] = function
                else:
                    logger.debug("Skipping function %s as it's not in tool_names", function_name)
            functions = filtered_functions

        # Create SessionManagers for each function
        session_managers: dict[str, SessionManager] = {}
        for function_name, function in functions.items():
            if isinstance(function, Workflow):
                logger.info("Function %s is a Workflow, using directly", function_name)
                session_managers[function_name] = await SessionManager.create(config=self.full_config,
                                                                              shared_builder=builder,
                                                                              entry_function=None)
            else:
                logger.info("Function %s is a regular function, building entry workflow", function_name)
                session_managers[function_name] = await SessionManager.create(config=self.full_config,
                                                                              shared_builder=builder,
                                                                              entry_function=function_name)

        # Register each function with FastMCP, passing SessionManager for observability
        for function_name, session_manager in session_managers.items():
            register_function_with_mcp(mcp, function_name, session_manager, function=functions.get(function_name))

        if not session_managers:
            raise RuntimeError("No functions found in workflow. Please check your configuration.")

        # After registration, expose debug endpoints for tool/schema inspection
        debug_functions = {name: sm.workflow for name, sm in session_managers.items()}
        self._setup_debug_endpoints(mcp, debug_functions)

    async def _get_all_functions(self, workflow: Workflow) -> dict[str, Function]:
        """Get all functions from the workflow.

        Args:
            workflow: The NeMo Agent Toolkit workflow.

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

        Args:
            wrapper_app: The FastAPI wrapper application that mounts the FastMCP server
            mcp: The FastMCP server instance (already mounted at base_path)
        """
        return None

    def _setup_debug_endpoints(self, mcp: FastMCP, functions: Mapping[str, FunctionBase]) -> None:
        """Set up HTTP debug endpoints for introspecting tools and schemas."""

        @mcp.custom_route("/debug/tools/list", methods=["GET"])
        async def list_tools(request: Request):
            """HTTP list tools endpoint."""

            from starlette.responses import JSONResponse

            from nat.plugins.fastmcp.server.tool_converter import get_function_description

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
                        "name": name,
                        "description": get_function_description(fn),
                    }
                    if include_schemas:
                        list_entry["input_schema"] = _build_schema_info(fn)
                    tools.append(list_entry)
                return {
                    "tools": tools,
                    "count": len(tools),
                }

            # Select specific tools if names provided
            if names:
                try:
                    functions_to_include = {n: functions[n] for n in names}
                except KeyError as e:
                    raise HTTPException(status_code=404, detail=f"Tool \"{e.args[0]}\" not found.") from e
            else:
                functions_to_include = functions

            # Default for listing all: detail defaults to False unless explicitly set true
            return JSONResponse(
                _build_final_json(functions_to_include, _parse_detail_param(detail_raw, has_names=bool(names))))


class FastMCPFrontEndPluginWorker(FastMCPFrontEndPluginWorkerBase):
    """Default FastMCP server worker implementation."""

    async def create_mcp_server(self) -> FastMCP:
        """Create default FastMCP server instance.

        Returns:
            FastMCP instance configured with settings from toolkit config.
        """
        auth_provider = None
        server_auth = self.front_end_config.server_auth
        if server_auth:
            from fastmcp.server.auth import RemoteAuthProvider
            from fastmcp.server.auth.providers.introspection import IntrospectionTokenVerifier

            verifier_kwargs = {
                "introspection_url": server_auth.introspection_endpoint,
                "client_id": server_auth.client_id,
                "client_secret": (server_auth.client_secret.get_secret_value() if server_auth.client_secret else None),
                "required_scopes": server_auth.scopes,
            }
            if server_auth.client_auth_method:
                verifier_kwargs["client_auth_method"] = server_auth.client_auth_method
            verifier = IntrospectionTokenVerifier(**verifier_kwargs)
            host = self.front_end_config.host
            if host in {"0.0.0.0", "::"}:
                host = "localhost"
            base_url = f"http://{host}:{self.front_end_config.port}"
            auth_provider = RemoteAuthProvider(
                token_verifier=verifier,
                authorization_servers=[server_auth.issuer_url],
                base_url=base_url,
                resource_name=self.front_end_config.name,
            )

        return FastMCP(
            name=self.front_end_config.name,
            debug=self.front_end_config.debug,
            auth=auth_provider,
        )

    async def add_routes(self, mcp: FastMCP, builder: WorkflowBuilder):
        """Add default routes to the FastMCP server.

        Args:
            mcp: The FastMCP server instance
            builder: The workflow builder instance
        """
        await self._default_add_routes(mcp, builder)
