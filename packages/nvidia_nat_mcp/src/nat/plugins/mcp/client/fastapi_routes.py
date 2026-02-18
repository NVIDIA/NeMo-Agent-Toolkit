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
"""FastAPI routes for MCP client tool listing."""

import logging
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel

from nat.builder.function import FunctionGroup
from nat.builder.workflow_builder import WorkflowBuilder
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class MCPToolInfo(BaseModel):
    """Information about a single MCP tool."""

    name: str
    description: str
    server: str
    available: bool


class MCPClientToolListResponse(BaseModel):
    """Response model for the MCP client tool list endpoint."""

    mcp_clients: list[dict[str, Any]]


async def _collect_mcp_client_tool_list(function_groups: dict[str, FunctionGroup]) -> list[dict[str, Any]]:
    """Collect MCP client tool information from all function groups.

    Iterates over function groups, identifies MCP client groups, checks session
    health, and builds a list of tool information including availability status.
    """
    mcp_clients_info: list[dict[str, Any]] = []

    for group_name, group_instance in function_groups.items():
        config = group_instance.get_config()
        if config.type not in {"mcp_client", "per_user_mcp_client"}:
            continue

        client = getattr(group_instance, "mcp_client", None)
        if client is None:
            raise RuntimeError(f"MCP client not found for group {group_name}")

        try:
            session_healthy = False
            server_tools: dict[str, Any] = {}

            try:
                server_tools = await client.get_tools()
                session_healthy = True
            except Exception as e:
                logger.exception(f"Failed to connect to MCP server {client.server_name}: {e}")
                session_healthy = False

            # Get workflow function group configuration (configured client-side tools)
            configured_short_names: list[str] = []
            configured_full_to_fn: dict[str, Any] = {}
            try:

                # Pass a no-op filter function to bypass any default filtering that might check
                # health status, preventing potential infinite recursion during health status checks.
                async def pass_through_filter(fn):
                    return fn

                accessible_functions = await group_instance.get_accessible_functions(filter_fn=pass_through_filter)
                configured_full_to_fn = accessible_functions
                configured_short_names = []
                for name in accessible_functions.keys():
                    if FunctionGroup.SEPARATOR in name:
                        configured_short_names.append(name.split(FunctionGroup.SEPARATOR, 1)[1])
                    elif FunctionGroup.LEGACY_SEPARATOR in name:
                        configured_short_names.append(name.split(FunctionGroup.LEGACY_SEPARATOR, 1)[1])
                    else:
                        configured_short_names.append(name)
            except Exception as e:
                logger.exception(f"Failed to get accessible functions for group {group_name}: {e}")

            # Build alias->original mapping and override configs from overrides
            alias_to_original: dict[str, str] = {}
            override_configs: dict[str, Any] = {}
            try:
                if config.tool_overrides is not None:
                    for orig_name, override in config.tool_overrides.items():
                        if override.alias is not None:
                            alias_to_original[override.alias] = orig_name
                            override_configs[override.alias] = override
                        else:
                            override_configs[orig_name] = override
            except Exception as e:
                logger.exception("Error processing tool overrides for MCP client group %s: %s", group_name, e)

            # Create tool info list (always return configured tools; mark availability)
            tools_info: list[dict[str, Any]] = []
            available_count = 0
            for full_name, wf_fn in configured_full_to_fn.items():
                if FunctionGroup.SEPARATOR in full_name:
                    fn_short = full_name.split(FunctionGroup.SEPARATOR, 1)[1]
                elif FunctionGroup.LEGACY_SEPARATOR in full_name:
                    fn_short = full_name.split(FunctionGroup.LEGACY_SEPARATOR, 1)[1]
                else:
                    fn_short = full_name
                orig_name = alias_to_original.get(fn_short, fn_short)
                available = session_healthy and (orig_name in server_tools)
                if available:
                    available_count += 1

                # Prefer tool override description, then workflow function description,
                # then server description
                description = ""
                if fn_short in override_configs and override_configs[fn_short].description:
                    description = override_configs[fn_short].description
                elif wf_fn.description:
                    description = wf_fn.description
                elif available and orig_name in server_tools:
                    description = server_tools[orig_name].description or ""

                tools_info.append(
                    MCPToolInfo(name=fn_short,
                                description=description or "",
                                server=client.server_name,
                                available=available).model_dump())

            # Sort tools_info by name to maintain consistent ordering
            tools_info.sort(key=lambda x: x['name'])

            mcp_clients_info.append({
                "function_group": group_name,
                "server": client.server_name,
                "transport": config.server.transport,
                "session_healthy": session_healthy,
                "protected": True if config.server.auth_provider is not None else False,
                "tools": tools_info,
                "total_tools": len(configured_short_names),
                "available_tools": available_count
            })

        except Exception as e:
            logger.exception("Error processing MCP client %s", group_name)
            mcp_clients_info.append({
                "function_group": group_name,
                "server": "unknown",
                "transport": config.server.transport if config.server else "unknown",
                "session_healthy": False,
                "protected": False,
                "error": str(e),
                "tools": [],
                "total_tools": 0,
                "available_tools": 0
            })

    return mcp_clients_info


async def add_mcp_client_tool_list_route(app: FastAPI, builder: WorkflowBuilder,
                                         session_managers: list[SessionManager]) -> None:
    """Add MCP client tool list endpoints to the FastAPI app.

    Registers two GET routes:
    - ``/mcp/client/tool/list`` for shared workflows.
    - ``/mcp/client/tool/list/per_user`` for per-user workflows.
    """

    async def get_mcp_client_tool_list() -> MCPClientToolListResponse:
        """Get the list of MCP tools from all MCP clients in the workflow configuration.

        Checks session health and compares with workflow function group configuration.
        """
        try:
            # Get all function groups from the builder
            function_groups = {name: cfg.instance for name, cfg in builder._function_groups.items()}
            mcp_clients_info = await _collect_mcp_client_tool_list(function_groups)
            return MCPClientToolListResponse(mcp_clients=mcp_clients_info)

        except Exception as e:
            logger.error(f"Error in MCP client tool list endpoint: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve MCP client information: {str(e)}") from e

    async def get_per_user_mcp_client_tool_list(
        request: Request,
        user_id: str | None = None,
    ) -> MCPClientToolListResponse:
        """Get the list of MCP tools for a specific user in per-user workflows.

        Uses the per-user workflow builder to resolve function groups and
        applies the same MCP client inspection logic as the shared endpoint.
        """
        per_user_manager = next((sm for sm in session_managers if sm.is_workflow_per_user), None)
        if per_user_manager is None:
            raise HTTPException(status_code=400, detail="No per-user workflow is configured.")

        try:
            async with per_user_manager.session(user_id=user_id, http_connection=request) as session:
                mcp_clients_info = await _collect_mcp_client_tool_list(session.workflow.function_groups)
                return MCPClientToolListResponse(mcp_clients=mcp_clients_info)
        except Exception as e:
            logger.exception("Error in per-user MCP client tool list endpoint: %s", e)
            raise HTTPException(status_code=500,
                                detail=f"Failed to retrieve per-user MCP client information: {str(e)}") from e

    # Add the route to the FastAPI app
    app.add_api_route(
        path="/mcp/client/tool/list",
        endpoint=get_mcp_client_tool_list,
        methods=["GET"],
        response_model=MCPClientToolListResponse,
        description="Get list of MCP client tools with session health and workflow configuration comparison",
        responses={
            200: {
                "description": "Successfully retrieved MCP client tool information",
                "content": {
                    "application/json": {
                        "example": {
                            "mcp_clients": [{
                                "function_group": "mcp_tools",
                                "server": "streamable-http:http://localhost:9901/mcp",
                                "transport": "streamable-http",
                                "session_healthy": True,
                                "protected": False,
                                "tools": [{
                                    "name": "tool_a",
                                    "description": "Tool A description",
                                    "server": "streamable-http:http://localhost:9901/mcp",
                                    "available": True
                                }],
                                "total_tools": 1,
                                "available_tools": 1
                            }]
                        }
                    }
                }
            },
            500: {
                "description": "Internal Server Error"
            }
        })

    app.add_api_route(path="/mcp/client/tool/list/per_user",
                      endpoint=get_per_user_mcp_client_tool_list,
                      methods=["GET"],
                      response_model=MCPClientToolListResponse,
                      description="Get list of MCP client tools for per-user workflows",
                      responses={
                          200: {
                              "description": "Successfully retrieved per-user MCP client tool information"
                          },
                          400: {
                              "description": "No per-user workflow is configured"
                          },
                          500: {
                              "description": "Internal Server Error"
                          }
                      })
