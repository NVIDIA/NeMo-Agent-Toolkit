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
#
# MCP tool list endpoint wiring for FastAPI front end.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from nat.builder.function import Function

if TYPE_CHECKING:
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

logger = logging.getLogger(__name__)


async def register_mcp_routes(worker: FastApiFrontEndPluginWorker, app: FastAPI, builder: WorkflowBuilder) -> None:
    """Add the MCP client tool list endpoint to the FastAPI app."""

    class MCPToolInfo(BaseModel):
        name: str
        description: str
        server: str
        available: bool

    class MCPClientToolListResponse(BaseModel):
        mcp_clients: list[dict[str, Any]]

    async def get_mcp_client_tool_list() -> MCPClientToolListResponse:
        """Get MCP client tool details, server status, and workflow mapping."""
        mcp_clients_info: list[dict[str, Any]] = []

        try:
            function_groups = builder._function_groups

            for group_name, configured_group in function_groups.items():
                if configured_group.config.type != "mcp_client":
                    continue

                from nat.plugins.mcp.client.client_config import MCPClientConfig

                config = cast(MCPClientConfig | Any, configured_group.config)
                group_instance = cast(Any, configured_group.instance)
                client = getattr(group_instance, "mcp_client", None)
                if client is None:
                    raise RuntimeError(f"MCP client not found for group {group_name}")

                # Collect server tools
                session_healthy = False
                server_tools: dict[str, Any] = {}
                try:
                    server_tools = await client.get_tools()
                    session_healthy = True
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to connect to MCP server %s: %s", getattr(client, "server_name", "?"), exc)

                # Collect configured workflow tools
                configured_short_names: set[str] = set()
                configured_full_to_fn: dict[str, Function] = {}
                try:

                    async def pass_through_filter(fn):
                        return fn

                    accessible_functions = await group_instance.get_accessible_functions(filter_fn=pass_through_filter)
                    configured_full_to_fn = accessible_functions
                    configured_short_names = {name.split('.', 1)[1] for name in accessible_functions.keys()}
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to get accessible functions for group %s: %s", group_name, exc)

                # Map overrides/aliases
                alias_to_original: dict[str, str] = {}
                override_configs: dict[str, Any] = {}
                try:
                    tool_overrides = getattr(config, "tool_overrides", None) or {}
                    for name, override in tool_overrides.items():
                        if override.alias:
                            alias_to_original[override.alias] = name
                        override_configs[name] = {
                            "description": override.description,
                            "parameters": getattr(override, "parameters", None),
                        }
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to process overrides for group %s: %s", group_name, exc)

                # Build tool list
                tools_info = []
                available_tools = 0
                all_tool_names = set(server_tools.keys()) | configured_short_names | set(alias_to_original.keys())

                for short_name in all_tool_names:
                    original_name = alias_to_original.get(short_name, short_name)
                    server_tool = server_tools.get(original_name)
                    is_available = bool(server_tool is not None)
                    if is_available:
                        available_tools += 1

                    description = None
                    if original_name in override_configs and override_configs[original_name].get("description"):
                        description = override_configs[original_name]["description"]
                    elif server_tool and server_tool.get("description"):
                        description = server_tool["description"]
                    elif f"{group_name}.{short_name}" in configured_full_to_fn:
                        description = configured_full_to_fn[f"{group_name}.{short_name}"].description

                    tools_info.append(
                        MCPToolInfo(name=short_name,
                                    description=description or "No description available",
                                    server=str(getattr(config, "server", "unknown")),
                                    available=is_available))

                mcp_clients_info.append({
                    "function_group": group_name,
                    "server": str(getattr(config, "server", "unknown")),
                    "transport": getattr(config, "transport", None) or "unknown",
                    "session_healthy": session_healthy,
                    "protected": bool(getattr(config, "protected_tools", None)),
                    "tools": tools_info,
                    "total_tools": len(tools_info),
                    "available_tools": available_tools,
                    "configured_overrides": len(alias_to_original),
                    "configured_tools": len(configured_short_names),
                    "workflow_tools": len(configured_full_to_fn),
                })

            return MCPClientToolListResponse(mcp_clients=mcp_clients_info)

        except Exception as exc:  # noqa: BLE001
            logger.error("Error in MCP client tool list endpoint: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to retrieve MCP client information: {exc}") from exc

    worker._register_api_route(
        app,
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


__all__ = ["register_mcp_routes"]
