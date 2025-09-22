# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any

from nat.builder import Builder
from nat.builder import LLMFrameworkEnum
from nat.experimental.decorators.experimental_warning_decorator import experimental

logger = logging.getLogger(__name__)


@experimental(feature_name="MCP Tools Refresh")
async def refresh_mcp_tools(builder: Builder) -> dict[str, Any]:
    """
    Refreshes all MCP client function groups in the builder.

    This function iterates through all function groups in the builder,
    identifies MCP client function groups, and calls get_accessible_functions_with_refresh
    with rebuild=True on each one.

    Parameters
    ----------
    builder : Builder
        The builder instance containing function groups to refresh.

    Returns
    -------
    dict[str, Any]
        Summary of refreshed function groups including:
        - refreshed_groups: List of function group names that were refreshed
        - total_groups: Total number of MCP function groups found
        - success: Boolean indicating if all refreshes succeeded
        - errors: List of any errors encountered during refresh
    """
    refreshed_groups = []
    errors = []

    try:
        # Get all function groups from the builder
        function_groups = getattr(builder, '_function_groups', {})

        # Filter for MCP client function groups
        mcp_groups = []
        for group_name, group_info in function_groups.items():
            # Check if this is an MCP client function group
            if group_info.instance._config.type == 'mcp_client':
                mcp_groups.append((group_name, group_info.instance))

        logger.info(f"Found {len(mcp_groups)} MCP client function groups to refresh")

        # Refresh each MCP function group
        for group_name, group_instance in mcp_groups:
            try:
                logger.info(f"Refreshing MCP function group: {group_name}")
                await group_instance.get_accessible_functions_with_refresh(rebuild=True)
                refreshed_groups.append(group_name)
                logger.info(f"Successfully refreshed MCP function group: {group_name}")
            except Exception as e:
                error_msg = f"Failed to refresh MCP function group '{group_name}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)

        success = len(errors) == 0

        result = {
            "refreshed_groups": refreshed_groups, "total_groups": len(mcp_groups), "success": success, "errors": errors
        }

        if success:
            logger.info(f"Successfully refreshed {len(refreshed_groups)} MCP function groups")
        else:
            logger.warning(f"Refreshed {len(refreshed_groups)} MCP function groups with {len(errors)} errors")

        return result

    except Exception as e:
        error_msg = f"Unexpected error during MCP function group refresh: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"refreshed_groups": refreshed_groups, "total_groups": 0, "success": False, "errors": [error_msg]}


@experimental(feature_name="Workflow Tools Refresh")
async def refresh_workflow_tools(builder: Builder) -> dict[str, Any]:
    """
    Refreshes all tools in a workflow by rebuilding them from the builder.

    This function first refreshes MCP function groups, then rebuilds all tools
    from the builder to ensure the workflow has the latest tool definitions.
    It automatically detects tool names from the workflow configuration.

    Parameters
    ----------
    builder : Builder
        The builder instance containing function groups to refresh.

    Returns
    -------
    dict[str, Any]
        Summary of the refresh operation including:
        - refreshed_groups: List of MCP function group names that were refreshed
        - refreshed_tools: List of tool names that were refreshed
        - total_groups: Total number of MCP function groups found
        - total_tools: Total number of tools refreshed
        - success: Boolean indicating if all refreshes succeeded
        - errors: List of any errors encountered during refresh
    """
    refreshed_groups = []
    refreshed_tools = []
    errors = []

    try:
        # First, refresh MCP function groups
        mcp_result = await refresh_mcp_tools(builder)
        refreshed_groups = mcp_result.get("refreshed_groups", [])
        errors.extend(mcp_result.get("errors", []))

        # Get all function groups to determine tool names
        function_groups = getattr(builder, '_function_groups', {})
        tool_names = list(function_groups.keys())

        # Then, rebuild all tools from the builder
        try:
            logger.info(f"Refreshing {len(tool_names)} tools from builder")
            new_tools = await builder.get_tools_with_refresh(tool_names=tool_names,
                                                             wrapper_type=LLMFrameworkEnum.LANGCHAIN)
            refreshed_tools = [tool.name for tool in new_tools]
            logger.info(f"Successfully refreshed {len(refreshed_tools)} tools")
        except Exception as e:
            error_msg = f"Failed to refresh tools from builder: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

        success = len(errors) == 0

        result = {
            "refreshed_groups": refreshed_groups,
            "refreshed_tools": refreshed_tools,
            "total_groups": mcp_result.get("total_groups", 0),
            "total_tools": len(refreshed_tools),
            "success": success,
            "errors": errors
        }

        if success:
            logger.info(f"Successfully refreshed {len(refreshed_groups)} MCP groups and {len(refreshed_tools)} tools")
        else:
            logger.warning(f"Refreshed with {len(errors)} errors")

        return result

    except Exception as e:
        error_msg = f"Unexpected error during workflow tool refresh: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "refreshed_groups": refreshed_groups,
            "refreshed_tools": refreshed_tools,
            "total_groups": 0,
            "total_tools": 0,
            "success": False,
            "errors": [error_msg]
        }
