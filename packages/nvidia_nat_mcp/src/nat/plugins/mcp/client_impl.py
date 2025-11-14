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

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.plugins.mcp.client_base import MCPBaseClient
from nat.plugins.mcp.client_config import MCPClientConfig
from nat.plugins.mcp.client_config import MCPToolOverrideConfig
from nat.plugins.mcp.tool import mcp_tool_function

logger = logging.getLogger(__name__)


class MCPFunctionGroup(FunctionGroup):
    """
    A specialized FunctionGroup for MCP clients that includes MCP-specific attributes.
    """

    def __init__(self, client: MCPBaseClient | None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mcp_client = client

    @property
    def mcp_client(self):
        """Get the MCP client instance."""
        return self._mcp_client


@register_function_group(config_type=MCPClientConfig)
async def mcp_client_function_group(config: MCPClientConfig, _builder: Builder):
    """
    Connect to an MCP server and expose tools as a function group.

    Args:
        config: The configuration for the MCP client
        _builder: The builder
    Returns:
        The function group
    """
    from nat.plugins.mcp.client_base import MCPSSEClient
    from nat.plugins.mcp.client_base import MCPStdioClient
    from nat.plugins.mcp.client_base import MCPStreamableHTTPClient

    # Resolve auth provider if specified
    auth_provider = None
    if config.server.auth_provider:
        auth_provider = await _builder.get_auth_provider(config.server.auth_provider)

    # Build the appropriate client
    if config.server.transport == "stdio":
        if not config.server.command:
            raise ValueError("command is required for stdio transport")
        client = MCPStdioClient(config.server.command,
                                config.server.args,
                                config.server.env,
                                tool_call_timeout=config.tool_call_timeout,
                                auth_flow_timeout=config.auth_flow_timeout,
                                reconnect_enabled=config.reconnect_enabled,
                                reconnect_max_attempts=config.reconnect_max_attempts,
                                reconnect_initial_backoff=config.reconnect_initial_backoff,
                                reconnect_max_backoff=config.reconnect_max_backoff)
    elif config.server.transport == "sse":
        client = MCPSSEClient(str(config.server.url),
                              tool_call_timeout=config.tool_call_timeout,
                              auth_flow_timeout=config.auth_flow_timeout,
                              reconnect_enabled=config.reconnect_enabled,
                              reconnect_max_attempts=config.reconnect_max_attempts,
                              reconnect_initial_backoff=config.reconnect_initial_backoff,
                              reconnect_max_backoff=config.reconnect_max_backoff)
    elif config.server.transport == "streamable-http":
        nat_context = Context.get()
        user_id = nat_context.user_id
        client = MCPStreamableHTTPClient(str(config.server.url),
                                         auth_provider=auth_provider,
                                         user_id=user_id,
                                         tool_call_timeout=config.tool_call_timeout,
                                         auth_flow_timeout=config.auth_flow_timeout,
                                         reconnect_enabled=config.reconnect_enabled,
                                         reconnect_max_attempts=config.reconnect_max_attempts,
                                         reconnect_initial_backoff=config.reconnect_initial_backoff,
                                         reconnect_max_backoff=config.reconnect_max_backoff)
    else:
        raise ValueError(f"Unsupported transport: {config.server.transport}")

    logger.info("Configured to use MCP server at %s", client.server_name)

    async with client:
        # Expose the live MCP client on the function group instance so other components (e.g., HTTP endpoints)
        # can reuse the already-established session instead of creating a new client per request.
        group = MCPFunctionGroup(client, config=config)

        all_tools = await client.get_tools()
        tool_overrides = mcp_apply_tool_alias_and_description(all_tools, config.tool_overrides)

        # Add each tool as a function to the group
        for tool_name, tool in all_tools.items():
            # Get override if it exists
            override = tool_overrides.get(tool_name)

            # Use override values or defaults
            function_name = override.alias if override and override.alias else tool_name
            description = override.description if override and override.description else tool.description

            # Create the tool function according to configuration
            tool_fn = mcp_tool_function(tool)

            # Add to group
            logger.info("Adding tool %s to group", function_name)
            group.add_function(name=function_name,
                               description=description,
                               fn=tool_fn.single_fn,
                               input_schema=tool_fn.input_schema,
                               converters=tool_fn.converters)

        yield group


def mcp_apply_tool_alias_and_description(
        all_tools: dict, tool_overrides: dict[str, MCPToolOverrideConfig] | None) -> dict[str, MCPToolOverrideConfig]:
    """
    Filter tool overrides to only include tools that exist in the MCP server.

    Args:
        all_tools: The tools from the MCP server
        tool_overrides: The tool overrides to apply
    Returns:
        Dictionary of valid tool overrides
    """
    if not tool_overrides:
        return {}

    return {name: override for name, override in tool_overrides.items() if name in all_tools}
