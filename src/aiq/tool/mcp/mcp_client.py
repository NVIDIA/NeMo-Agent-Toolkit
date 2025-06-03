# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.client_functions import ClientFunctionConfig
from aiq.data_models.function import FunctionBaseConfig
from aiq.tool.client_functions import register_client_function_handler
from aiq.tool.mcp.mcp_client_base import MCPBaseClient

logger = logging.getLogger(__name__)

# Global client cache for sharing MCP clients
# TODO:this needs to be added to the builder
_mcp_clients: dict[str, MCPBaseClient] = {}


class MCPServerConfig(BaseModel):
    """
    Server connection details for MCP client.
    Supports stdio, sse, and streamable-http transports.
    """
    client_id: str | None = Field(default=None,
                                  description="Enables setting up multiple MCP clients with the same server")
    transport: Literal["stdio", "sse", "streamable-http"] = Field(
        ..., description="Transport type to connect to the MCP server (stdio, sse, or streamable-http)")
    url: HttpUrl | None = Field(default=None,
                                description="URL of the MCP server (for sse or streamable-http transport)")
    command: str | None = Field(default=None,
                                description="Command to run for stdio transport (e.g. 'python' or 'docker')")
    args: list[str] | None = Field(default=None, description="Arguments for the stdio command")
    env: dict[str, str] | None = Field(default=None, description="Environment variables for the stdio process")

    def model_post_init(self, __context):
        """Validate that stdio and SSE/Streamable HTTP properties are mutually exclusive."""
        super().model_post_init(__context)

        if self.transport == "stdio":
            if self.url is not None:
                raise ValueError("url should not be set when using stdio transport")
            if not self.command:
                raise ValueError("command is required when using stdio transport")
        elif self.transport in ("sse", "streamable-http"):
            if self.command is not None or self.args is not None or self.env is not None:
                raise ValueError("command, args, and env should not be set when using sse or streamable-http transport")
            if not self.url:
                raise ValueError("url is required when using sse or streamable-http transport")


class MCPClientConfig(ClientFunctionConfig, name="mcp_client"):
    """
    Configuration for connecting to an MCP server as a client and exposing selected tools.
    Supports stdio, sse, and streamable-http transports, as well as tool filtering and aliasing.
    """
    server: MCPServerConfig = Field(..., description="Server connection details (transport, url/command, etc.)")
    # TODO: Add tool_filter support
    tool_filter: dict | list | None = Field(default=None,
                                            description="Filter or map tools to expose from the server. \
            Can be a list of tool names or a dict mapping tool names to alias/description.")

    def model_post_init(self, __context):
        super().model_post_init(__context)
        # ServerConfig already validates mutually exclusive fields


class MCPDynamicToolConfig(FunctionBaseConfig, name="mcp_dynamic_tool"):
    """
    Configuration for individual MCP tools that are dynamically discovered and registered.
    """
    server: MCPServerConfig = Field(..., description="Server connection details")
    description: str | None = Field(default=None, description="Description of the tool")
    tool_name: str = Field(..., description="Name of the specific tool to use")


def _get_client_key(server_config: MCPServerConfig) -> str:
    """Create a unique key for the server config to enable client sharing.
    If client_id is set, it will be included in the key.
    """

    if server_config.transport == "stdio":
        args_str = ':'.join(server_config.args or [])
        env_str = ':'.join(f"{k}={v}" for k, v in sorted((server_config.env or {}).items()))
        if server_config.client_id:
            return f"{server_config.client_id}:stdio:{server_config.command}:{args_str}:{env_str}"
        else:
            return f"stdio:{server_config.command}:{args_str}:{env_str}"
    else:
        if server_config.client_id:
            return f"{server_config.client_id}:{server_config.transport}:{server_config.url}"
        else:
            return f"{server_config.transport}:{server_config.url}"


@register_function(config_type=MCPDynamicToolConfig)
async def mcp_dynamic_tool(config: MCPDynamicToolConfig, builder: Builder):
    """
    Dynamic function for an MCP tool that uses shared client connections.
    """
    global _mcp_clients

    client_key = _get_client_key(config.server)

    # If the client is not there throw an error
    if client_key not in _mcp_clients:
        raise ValueError(f"MCP client for key: {client_key} not found")
    client = _mcp_clients[client_key]

    # This is for a single tool, so we don't need to get all tools
    tool = await client.get_tool(config.tool_name)

    # Create function info for the specific tool
    input_schema = tool.input_schema

    def _convert_from_str(input_str: str) -> input_schema:
        return input_schema.model_validate_json(input_str)

    async def _response_fn(tool_input: input_schema | None = None, **kwargs) -> str:
        try:
            if tool_input:
                args = tool_input.model_dump()
                return await tool.acall(args)

            _ = input_schema.model_validate(kwargs)
            return await tool.acall(kwargs)
        except Exception as e:
            return str(e)

    fn = FunctionInfo.create(single_fn=_response_fn,
                             description=tool.description,
                             input_schema=input_schema,
                             converters=[_convert_from_str])
    yield fn


@register_client_function_handler(MCPClientConfig)
async def mcp_client_function_handler(config: MCPClientConfig, builder: Builder):  # pylint: disable=unused-argument
    """
    Connects to an MCP server, discovers all tools, and adds them as functions to the builder.
    TODO: Add tool_filter support and tool name/description override support
    """
    from aiq.tool.mcp.mcp_client_base import MCPSSEClient
    from aiq.tool.mcp.mcp_client_base import MCPStdioClient
    from aiq.tool.mcp.mcp_client_base import MCPStreamableHTTPClient
    global _mcp_clients

    # 1. Check if the client already exists
    client_key = _get_client_key(config.server)
    # If the client is already there, throw an error
    if client_key in _mcp_clients:
        raise ValueError(f"MCP client for key: {client_key} already exists")

    # 2. Instantiate the client
    if config.server.transport == "stdio":
        source = f"{config.server.command} {' '.join(config.server.args) if config.server.args else ''}"
        client = MCPStdioClient(command=config.server.command, args=config.server.args, env=config.server.env)
    elif config.server.transport == "sse":
        source = str(config.server.url)
        client = MCPSSEClient(url=source)
    elif config.server.transport == "streamable-http":
        client = MCPStreamableHTTPClient(url=str(config.server.url))
    else:
        raise ValueError("Unsupported transport")

    # 3. Store the client in the global cache, this is used by the dynamic tool function
    _mcp_clients[client_key] = client

    # 4. Connect to the server and find all tools
    # Store the client in the builder's exit stack to ensure it's cleaned up when the builder is done
    await builder._get_exit_stack().enter_async_context(client)
    # Find all tools
    all_tools = await client.get_tools()

    # 5. Add all tools to the builder
    for tool in all_tools.values():
        # Add a tool of type MCPDynamicToolConfig, don't worry about the tool description or input schema
        # as they are handled by the dynamic tool function
        await builder.add_function(tool.name, MCPDynamicToolConfig(server=config.server, tool_name=tool.name))


"""
TODO:
- Add tool_filter support
- Add tool name/description override support
- Move the client cache to the builder
- Add a way to get all dynamic tools via a special workflow keyword
- Allow the react agent to use all dynamic tools
- Add ClientFunctionConfig to the registry by making mcp_client_function_handler yield an idle function
"""
