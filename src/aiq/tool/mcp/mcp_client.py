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


class MCPServerConfig(BaseModel):
    """
    Server connection details for MCP client.
    Supports stdio, sse, and streamable-http transports.
    """
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


class MCPSingleToolConfig(FunctionBaseConfig, name="mcp_single_tool"):
    client: MCPBaseClient = Field(..., description="MCP client to use for the tool")
    tool_name: str = Field(..., description="Name of the tool to use")
    tool_description: str | None = Field(default=None, description="Description of the tool")

    model_config = {"arbitrary_types_allowed": True}


@register_function(config_type=MCPSingleToolConfig)
async def mcp_single_tool(config: MCPSingleToolConfig, builder: Builder):
    tool = await config.client.get_tool(config.tool_name)
    if config.tool_description:
        tool.set_description(description=config.tool_description)
    input_schema = tool.input_schema

    logger.info("Configured to use tool: %s from MCP server at %s", tool.name, config.client.server_name)

    def _convert_from_str(input_str: str) -> BaseModel:
        return input_schema.model_validate_json(input_str)

    async def _response_fn(tool_input: input_schema | None = None, **kwargs) -> str:
        try:
            if tool_input:
                return await tool.acall(tool_input.model_dump())
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

    builder: This is the main workflow builder (not the child builder). This is needed because -
    1. we need access to the builder's lifecycle to add the client to the exit stack
    2. we dynamically add tools to the builder

    TODO: Add tool_filter support and tool name/description override support
    """
    from aiq.tool.mcp.mcp_client_base import MCPSSEClient
    from aiq.tool.mcp.mcp_client_base import MCPStdioClient
    from aiq.tool.mcp.mcp_client_base import MCPStreamableHTTPClient

    # 1. Instantiate the client
    if config.server.transport == "stdio":
        client = MCPStdioClient(command=config.server.command, args=config.server.args, env=config.server.env)
    elif config.server.transport == "sse":
        client = MCPSSEClient(url=str(config.server.url))
    elif config.server.transport == "streamable-http":
        client = MCPStreamableHTTPClient(url=str(config.server.url))
    else:
        raise ValueError("Unsupported transport")

    logger.info("Configured to use MCP server at %s", client.server_name)

    # 2. Connect to the server and find all tools
    # Store the client in the builder's exit stack to ensure it's cleaned up when the builder is done
    await builder._get_exit_stack().enter_async_context(client)

    # Find all tools
    all_tools = await client.get_tools()

    # 3. Add all tools to the builder dynamically
    for tool in all_tools.values():
        await builder.add_function(tool.name,
                                   MCPSingleToolConfig(
                                       client=client,
                                       tool_name=tool.name,
                                       tool_description=None,
                                   ))


"""
TODO:
- Add tool_filter support
- Add ClientFunctionConfig to the registry by making mcp_client_function_handler yield an idle function
- Add a way to get all dynamic tools via a special workflow keyword
"""
