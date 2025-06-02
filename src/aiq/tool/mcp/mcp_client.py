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
from aiq.data_models.function import FunctionBaseConfig

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


class MCPClientConfig(FunctionBaseConfig, name="mcp_client"):
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


@register_function(config_type=MCPClientConfig)
async def mcp_client(config: MCPClientConfig, builder: Builder):  # pylint: disable=unused-argument
    """
    Connects to an MCP server, discovers all tools, and yields a FunctionInfo for each tool.
    Skips tool_filter handling for now.
    """
    from aiq.tool.mcp.mcp_client_base import MCPSSEClient
    from aiq.tool.mcp.mcp_client_base import MCPStdioClient
    from aiq.tool.mcp.mcp_client_base import MCPStreamableHTTPClient

    # 1. Select and instantiate the client
    if config.server.transport == "stdio":
        client = MCPStdioClient(command=config.server.command, args=config.server.args, env=config.server.env)
    elif config.server.transport == "sse":
        client = MCPSSEClient(url=str(config.server.url))
    elif config.server.transport == "streamable-http":
        client = MCPStreamableHTTPClient(url=str(config.server.url))
    else:
        raise ValueError("Unsupported transport")

    # 2. Connect and discover tools
    async with client:
        all_tools = await client.get_tools()  # Dict[str, MCPToolClient]
        for tool in all_tools.values():
            input_schema = tool.input_schema  # Pydantic model

            def _convert_from_str(input_str: str) -> input_schema:
                return input_schema.model_validate_json(input_str)

            async def _response_fn(tool_input: input_schema | None = None, **kwargs):
                try:
                    if tool_input:
                        args = tool_input.model_dump()
                        return await tool.acall(args)
                    _ = input_schema.model_validate(kwargs)
                    return await tool.acall(kwargs)
                except Exception as e:
                    # Optionally handle/return exception
                    return str(e)

            fn = FunctionInfo.create(single_fn=_response_fn,
                                     description=tool.description,
                                     input_schema=input_schema,
                                     converters=[_convert_from_str])
            yield fn
