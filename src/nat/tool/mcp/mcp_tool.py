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
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class MCPToolConfig(FunctionBaseConfig, name="mcp_tool_wrapper"):
    """
    Function which connects to a Model Context Protocol (MCP) server and wraps the selected tool as a NeMo Agent toolkit
    function.
    """
    # Add your custom configuration parameters here
    url: HttpUrl | None = Field(default=None,
                                description="The URL of the MCP server (for streamable-http or sse modes)")
    mcp_tool_name: str = Field(description="The name of the tool served by the MCP Server that you want to use")
    transport: Literal["sse", "stdio", "streamable-http"] = Field(
        default="streamable-http",
        description="The type of transport to use (default: streamable-http, backwards compatible with sse)")
    command: str | None = Field(default=None,
                                description="The command to run for stdio mode (e.g. 'docker' or 'python')")
    args: list[str] | None = Field(default=None, description="Additional arguments for the stdio command")
    env: dict[str, str] | None = Field(default=None, description="Environment variables to set for the stdio process")
    description: str | None = Field(default=None,
                                    description="""
        Description for the tool that will override the description provided by the MCP server. Should only be used if
        the description provided by the server is poor or nonexistent
        """)
    return_exception: bool = Field(default=True,
                                   description="""
        If true, the tool will return the exception message if the tool call fails.
        If false, raise the exception.
        """)

    @model_validator(mode="after")
    def validate_model(self):
        """Validate that stdio and SSE/Streamable HTTP properties are mutually exclusive."""
        if self.transport == 'stdio':
            if self.url is not None:
                raise ValueError("url should not be set when using stdio client type")
            if not self.command:
                raise ValueError("command is required when using stdio client type")
        elif self.transport in ['streamable-http', 'sse']:
            if self.command is not None or self.args is not None or self.env is not None:
                raise ValueError(
                    "command, args, and env should not be set when using streamable-http or sse client type")
            if not self.url:
                raise ValueError("url is required when using streamable-http or sse client type")
        return self


@register_function(config_type=MCPToolConfig)
async def mcp_tool(config: MCPToolConfig, builder: Builder):
    """
    Generate a NeMo Agent Toolkit Function that wraps a tool provided by the MCP server.
    """

    from nat.tool.mcp.mcp_client_base import MCPSSEClient
    from nat.tool.mcp.mcp_client_base import MCPStdioClient
    from nat.tool.mcp.mcp_client_base import MCPStreamableHTTPClient
    from nat.tool.mcp.mcp_client_base import MCPToolClient

    # Initialize the client
    if config.transport == 'stdio':
        client = MCPStdioClient(command=config.command, args=config.args, env=config.env)
    elif config.transport == 'streamable-http':
        client = MCPStreamableHTTPClient(url=str(config.url))
    elif config.transport == 'sse':
        client = MCPSSEClient(url=str(config.url))
    else:
        raise ValueError(f"Invalid transport type: {config.transport}")

    async with client:
        # If the tool is found create a MCPToolClient object and set the description if provided
        tool: MCPToolClient = await client.get_tool(config.mcp_tool_name)
        if config.description:
            tool.set_description(description=config.description)

        logger.info("Configured to use tool: %s from MCP server at %s", tool.name, client.server_name)

        def _convert_from_str(input_str: str) -> tool.input_schema:
            return tool.input_schema.model_validate_json(input_str)

        async def _response_fn(tool_input: BaseModel | None = None, **kwargs) -> str:
            # Run the tool, catching any errors and sending to agent for correction
            try:
                if tool_input:
                    args = tool_input.model_dump()
                    return await tool.acall(args)

                _ = tool.input_schema.model_validate(kwargs)
                return await tool.acall(kwargs)
            except Exception as e:
                if config.return_exception:
                    if tool_input:
                        logger.warning("Error calling tool %s with serialized input: %s",
                                       tool.name,
                                       tool_input.model_dump(),
                                       exc_info=True)
                    else:
                        logger.warning("Error calling tool %s with input: %s", tool.name, kwargs, exc_info=True)
                    return str(e)
                # If the tool call fails, raise the exception.
                raise

        yield FunctionInfo.create(single_fn=_response_fn,
                                  description=tool.description,
                                  input_schema=tool.input_schema,
                                  converters=[_convert_from_str])
