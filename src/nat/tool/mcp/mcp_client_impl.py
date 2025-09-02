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
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.experimental.decorators.experimental_warning_decorator import experimental
from nat.tool.mcp.mcp_client_base import MCPBaseClient

logger = logging.getLogger(__name__)

# All functions in this file are experimental


class ToolOverrideConfig(BaseModel):
    """
    Configuration for overriding tool properties when exposing from MCP server.
    """
    alias: str | None = Field(default=None, description="Override the tool name (function name in the workflow)")
    description: str | None = Field(default=None, description="Override the tool description")


class MCPServerConfig(BaseModel):
    """
    Server connection details for MCP client.
    Supports stdio, sse, and streamable-http transports.
    streamable-http is the recommended default for HTTP-based connections.
    """
    transport: Literal["stdio", "sse", "streamable-http"] = Field(
        ..., description="Transport type to connect to the MCP server (stdio, sse, or streamable-http)")
    url: HttpUrl | None = Field(default=None,
                                description="URL of the MCP server (for sse or streamable-http transport)")
    command: str | None = Field(default=None,
                                description="Command to run for stdio transport (e.g. 'python' or 'docker')")
    args: list[str] | None = Field(default=None, description="Arguments for the stdio command")
    env: dict[str, str] | None = Field(default=None, description="Environment variables for the stdio process")

    @model_validator(mode="after")
    def validate_model(self):
        """Validate that stdio and SSE/Streamable HTTP properties are mutually exclusive."""
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
        return self


class MCPClientConfig(FunctionBaseConfig, name="mcp_client"):
    """
    Configuration for connecting to an MCP server as a client and exposing selected tools.
    """
    server: MCPServerConfig = Field(..., description="Server connection details (transport, url/command, etc.)")
    tool_filter: dict[str, ToolOverrideConfig] | list[str] | None = Field(
        default=None,
        description="""Filter or map tools to expose from the server (list or dict).
        Can be:
        - A list of tool names to expose: ['tool1', 'tool2']
        - A dict mapping tool names to override configs:
          {'tool1': {'alias': 'new_name', 'description': 'New desc'}}
          {'tool2': {'description': 'Override description only'}}  # alias defaults to 'tool2'
        """)


class MCPSingleToolConfig(FunctionBaseConfig, name="mcp_single_tool"):
    """
    Configuration for wrapping a single tool from an MCP server as a NeMo Agent toolkit function.
    """
    client: MCPBaseClient = Field(..., description="MCP client to use for the tool")
    tool_name: str = Field(..., description="Name of the tool to use")
    tool_description: str | None = Field(default=None, description="Description of the tool")

    model_config = {"arbitrary_types_allowed": True}


def _get_server_name_safe(client: MCPBaseClient) -> str:

    # Avoid leaking env secrets from stdio client in logs.
    if client.transport == "stdio":
        safe_server = f"stdio: {client.command}"
    else:
        safe_server = f"{client.transport}: {client.url}"

    return safe_server


@register_function(config_type=MCPSingleToolConfig)
async def mcp_single_tool(config: MCPSingleToolConfig, builder: Builder):
    """
    Wrap a single tool from an MCP server as a NeMo Agent toolkit function.
    """
    tool = await config.client.get_tool(config.tool_name)
    if config.tool_description:
        tool.set_description(description=config.tool_description)
    input_schema = tool.input_schema

    logger.info("Configured to use tool: %s from MCP server at %s", tool.name, _get_server_name_safe(config.client))

    def _convert_from_str(input_str: str) -> BaseModel:
        return input_schema.model_validate_json(input_str)

    @experimental(feature_name="mcp_client")
    async def _response_fn(tool_input: BaseModel | None = None, **kwargs) -> str:
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


@register_function(MCPClientConfig)
async def mcp_client_function_handler(config: MCPClientConfig, builder: Builder):
    """
    Connect to an MCP server, discover tools, and register them as functions in the workflow.

    Note:
    - Uses builder's exit stack to manage client lifecycle
    - Applies tool filters if provided
    """
    from nat.tool.mcp.mcp_client_base import MCPSSEClient
    from nat.tool.mcp.mcp_client_base import MCPStdioClient
    from nat.tool.mcp.mcp_client_base import MCPStreamableHTTPClient

    # Build the appropriate client
    client_cls = {
        "stdio": lambda: MCPStdioClient(config.server.command, config.server.args, config.server.env),
        "sse": lambda: MCPSSEClient(str(config.server.url)),
        "streamable-http": lambda: MCPStreamableHTTPClient(str(config.server.url)),
    }.get(config.server.transport)

    if not client_cls:
        raise ValueError(f"Unsupported transport: {config.server.transport}")

    client = client_cls()
    logger.info("Configured to use MCP server at %s", _get_server_name_safe(client))

    # client aenter connects to the server and stores the client in the exit stack
    # so it's cleaned up when the workflow is done
    async with client:
        all_tools = await client.get_tools()
        tool_configs = _filter_and_configure_tools(all_tools, config.tool_filter)

        for tool_name, tool_cfg in tool_configs.items():
            await builder.add_function(
                tool_cfg["function_name"],
                MCPSingleToolConfig(
                    client=client,
                    tool_name=tool_name,
                    tool_description=tool_cfg["description"],
                ))

        @experimental(feature_name="mcp_client")
        async def idle_fn(text: str) -> str:
            # This function is a placeholder and will be removed when function groups are used
            return f"MCP client connected: {text}"

        yield FunctionInfo.create(single_fn=idle_fn, description="MCP client")


def _filter_and_configure_tools(all_tools: dict, tool_filter) -> dict[str, dict]:
    """
    Apply tool filtering and optional aliasing/description overrides.

    Returns:
        Dict[str, dict] where each value has:
            - function_name
            - description
    """
    if tool_filter is None:
        return {name: {"function_name": name, "description": tool.description} for name, tool in all_tools.items()}

    if isinstance(tool_filter, list):
        return {
            name: {
                "function_name": name, "description": all_tools[name].description
            }
            for name in tool_filter if name in all_tools
        }

    if isinstance(tool_filter, dict):
        result = {}
        for name, override in tool_filter.items():
            tool = all_tools.get(name)
            if not tool:
                logger.warning("Tool '%s' specified in tool_filter not found in MCP server", name)
                continue

            if isinstance(override, ToolOverrideConfig):
                result[name] = {
                    "function_name": override.alias or name, "description": override.description or tool.description
                }
            else:
                logger.warning("Unsupported override type for '%s': %s", name, type(override))
                result[name] = {"function_name": name, "description": tool.description}
        return result
