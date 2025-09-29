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
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.function import FunctionGroupBaseConfig
from nat.plugins.mcp.tool import mcp_tool_function

logger = logging.getLogger(__name__)


class MCPFunctionGroup(FunctionGroup):
    """
    A specialized FunctionGroup for MCP clients that includes MCP-specific attributes
    with proper type safety.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MCP client attributes with proper typing
        self._mcp_client = None  # Will be set to the actual MCP client instance
        self._mcp_client_server_name: str | None = None
        self._mcp_client_transport: str | None = None

    @property
    def mcp_client(self):
        """Get the MCP client instance."""
        return self._mcp_client

    @mcp_client.setter
    def mcp_client(self, client):
        """Set the MCP client instance."""
        self._mcp_client = client

    @property
    def mcp_client_server_name(self) -> str | None:
        """Get the MCP client server name."""
        return self._mcp_client_server_name

    @mcp_client_server_name.setter
    def mcp_client_server_name(self, server_name: str | None):
        """Set the MCP client server name."""
        self._mcp_client_server_name = server_name

    @property
    def mcp_client_transport(self) -> str | None:
        """Get the MCP client transport type."""
        return self._mcp_client_transport

    @mcp_client_transport.setter
    def mcp_client_transport(self, transport: str | None):
        """Set the MCP client transport type."""
        self._mcp_client_transport = transport


class MCPToolOverrideConfig(BaseModel):
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

    # Authentication configuration
    auth_provider: str | AuthenticationRef | None = Field(default=None,
                                                          description="Reference to authentication provider")

    @model_validator(mode="after")
    def validate_model(self):
        """Validate that stdio and SSE/Streamable HTTP properties are mutually exclusive."""
        if self.transport == "stdio":
            if self.url is not None:
                raise ValueError("url should not be set when using stdio transport")
            if not self.command:
                raise ValueError("command is required when using stdio transport")
            # Auth is not supported for stdio transport
            if self.auth_provider is not None:
                raise ValueError("Authentication is not supported for stdio transport")
        elif self.transport == "sse":
            if self.command is not None or self.args is not None or self.env is not None:
                raise ValueError("command, args, and env should not be set when using sse transport")
            if not self.url:
                raise ValueError("url is required when using sse transport")
            # Auth is not supported for SSE transport
            if self.auth_provider is not None:
                raise ValueError("Authentication is not supported for SSE transport.")
        elif self.transport == "streamable-http":
            if self.command is not None or self.args is not None or self.env is not None:
                raise ValueError("command, args, and env should not be set when using streamable-http transport")
            if not self.url:
                raise ValueError("url is required when using streamable-http transport")

        return self


class MCPClientConfig(FunctionGroupBaseConfig, name="mcp_client"):
    """
    Configuration for connecting to an MCP server as a client and exposing selected tools.
    """
    server: MCPServerConfig = Field(..., description="Server connection details (transport, url/command, etc.)")
    tool_call_timeout: timedelta = Field(
        default=timedelta(seconds=60),
        description="Timeout (in seconds) for the MCP tool call. Defaults to 60 seconds.")
    reconnect_enabled: bool = Field(
        default=True,
        description="Whether to enable reconnecting to the MCP server if the connection is lost. \
        Defaults to True.")
    reconnect_max_attempts: int = Field(default=2,
                                        ge=0,
                                        description="Maximum number of reconnect attempts. Defaults to 2.")
    reconnect_initial_backoff: float = Field(
        default=0.5, ge=0.0, description="Initial backoff time for reconnect attempts. Defaults to 0.5 seconds.")
    reconnect_max_backoff: float = Field(
        default=50.0, ge=0.0, description="Maximum backoff time for reconnect attempts. Defaults to 50 seconds.")
    tool_overrides: dict[str, MCPToolOverrideConfig] | None = Field(
        default=None,
        description="""Optional tool name overrides and description changes.
        Example:
          tool_overrides:
            calculator_add:
              alias: "add_numbers"
              description: "Add two numbers together"
            calculator_multiply:
              description: "Multiply two numbers"  # alias defaults to original name
        """)

    @model_validator(mode="after")
    def _validate_reconnect_backoff(self) -> "MCPClientConfig":
        """Validate reconnect backoff values."""
        if self.reconnect_max_backoff < self.reconnect_initial_backoff:
            raise ValueError("reconnect_max_backoff must be greater than or equal to reconnect_initial_backoff")
        return self


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
                                config.tool_call_timeout,
                                reconnect_enabled=config.reconnect_enabled,
                                reconnect_max_attempts=config.reconnect_max_attempts,
                                reconnect_initial_backoff=config.reconnect_initial_backoff,
                                reconnect_max_backoff=config.reconnect_max_backoff)
    elif config.server.transport == "sse":
        client = MCPSSEClient(str(config.server.url),
                              tool_call_timeout=config.tool_call_timeout,
                              reconnect_enabled=config.reconnect_enabled,
                              reconnect_max_attempts=config.reconnect_max_attempts,
                              reconnect_initial_backoff=config.reconnect_initial_backoff,
                              reconnect_max_backoff=config.reconnect_max_backoff)
    elif config.server.transport == "streamable-http":
        client = MCPStreamableHTTPClient(str(config.server.url),
                                         auth_provider=auth_provider,
                                         tool_call_timeout=config.tool_call_timeout,
                                         reconnect_enabled=config.reconnect_enabled,
                                         reconnect_max_attempts=config.reconnect_max_attempts,
                                         reconnect_initial_backoff=config.reconnect_initial_backoff,
                                         reconnect_max_backoff=config.reconnect_max_backoff)
    else:
        raise ValueError(f"Unsupported transport: {config.server.transport}")

    logger.info("Configured to use MCP server at %s", client.server_name)

    # Create the MCP function group
    group = MCPFunctionGroup(config=config)

    async with client:
        # Expose the live MCP client on the function group instance so other components (e.g., HTTP endpoints)
        # can reuse the already-established session instead of creating a new client per request.
        group.mcp_client = client
        group.mcp_client_server_name = client.server_name
        group.mcp_client_transport = client.transport

        all_tools = await client.get_tools()
        tool_overrides = mcp_apply_tool_alias_and_description(all_tools, config.tool_overrides)

        # Add each tool as a function to the group
        for tool_name, tool in all_tools.items():
            # Get override if it exists
            override = tool_overrides.get(tool_name)

            # Use override values or defaults
            function_name = override.alias if override and override.alias else tool_name
            description = override.description if override and override.description else tool.description

            # Create the tool function
            tool_fn = mcp_tool_function(tool)

            # Normalize optional typing for linter/type-checker compatibility
            single_fn = tool_fn.single_fn
            if single_fn is None:
                # Should not happen because mcp_tool_function always sets a single_fn
                logger.warning("Skipping tool %s because single_fn is None", function_name)
                continue

            input_schema = tool_fn.input_schema
            # Convert NoneType sentinel to None for FunctionGroup.add_function signature
            if input_schema is type(None):  # noqa: E721
                input_schema = None

            # Add to group
            logger.info("Adding tool %s to group", function_name)
            group.add_function(name=function_name,
                               description=description,
                               fn=single_fn,
                               input_schema=input_schema,
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
