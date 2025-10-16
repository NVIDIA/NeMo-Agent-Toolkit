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

from datetime import timedelta
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl
from pydantic import model_validator

from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.function import FunctionGroupBaseConfig


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
    auth_flow_timeout: timedelta = Field(
        default=timedelta(seconds=300),
        description="Timeout (in seconds) for the MCP auth flow. When the tool call requires interactive \
        authentication, this timeout is used. Defaults to 300 seconds.")
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
    session_aware_tools: bool = Field(default=True,
                                      description="Session-aware tools are created if True. Defaults to True.")
    max_sessions: int = Field(default=100,
                              ge=1,
                              description="Maximum number of concurrent session clients. Defaults to 100.")
    session_idle_timeout: timedelta = Field(
        default=timedelta(hours=1),
        description="Time after which inactive sessions are cleaned up. Defaults to 1 hour.")

    @model_validator(mode="after")
    def _validate_reconnect_backoff(self) -> "MCPClientConfig":
        """Validate reconnect backoff values."""
        if self.reconnect_max_backoff < self.reconnect_initial_backoff:
            raise ValueError("reconnect_max_backoff must be greater than or equal to reconnect_initial_backoff")
        return self
