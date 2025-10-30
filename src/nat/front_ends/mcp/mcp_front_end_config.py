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

from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from nat.data_models.front_end import FrontEndBaseConfig

logger = logging.getLogger(__name__)


class MCPFrontEndConfig(FrontEndBaseConfig, name="mcp"):
    """MCP front end configuration.

    A simple MCP (Model Context Protocol) front end for NeMo Agent toolkit.
    """

    name: str = Field(default="NeMo Agent Toolkit MCP",
                      description="Name of the MCP server (default: NeMo Agent Toolkit MCP)")
    host: str = Field(default="localhost", description="Host to bind the server to (default: localhost)")
    port: int = Field(default=9901, description="Port to bind the server to (default: 9901)", ge=0, le=65535)
    debug: bool = Field(default=False, description="Enable debug mode (default: False)")
    log_level: str = Field(default="INFO", description="Log level for the MCP server (default: INFO)")
    tool_names: list[str] = Field(
        default_factory=list,
        description="The list of tools MCP server will expose (default: all tools)."
        "Tool names can be functions or function groups",
    )
    transport: Literal["sse", "streamable-http"] = Field(
        default="streamable-http",
        description="Transport type for the MCP server (default: streamable-http, backwards compatible with sse)")
    runner_class: str | None = Field(
        default=None, description="Custom worker class for handling MCP routes (default: built-in worker)")
    base_path: str | None = Field(default=None,
                                  description="Base path to mount the MCP server at (e.g., '/api/v1'). "
                                  "If specified, the server will be accessible at http://host:port{base_path}/mcp. "
                                  "If None, server runs at root path /mcp.")

    server_auth: OAuth2ResourceServerConfig | None = Field(
        default=None, description=("OAuth 2.0 Resource Server configuration for token verification."))

    @field_validator('base_path')
    @classmethod
    def validate_base_path(cls, v: str | None) -> str | None:
        """Validate that base_path starts with '/' and doesn't end with '/'."""
        if v is not None:
            if not v.startswith('/'):
                raise ValueError("base_path must start with '/'")
            if v.endswith('/'):
                raise ValueError("base_path must not end with '/'")
        return v

    # Memory profiling configuration
    enable_memory_profiling: bool = Field(default=False,
                                          description="Enable memory profiling and diagnostics (default: False)")
    memory_profile_interval: int = Field(default=50,
                                         description="Log memory stats every N requests (default: 50)",
                                         ge=1)
    memory_profile_top_n: int = Field(default=10,
                                      description="Number of top memory allocations to log (default: 10)",
                                      ge=1,
                                      le=50)
    memory_profile_log_level: str = Field(default="DEBUG",
                                          description="Log level for memory profiling output (default: DEBUG)")

    @model_validator(mode="after")
    def validate_security_configuration(self):
        """Validate security configuration to prevent accidental misconfigurations."""
        # Check if server is bound to a non-localhost interface without authentication
        localhost_hosts = {"localhost", "127.0.0.1", "::1"}
        if self.host not in localhost_hosts and self.server_auth is None:
            logger.warning(
                "MCP server is configured to bind to '%s' without authentication. "
                "This may expose your server to unauthorized access. "
                "Consider either: (1) binding to localhost for local-only access, "
                "or (2) configuring server_auth for production deployments on public interfaces.",
                self.host)

        # Check if SSE transport is used (which doesn't support authentication)
        if self.transport == "sse":
            if self.server_auth is not None:
                logger.warning("SSE transport does not support authentication. "
                               "The configured server_auth will be ignored. "
                               "For production use with authentication, use 'streamable-http' transport instead.")
            elif self.host not in localhost_hosts:
                logger.warning(
                    "SSE transport does not support authentication and is bound to '%s'. "
                    "This configuration is not recommended for production use. "
                    "For production deployments, use 'streamable-http' transport with server_auth configured.",
                    self.host)

        return self
