# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""FastMCP front end configuration."""

import logging
from typing import Literal

from pydantic import Field
from pydantic import field_validator

from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from nat.data_models.front_end import FrontEndBaseConfig

logger = logging.getLogger(__name__)


class FastMCPFrontEndConfig(FrontEndBaseConfig, name="fastmcp"):
    """FastMCP front end configuration.

    A FastMCP front end for NeMo Agent Toolkit workflows.
    """

    name: str = Field(default="NeMo Agent Toolkit FastMCP",
                      description="Name of the FastMCP server (default: NeMo Agent Toolkit FastMCP)")
    host: str = Field(default="localhost", description="Host to bind the server to (default: localhost)")
    port: int = Field(default=9902, description="Port to bind the server to (default: 9902)", ge=0, le=65535)
    debug: bool = Field(default=False, description="Enable debug mode (default: False)")
    log_level: str = Field(default="INFO", description="Log level for the FastMCP server (default: INFO)")
    tool_names: list[str] = Field(
        default_factory=list,
        description="The list of tools FastMCP server will expose (default: all tools). "
        "Tool names can be functions or function groups",
    )
    transport: Literal["sse", "streamable-http"] = Field(
        default="streamable-http",
        description="Transport type for the FastMCP server (default: streamable-http, backwards compatible with sse)")
    runner_class: str | None = Field(
        default=None, description="Custom worker class for handling FastMCP routes (default: built-in worker)")
    base_path: str | None = Field(default=None,
                                  description="Base path to mount the FastMCP server at (e.g., '/api/v1'). "
                                  "If specified, the server will be accessible at http://host:port{base_path}/mcp. "
                                  "If None, server runs at root path /mcp.")

    server_auth: OAuth2ResourceServerConfig | None = Field(
        default=None, description=("OAuth 2.0 Resource Server configuration for token verification."))

    @field_validator('base_path')
    @classmethod
    def validate_base_path(cls, v: str | None) -> str | None:
        """Validate that `base_path` starts with '/' and does not end with '/'."""
        if v is not None:
            if not v.startswith('/'):
                raise ValueError("base_path must start with '/'")
            if v.endswith('/'):
                raise ValueError("base_path must not end with '/'")
        return v
