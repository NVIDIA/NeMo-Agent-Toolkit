# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""Configuration for A365 tooling integration with NAT MCP client."""

from datetime import timedelta

from pydantic import Field, model_validator

from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.function import FunctionGroupBaseConfig
from nat.plugins.a365.exceptions import A365ConfigurationError


class A365MCPToolingConfig(FunctionGroupBaseConfig, name="a365_mcp_tooling"):
    """Configuration for discovering and registering MCP servers from Agent 365.

    This configuration uses the A365 tooling service to discover MCP servers
    configured for the agent and registers them as NAT function groups.

    **Prerequisites:**
    - The `nvidia-nat-mcp` package must be installed.
    - Install with: ``uv pip install nvidia-nat-mcp`` or ``uv pip install 'nvidia-nat[mcp]'``
    - If installing from source: ``uv pip install 'nvidia-nat-a365[mcp]'``

    Example:
        ```yaml
        function_groups:
          - type: a365_mcp_tooling
            agentic_app_id: "your-agent-id"
            auth_token: "your-auth-token"
        ```
    """

    agentic_app_id: str = Field(..., description="Agent 365 agentic app ID")
    auth_token: str | AuthenticationRef = Field(
        ..., description="Authentication token or reference to auth provider for A365 tooling gateway"
    )
    tool_call_timeout: timedelta = Field(
        default=timedelta(seconds=60),
        description="Timeout for MCP tool calls. Defaults to 60 seconds.",
    )
    auth_flow_timeout: timedelta = Field(
        default=timedelta(seconds=300),
        description="Timeout for MCP auth flows. Defaults to 300 seconds.",
    )
    reconnect_enabled: bool = Field(
        default=True,
        description="Whether to enable reconnecting to MCP servers if connection is lost. Defaults to True.",
    )
    reconnect_max_attempts: int = Field(
        default=2,
        ge=0,
        description="Maximum number of reconnect attempts. Defaults to 2.",
    )
    reconnect_initial_backoff: float = Field(
        default=0.5,
        ge=0.0,
        description="Initial backoff time for reconnect attempts in seconds. Defaults to 0.5 seconds.",
    )
    reconnect_max_backoff: float = Field(
        default=50.0,
        ge=0.0,
        description="Maximum backoff time for reconnect attempts in seconds. Defaults to 50.0 seconds.",
    )
    session_aware_tools: bool = Field(
        default=True,
        description="Create session-aware tools if True. Defaults to True.",
    )
    max_sessions: int = Field(
        default=100,
        ge=1,
        description="Maximum number of concurrent session clients. Defaults to 100.",
    )
    session_idle_timeout: timedelta = Field(
        default=timedelta(hours=1),
        description="Time after which inactive sessions are cleaned up. Defaults to 1 hour.",
    )
    tool_overrides: dict[str, dict[str, str]] | None = Field(
        default=None,
        description="""Optional tool name overrides and description changes applied to all discovered servers.
        Example:
          tool_overrides:
            calculator_add:
              alias: "add_numbers"
              description: "Add two numbers together"
        """,
    )
    server_auth_providers: dict[str, str | AuthenticationRef] | None = Field(
        default=None,
        description="""Optional per-server authentication provider overrides.
        Maps MCP server names (from A365 discovery) to authentication provider references.
        If not specified, discovered servers will use the same auth provider as the A365 gateway
        (when auth_token is an AuthenticationRef). If auth_token is a string, servers will not have auth.
        
        Example:
          server_auth_providers:
            "my-custom-server": "custom_auth_provider"
            "another-server": "another_auth_provider"
        """,
    )

    @model_validator(mode="after")
    def _validate_reconnect_backoff(self) -> "A365MCPToolingConfig":
        """Validate reconnect backoff values."""
        if self.reconnect_max_backoff < self.reconnect_initial_backoff:
            raise A365ConfigurationError(
                "reconnect_max_backoff must be greater than or equal to reconnect_initial_backoff"
            )
        return self
