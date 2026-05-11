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
from typing import Literal

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.function import FunctionGroupBaseConfig
from nat.plugins.a365.exceptions import A365ConfigurationError

# Policy for how to handle per-server registration failures during MCP discovery.
# See ``on_server_registration_error`` on ``A365MCPToolingConfig`` for semantics.
ServerRegistrationErrorPolicy = Literal["fail_fast", "skip_with_warning", "skip_silently"]


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
        ..., description="Authentication token or reference to auth provider for A365 tooling gateway")
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
        Lookups are case-insensitive against the server names returned by the A365 gateway.
        If not specified, discovered servers will use the same auth provider as the A365 gateway
        (when auth_token is an AuthenticationRef). If auth_token is a string, servers will not have auth.

        Override keys that don't match any discovered server are logged as warnings so dead
        configuration is surfaced.

        Example:
          server_auth_providers:
            "my-custom-server": "custom_auth_provider"
            "another-server": "another_auth_provider"
        """,
    )
    on_server_registration_error: ServerRegistrationErrorPolicy = Field(
        default="skip_with_warning",
        description="""Policy for handling per-MCP-server registration failures.

        - ``fail_fast``: abort registration with ``A365SDKError`` on the first server failure.
          Recommended for production where a missing tool should fail loudly rather than
          allow the agent to run with a silently-degraded toolset.
        - ``skip_with_warning`` (default): log a WARN per skipped server and continue. The
          composite function group records the skipped server names; check via
          ``A365MCPToolingFunctionGroup.skipped_servers``. Matches today's tolerant behavior
          but escalates from a lossy ``logger.error`` line to an explicit policy.
        - ``skip_silently``: log at DEBUG only. Reserved for dev / local manifest setups
          where intermittent registration failures are expected.

        Discovery failures from the A365 gateway itself (no servers returned, gateway 4xx)
        always raise; this policy only governs per-server registration after discovery
        succeeds.
        """,
    )

    @model_validator(mode="after")
    def _validate_reconnect_backoff(self) -> "A365MCPToolingConfig":
        """Validate reconnect backoff values."""
        if self.reconnect_max_backoff < self.reconnect_initial_backoff:
            raise A365ConfigurationError(
                "reconnect_max_backoff must be greater than or equal to reconnect_initial_backoff")
        return self
