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

"""Registration for A365 tooling integration with NAT MCP client."""

import logging
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Union

from pydantic import BaseModel, Field, model_validator

from nat.builder.function import FunctionGroup, FunctionGroupBaseConfig
from nat.cli.register_workflow import register_function_group
from nat.data_models.component_ref import AuthenticationRef

logger = logging.getLogger(__name__)


class A365MCPToolingFunctionGroup(FunctionGroup):
    """
    Composite FunctionGroup that aggregates functions from multiple MCP servers.
    
    Instead of merging functions into a single group, this class delegates to
    multiple MCP FunctionGroups and aggregates their results. This preserves
    the original function bindings and avoids double-wrapping.
    """
    
    def __init__(self, config: "A365MCPToolingConfig", mcp_groups: list[FunctionGroup]):
        """
        Initialize the composite function group.
        
        Args:
            config: The A365 MCP tooling configuration
            mcp_groups: List of MCP FunctionGroups to aggregate
        """
        super().__init__(config=config, instance_name="a365_mcp_tooling")
        self._mcp_groups = mcp_groups
    
    async def get_all_functions(
        self,
        filter_fn: Union["Callable[[Sequence[str]], Awaitable[Sequence[str]]]", None] = None,
    ) -> dict[str, "Function"]:
        """
        Aggregate all functions from all MCP groups.
        
        Functions are collected from each MCP group and merged. Function names
        from MCP groups are already namespaced (e.g., "mcp_client__tool_name"),
        so we preserve those namespaces to avoid conflicts.
        """
        all_functions = {}
        for mcp_group in self._mcp_groups:
            mcp_functions = await mcp_group.get_all_functions(filter_fn=filter_fn)
            all_functions.update(mcp_functions)
        return all_functions
    
    async def get_accessible_functions(
        self,
        filter_fn: Union["Callable[[Sequence[str]], Awaitable[Sequence[str]]]", None] = None,
    ) -> dict[str, "Function"]:
        """
        Aggregate accessible functions from all MCP groups.
        
        Respects include/exclude configuration from each MCP group.
        """
        all_functions = {}
        for mcp_group in self._mcp_groups:
            mcp_functions = await mcp_group.get_accessible_functions(filter_fn=filter_fn)
            all_functions.update(mcp_functions)
        return all_functions
    
    async def get_included_functions(
        self,
        filter_fn: Union["Callable[[Sequence[str]], Awaitable[Sequence[str]]]", None] = None,
    ) -> dict[str, "Function"]:
        """Aggregate included functions from all MCP groups."""
        all_functions = {}
        for mcp_group in self._mcp_groups:
            mcp_functions = await mcp_group.get_included_functions(filter_fn=filter_fn)
            all_functions.update(mcp_functions)
        return all_functions
    
    async def get_excluded_functions(
        self,
        filter_fn: Union["Callable[[Sequence[str]], Awaitable[Sequence[str]]]", None] = None,
    ) -> dict[str, "Function"]:
        """Aggregate excluded functions from all MCP groups."""
        all_functions = {}
        for mcp_group in self._mcp_groups:
            mcp_functions = await mcp_group.get_excluded_functions(filter_fn=filter_fn)
            all_functions.update(mcp_functions)
        return all_functions


class A365MCPToolingConfig(FunctionGroupBaseConfig, name="a365_mcp_tooling"):
    """Configuration for discovering and registering MCP servers from Agent 365.

    This configuration uses the A365 tooling service to discover MCP servers
    configured for the agent and registers them as NAT function groups.

    **Prerequisites:**
    - The `nvidia-nat-mcp` package must be installed.
    - Install with: ``pip install nvidia-nat-mcp`` or ``pip install 'nvidia-nat[mcp]'``
    - If installing from source: ``pip install 'nvidia-nat-a365[mcp]'``

    Example:
        ```yaml
        function_groups:
          - type: a365_mcp_tooling
            agentic_app_id: "your-agent-id"
            auth_token: "your-auth-token"
            orchestrator_name: "nemo-agent-toolkit"  # optional
        ```
    """

    agentic_app_id: str = Field(..., description="Agent 365 agentic app ID")
    auth_token: str | AuthenticationRef = Field(
        ..., description="Authentication token or reference to auth provider for A365 tooling gateway"
    )
    orchestrator_name: str | None = Field(
        default=None, description="Optional orchestrator name for user agent header"
    )
    # MCP Client Configuration - applied to all discovered servers
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
            raise ValueError("reconnect_max_backoff must be greater than or equal to reconnect_initial_backoff")
        return self


@register_function_group(config_type=A365MCPToolingConfig)
async def a365_mcp_tooling_function_group(config: A365MCPToolingConfig, builder: "Builder"):
    """Register MCP servers discovered from A365 tooling as NAT function groups.

    This function:
    1. Uses A365 tooling service to discover configured MCP servers
    2. Creates MCP client function groups for each discovered server
    3. Returns a composite function group containing all discovered tools

    Args:
        config: A365MCPToolingConfig with agent ID and auth token
        builder: NAT Builder instance

    Returns:
        FunctionGroup containing tools from all discovered MCP servers

    Raises:
        ImportError: If nvidia-nat-mcp package is not installed
        Exception: If MCP server discovery fails
    """
    try:
        from nat.plugins.mcp.client.client_config import MCPClientConfig, MCPServerConfig
    except ImportError as e:
        from nat.utils.optional_imports import OptionalImportError

        raise OptionalImportError(
            "nvidia-nat-mcp",
            additional_message=(
                "The A365 tooling feature requires the MCP client functionality. "
                "Install it with one of the following:\n"
                "  - pip install nvidia-nat-mcp\n"
                "  - pip install 'nvidia-nat[mcp]'\n"
                "  - pip install 'nvidia-nat-a365[mcp]' (if installing from source)"
            ),
        ) from e

    from nat.plugins.a365.tooling import A365ToolingService

    # Resolve auth token
    auth_token_str: str
    # Check for AuthenticationRef first (it's a subclass of str, so isinstance check must be specific)
    if isinstance(config.auth_token, AuthenticationRef):
        # Resolve from auth provider
        auth_provider = await builder.get_auth_provider(config.auth_token)
        
        # Get user_id from context if available (needed for OAuth flows)
        from nat.builder.context import Context
        user_id = Context.get().user_id
        
        auth_result = await auth_provider.authenticate(user_id=user_id)
        if not auth_result.credentials:
            raise ValueError("No credentials available from auth provider")
        
        # Extract bearer token from credentials
        # Support both BearerTokenCred and HeaderCred with Authorization header
        from nat.data_models.authentication import BearerTokenCred, HeaderCred
        from nat.authentication.interfaces import AUTHORIZATION_HEADER

        auth_token_str: str | None = None
        for cred in auth_result.credentials:
            if isinstance(cred, BearerTokenCred):
                # Standard Bearer token credential
                auth_token_str = cred.token.get_secret_value()
                break
            elif isinstance(cred, HeaderCred) and cred.name == AUTHORIZATION_HEADER:
                # Authorization header credential (e.g., "Bearer <token>")
                header_value = cred.value.get_secret_value()
                # Strip "Bearer " prefix if present
                if header_value.startswith("Bearer "):
                    auth_token_str = header_value[7:]  # Remove "Bearer " prefix
                else:
                    auth_token_str = header_value
                break

        if auth_token_str is None:
            raise ValueError(
                f"No bearer token found in auth provider credentials. "
                f"Found credential types: {[type(c).__name__ for c in auth_result.credentials]}"
            )
    else:
        # Plain string token (not an AuthenticationRef)
        auth_token_str = config.auth_token

    # Discover MCP servers from A365
    service = A365ToolingService()
    logger.info(f"Discovering MCP servers for agent {config.agentic_app_id}")
    servers = await service.list_tool_servers(
        agentic_app_id=config.agentic_app_id,
        auth_token=auth_token_str,
        orchestrator_name=config.orchestrator_name,
    )

    logger.info(f"Discovered {len(servers)} MCP servers, registering as function groups")

    # Register each discovered server as an MCP client function group
    from nat.plugins.mcp.client.client_impl import mcp_client_function_group

    # Collect MCP groups - we'll aggregate them using delegation pattern
    mcp_groups: list[FunctionGroup] = []
    
    # Use AsyncExitStack to keep all MCP client contexts open for the lifetime of this function group
    async with AsyncExitStack() as exit_stack:
        for server in servers:
            if not server.url:
                logger.warning(f"Skipping server {server.mcp_server_name}: no URL configured")
                continue

            # Convert tool_overrides dict to MCPToolOverrideConfig if provided
            mcp_tool_overrides = None
            if config.tool_overrides:
                from nat.plugins.mcp.client.client_config import MCPToolOverrideConfig

                mcp_tool_overrides = {
                    tool_name: MCPToolOverrideConfig(**override)
                    for tool_name, override in config.tool_overrides.items()
                }

            # Determine auth provider for this server
            # Priority: 1) Per-server override, 2) Same as A365 gateway (if AuthenticationRef), 3) None
            server_auth_provider = None
            if config.server_auth_providers and server.mcp_server_name in config.server_auth_providers:
                # Per-server override
                server_auth_provider = config.server_auth_providers[server.mcp_server_name]
                logger.debug(
                    f"Using per-server auth provider '{server_auth_provider}' for server '{server.mcp_server_name}'"
                )
            elif isinstance(config.auth_token, AuthenticationRef):
                # Use same auth provider as A365 gateway
                server_auth_provider = config.auth_token
                logger.debug(
                    f"Using A365 gateway auth provider for server '{server.mcp_server_name}'"
                )
            # If auth_token is a string, we don't pass auth to MCP servers (they may have their own)

            # Create MCP client config for this server
            mcp_config = MCPClientConfig(
                server=MCPServerConfig(
                    transport="streamable-http",
                    url=server.url,
                    auth_provider=server_auth_provider,
                ),
                tool_call_timeout=config.tool_call_timeout,
                auth_flow_timeout=config.auth_flow_timeout,
                reconnect_enabled=config.reconnect_enabled,
                reconnect_max_attempts=config.reconnect_max_attempts,
                reconnect_initial_backoff=config.reconnect_initial_backoff,
                reconnect_max_backoff=config.reconnect_max_backoff,
                session_aware_tools=config.session_aware_tools,
                max_sessions=config.max_sessions,
                session_idle_timeout=config.session_idle_timeout,
                tool_overrides=mcp_tool_overrides,
            )

            # Register the MCP client function group
            # Note: mcp_client_function_group is wrapped as an async context manager by @register_function_group
            # We use AsyncExitStack to keep all contexts open for the lifetime of the composite group
            try:
                mcp_group = await exit_stack.enter_async_context(
                    mcp_client_function_group(mcp_config, builder)
                )
                mcp_groups.append(mcp_group)
                
                logger.info(
                    f"Registered MCP server '{server.mcp_server_name}'"
                )
            except Exception as e:
                logger.error(f"Failed to register MCP server '{server.mcp_server_name}': {e}", exc_info=True)
                continue

        # Create composite group that delegates to all MCP groups
        if not mcp_groups:
            logger.warning(
                f"No MCP servers successfully registered for agent {config.agentic_app_id}. "
                f"Discovered {len(servers)} servers but none could be registered."
            )
        
        composite_group = A365MCPToolingFunctionGroup(config=config, mcp_groups=mcp_groups)
        
        # Get total count of functions from all groups for logging
        all_functions = await composite_group.get_all_functions()
        logger.info(
            f"A365 MCP tooling: registered {len(all_functions)} total tools from {len(mcp_groups)} servers"
        )

        # Yield the composite function group while keeping all MCP client contexts open
        yield composite_group
