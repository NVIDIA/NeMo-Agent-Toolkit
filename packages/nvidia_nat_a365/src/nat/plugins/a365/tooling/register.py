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

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack

from nat.builder.builder import Builder
from nat.builder.function import Function, FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.component_ref import AuthenticationRef
from nat.plugins.a365.exceptions import A365AuthenticationError, A365ConfigurationError, A365SDKError
from nat.plugins.a365.tooling.tooling_config import A365MCPToolingConfig

logger = logging.getLogger(__name__)


class A365MCPToolingFunctionGroup(FunctionGroup):
    """
    Composite FunctionGroup that aggregates functions from multiple MCP servers.
    
    Instead of merging functions into a single group, this class delegates to
    multiple MCP FunctionGroups and aggregates their results. This preserves
    the original function bindings and avoids double-wrapping.
    """
    
    def __init__(self, config: A365MCPToolingConfig, mcp_groups: list[FunctionGroup]):
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
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
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
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
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
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
        """
        Aggregate included functions from all MCP groups.

        Returns only functions that are explicitly included via each MCP group's
        include configuration.
        """
        all_functions = {}
        for mcp_group in self._mcp_groups:
            mcp_functions = await mcp_group.get_included_functions(filter_fn=filter_fn)
            all_functions.update(mcp_functions)
        return all_functions
    
    async def get_excluded_functions(
        self,
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
        """
        Aggregate excluded functions from all MCP groups.

        Returns functions that are explicitly excluded via each MCP group's
        exclude configuration.
        """
        all_functions = {}
        for mcp_group in self._mcp_groups:
            mcp_functions = await mcp_group.get_excluded_functions(filter_fn=filter_fn)
            all_functions.update(mcp_functions)
        return all_functions


@register_function_group(config_type=A365MCPToolingConfig)
async def a365_mcp_tooling_function_group(config: A365MCPToolingConfig, builder: Builder):
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
        OptionalImportError: If nvidia-nat-mcp package is not installed
        A365AuthenticationError: If authentication fails when resolving auth token or discovering servers
        A365SDKError: If MCP server discovery fails
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
                "  - uv pip install nvidia-nat-mcp\n"
                "  - uv pip install 'nvidia-nat[mcp]'\n"
                "  - uv pip install 'nvidia-nat-a365[mcp]' (if installing from source)"
            ),
        ) from e

    from nat.plugins.a365.tooling import A365ToolingService

    auth_token_str: str
    if isinstance(config.auth_token, AuthenticationRef):
        auth_provider = await builder.get_auth_provider(config.auth_token)
        
        # Get user_id from context if available (needed for OAuth flows)
        from nat.builder.context import Context
        user_id = Context.get().user_id
        
        auth_result = await auth_provider.authenticate(user_id=user_id)
        if not auth_result.credentials:
            raise A365AuthenticationError("No credentials available from auth provider")
        
        # Support both BearerTokenCred and HeaderCred with Authorization header
        from nat.data_models.authentication import BearerTokenCred, HeaderCred
        from nat.authentication.interfaces import AUTHORIZATION_HEADER

        auth_token_str: str | None = None
        for cred in auth_result.credentials:
            if isinstance(cred, BearerTokenCred):
                auth_token_str = cred.token.get_secret_value()
                break
            elif isinstance(cred, HeaderCred) and cred.name == AUTHORIZATION_HEADER:
                header_value = cred.value.get_secret_value()
                if header_value.startswith("Bearer "):
                    auth_token_str = header_value[7:]
                else:
                    auth_token_str = header_value
                break

        if auth_token_str is None:
            raise A365AuthenticationError(
                f"No bearer token found in auth provider credentials. "
                    f"Found credential types: {[type(c).__name__ for c in auth_result.credentials]}"
            )
    else:
        auth_token_str = config.auth_token

    service = A365ToolingService()
    logger.info(f"Discovering MCP servers for agent {config.agentic_app_id}")
    try:
        servers = await service.list_tool_servers(
            agentic_app_id=config.agentic_app_id,
            auth_token=auth_token_str,
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "authentication" in error_msg or "unauthorized" in error_msg or "forbidden" in error_msg:
            raise A365AuthenticationError(
                f"Failed to authenticate with A365 tooling gateway: {str(e)}",
                original_error=e
            ) from e
        else:
            raise A365SDKError(
                f"Failed to discover MCP servers from A365 tooling gateway: {str(e)}",
                sdk_component="McpToolServerConfigurationService",
                original_error=e
            ) from e

    logger.info(f"Discovered {len(servers)} MCP servers, registering as function groups")

    from nat.plugins.mcp.client.client_impl import mcp_client_function_group

    # Convert tool_overrides dict to MCPToolOverrideConfig if provided (once, before the loop)
    mcp_tool_overrides = None
    if config.tool_overrides:
        from nat.plugins.mcp.client.client_config import MCPToolOverrideConfig
        from pydantic import ValidationError

        try:
            mcp_tool_overrides = {
                tool_name: MCPToolOverrideConfig(**override)
                for tool_name, override in config.tool_overrides.items()
            }
        except ValidationError as e:
            raise A365ConfigurationError(
                f"Invalid tool_overrides configuration: {str(e)}"
            ) from e

    mcp_groups: list[FunctionGroup] = []
    
    # Use AsyncExitStack to keep all MCP client contexts open for the lifetime of this function group
    async with AsyncExitStack() as exit_stack:
        for server in servers:
            if not server.url:
                server_name = getattr(server, "mcp_server_name", "unknown") or "unknown"
                logger.warning(f"Skipping server {server_name}: no URL configured")
                continue

            server_name = getattr(server, "mcp_server_name", None) or "unknown"

            # Priority: 1) Per-server override, 2) A365 gateway auth (if AuthenticationRef), 3) None
            server_auth_provider = None
            if config.server_auth_providers and server_name in config.server_auth_providers:
                server_auth_provider = config.server_auth_providers[server_name]
                logger.debug(
                    f"Using per-server auth provider '{server_auth_provider}' for server '{server_name}'"
                )
            elif isinstance(config.auth_token, AuthenticationRef):
                server_auth_provider = config.auth_token
                logger.debug(
                    f"Using A365 gateway auth provider for server '{server_name}'"
                )

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

            # mcp_client_function_group is an async context manager; AsyncExitStack keeps
            # all contexts open for the lifetime of the composite group
            try:
                mcp_group = await exit_stack.enter_async_context(
                    mcp_client_function_group(mcp_config, builder)
                )
                mcp_groups.append(mcp_group)
                
                logger.info(
                    f"Registered MCP server '{server_name}'"
                )
            except Exception as e:
                logger.error(f"Failed to register MCP server '{server_name}': {e}", exc_info=True)
                continue

        if not mcp_groups:
            logger.warning(
                f"No MCP servers successfully registered for agent {config.agentic_app_id}. "
                f"Discovered {len(servers)} servers but none could be registered."
            )
        
        composite_group = A365MCPToolingFunctionGroup(config=config, mcp_groups=mcp_groups)
        
        all_functions = await composite_group.get_all_functions()
        logger.info(
            f"A365 MCP tooling: registered {len(all_functions)} total tools from {len(mcp_groups)} servers"
        )

        yield composite_group
