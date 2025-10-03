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

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timedelta

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.plugins.mcp.client_base import MCPBaseClient
from nat.plugins.mcp.client_config import MCPClientConfig
from nat.plugins.mcp.client_config import MCPToolOverrideConfig
from nat.plugins.mcp.utils import truncate_session_id

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

        # Session management attributes for per-session isolation
        self._session_clients: dict[str, MCPBaseClient] = {}
        self._session_last_activity: dict[str, datetime] = {}
        self._session_ref_counts: dict[str, int] = {}  # Track active operations per session
        self._cleanup_lock = asyncio.Lock()
        self._session_creation_lock = asyncio.Lock()
        # Throttled cleanup control
        self._last_cleanup_check: datetime = datetime.now()
        self._cleanup_check_interval: timedelta = timedelta(minutes=5)

        # Shared components for session client creation
        self._shared_auth_provider = None
        self._client_config: MCPClientConfig | None = None

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

    @property
    def session_count(self) -> int:
        """Current number of active sessions."""
        return len(self._session_clients)

    @property
    def session_limit(self) -> int:
        """Maximum allowed sessions."""
        return self._client_config.max_sessions if self._client_config else 100

    def _get_session_id_from_context(self) -> str | None:
        """Get the session ID from the current context."""
        try:
            from nat.builder.context import Context as _Ctx

            # Get session id from context, authentication is done per-websocket session for tool calls
            session_id = None
            cookies = getattr(_Ctx.get().metadata, "cookies", None)
            if cookies:
                session_id = cookies.get("nat-session")

            if not session_id:
                # use default user id if allowed
                if self._shared_auth_provider and \
                    self._shared_auth_provider.config.allow_default_user_id_for_tool_calls:
                    session_id = self._shared_auth_provider.config.default_user_id
            return session_id
        except Exception:
            return None

    async def _cleanup_inactive_sessions(self, max_age: timedelta | None = None):
        """Remove clients for sessions inactive longer than max_age.

        This method uses its own cleanup_lock to ensure thread-safe cleanup.
        It can be called from within the session_creation_lock without deadlock.
        """
        if max_age is None:
            max_age = self._client_config.session_idle_timeout if self._client_config else timedelta(hours=1)

        async with self._cleanup_lock:
            current_time = datetime.now()
            inactive_sessions = []

            for session_id, last_activity in self._session_last_activity.items():
                # Skip cleanup if session is actively being used
                if session_id in self._session_ref_counts and self._session_ref_counts[session_id] > 0:
                    continue

                if current_time - last_activity > max_age:
                    inactive_sessions.append(session_id)

            for session_id in inactive_sessions:
                try:
                    logger.info("Cleaning up inactive session client: %s", truncate_session_id(session_id))
                    client = self._session_clients[session_id]
                    # Close the client connection
                    await client.__aexit__(None, None, None)
                    del self._session_clients[session_id]
                    del self._session_last_activity[session_id]
                    # Clean up reference count if it exists
                    if session_id in self._session_ref_counts:
                        del self._session_ref_counts[session_id]
                    logger.info("Cleaned up inactive session client: %s", truncate_session_id(session_id))
                    logger.info(" Total sessions: %d", len(self._session_clients))
                except Exception as e:
                    logger.warning("Error cleaning up session client %s: %s", truncate_session_id(session_id), e)

    async def _get_session_client(self, session_id: str) -> MCPBaseClient:
        """Get the appropriate MCP client for the session."""
        # Throttled cleanup on access
        now = datetime.now()
        if now - self._last_cleanup_check > self._cleanup_check_interval:
            await self._cleanup_inactive_sessions()
            self._last_cleanup_check = now

        # If the session_id equals the configured default_user_id use the base client
        # instead of creating a per-session client
        default_uid = self._shared_auth_provider.config.default_user_id
        if default_uid and session_id == default_uid:
            return self.mcp_client

        # Fast path: check if session already exists (no lock needed)
        if session_id in self._session_clients:
            # Update last activity for existing client
            self._session_last_activity[session_id] = datetime.now()
            return self._session_clients[session_id]

        # Slow path: create session with proper locking
        async with self._session_creation_lock:
            # Double-check after acquiring lock (another coroutine might have created it)
            if session_id in self._session_clients:
                self._session_last_activity[session_id] = datetime.now()
                return self._session_clients[session_id]

            # Check session limit before creating new client
            if len(self._session_clients) >= self._client_config.max_sessions:
                # Try cleanup first to free up space
                await self._cleanup_inactive_sessions()

                # Re-check after cleanup
                if len(self._session_clients) >= self._client_config.max_sessions:
                    logger.warning("Session limit reached (%d), rejecting new session: %s",
                                   self._client_config.max_sessions,
                                   truncate_session_id(session_id))
                    raise RuntimeError(f"Service temporarily unavailable: Maximum concurrent sessions \
                            ({self._client_config.max_sessions}) exceeded. Please try again later.")

            # Create session client lazily
            logger.info("Creating new MCP client for session: %s", truncate_session_id(session_id))
            session_client = await self._create_session_client(session_id)

            # Cache the client
            self._session_clients[session_id] = session_client
            self._session_last_activity[session_id] = datetime.now()
            return session_client

    @asynccontextmanager
    async def _session_usage_context(self, session_id: str):
        """Context manager to track active session usage and prevent cleanup."""
        # Increment reference count
        self._session_ref_counts[session_id] = self._session_ref_counts.get(session_id, 0) + 1

        try:
            yield
        finally:
            # Decrement reference count
            if session_id in self._session_ref_counts:
                self._session_ref_counts[session_id] -= 1
                if self._session_ref_counts[session_id] <= 0:
                    del self._session_ref_counts[session_id]

    async def _create_session_client(self, session_id: str) -> MCPBaseClient:
        """Create a new MCP client instance for the session."""
        from nat.plugins.mcp.client_base import MCPStreamableHTTPClient

        config = self._client_config

        if config.server.transport == "streamable-http":
            client = MCPStreamableHTTPClient(
                str(config.server.url),
                auth_provider=self._shared_auth_provider,
                user_id=session_id,  # Pass session_id as user_id for cache isolation
                tool_call_timeout=config.tool_call_timeout,
                auth_flow_timeout=config.auth_flow_timeout,
                reconnect_enabled=config.reconnect_enabled,
                reconnect_max_attempts=config.reconnect_max_attempts,
                reconnect_initial_backoff=config.reconnect_initial_backoff,
                reconnect_max_backoff=config.reconnect_max_backoff)
        else:
            # per-user sessions are only supported for streamable-http transport
            raise ValueError(f"Unsupported transport: {config.server.transport}")

        # Initialize the client
        await client.__aenter__()

        logger.info("Created session client for session: %s", truncate_session_id(session_id))
        logger.info(" Total sessions: %d", len(self._session_clients))
        return client


def mcp_session_tool_function(tool, function_group: MCPFunctionGroup):
    """Create a session-aware NAT function for an MCP tool.

    Routes each invocation to the appropriate per-session MCP client while
    preserving the original tool input schema, converters, and description.
    """
    from pydantic import BaseModel

    from nat.builder.function import FunctionInfo

    def _convert_from_str(input_str: str) -> tool.input_schema:
        return tool.input_schema.model_validate_json(input_str)

    async def _response_fn(tool_input: BaseModel | None = None, **kwargs) -> str:
        try:
            # Route to the appropriate session client
            session_id = function_group._get_session_id_from_context()

            # If no session is available and default-user fallback is disabled, deny the call
            if session_id is None:
                return "User not authorized to call the tool"

            # Use session usage context to prevent cleanup during tool execution
            async with function_group._session_usage_context(session_id):
                client = await function_group._get_session_client(session_id)

                # Resolve the tool from the routed client
                session_tool = await client.get_tool(tool.name)

                # Preserve original calling convention
                if tool_input:
                    args = tool_input.model_dump()
                    return await session_tool.acall(args)

                _ = session_tool.input_schema.model_validate(kwargs)
                return await session_tool.acall(kwargs)
        except Exception as e:
            if tool_input:
                logger.warning("Error calling tool %s with serialized input: %s",
                               tool.name,
                               tool_input.model_dump(),
                               exc_info=True)
            else:
                logger.warning("Error calling tool %s with input: %s", tool.name, kwargs, exc_info=True)
            return str(e)

    return FunctionInfo.create(single_fn=_response_fn,
                               description=tool.description,
                               input_schema=tool.input_schema,
                               converters=[_convert_from_str])


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
                                tool_call_timeout=config.tool_call_timeout,
                                auth_flow_timeout=config.auth_flow_timeout,
                                reconnect_enabled=config.reconnect_enabled,
                                reconnect_max_attempts=config.reconnect_max_attempts,
                                reconnect_initial_backoff=config.reconnect_initial_backoff,
                                reconnect_max_backoff=config.reconnect_max_backoff)
    elif config.server.transport == "sse":
        client = MCPSSEClient(str(config.server.url),
                              tool_call_timeout=config.tool_call_timeout,
                              auth_flow_timeout=config.auth_flow_timeout,
                              reconnect_enabled=config.reconnect_enabled,
                              reconnect_max_attempts=config.reconnect_max_attempts,
                              reconnect_initial_backoff=config.reconnect_initial_backoff,
                              reconnect_max_backoff=config.reconnect_max_backoff)
    elif config.server.transport == "streamable-http":
        # Use default_user_id for the base client
        base_user_id = auth_provider.config.default_user_id if auth_provider else None
        client = MCPStreamableHTTPClient(str(config.server.url),
                                         auth_provider=auth_provider,
                                         user_id=base_user_id,
                                         tool_call_timeout=config.tool_call_timeout,
                                         auth_flow_timeout=config.auth_flow_timeout,
                                         reconnect_enabled=config.reconnect_enabled,
                                         reconnect_max_attempts=config.reconnect_max_attempts,
                                         reconnect_initial_backoff=config.reconnect_initial_backoff,
                                         reconnect_max_backoff=config.reconnect_max_backoff)
    else:
        raise ValueError(f"Unsupported transport: {config.server.transport}")

    logger.info("Configured to use MCP server at %s", client.server_name)

    # Create the MCP function group
    group = MCPFunctionGroup(config=config)

    # Store shared components for session client creation
    group._shared_auth_provider = auth_provider
    group._client_config = config

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

            # Create the tool function according to configuration
            if config.session_aware_tools:
                tool_fn = mcp_session_tool_function(tool, group)
            else:
                from nat.plugins.mcp.tool import mcp_tool_function  # noqa: F401
                tool_fn = mcp_tool_function(tool)

            # Normalize optional typing for linter/type-checker compatibility
            single_fn = tool_fn.single_fn
            if single_fn is None:
                # Should not happen because FunctionInfo always sets a single_fn
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
