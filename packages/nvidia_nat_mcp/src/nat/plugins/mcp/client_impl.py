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
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta

import aiorwlock
from pydantic import BaseModel

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.plugins.mcp.client_base import MCPBaseClient
from nat.plugins.mcp.client_config import MCPClientConfig
from nat.plugins.mcp.client_config import MCPToolOverrideConfig
from nat.plugins.mcp.utils import truncate_session_id

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Container for all session-related data."""
    client: MCPBaseClient
    last_activity: datetime
    ref_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # lifetime task to respect task boundaries
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    lifetime_task: asyncio.Task | None = None


class MCPFunctionGroup(FunctionGroup):
    """
    A specialized FunctionGroup for MCP clients that includes MCP-specific attributes
    with session management.

    Locking model (simple + safe; occasional 'temporarily unavailable' is acceptable).

    RW semantics:
    - Multiple readers may hold the reader lock concurrently.
    - While any reader holds the lock, writers cannot proceed.
    - While the writer holds the lock, no new readers can proceed.

    Data:
    - _sessions: dict[str, SessionData]; SessionData = {client, last_activity, ref_count, lock}.

    Locks:
    - _session_rwlock (aiorwlock.RWLock)
    • Reader: very short sections — dict lookups, ref_count ++/--, touch last_activity.
    • Writer: structural changes — create session entries, enforce limits, remove on cleanup.
    - SessionData.lock (asyncio.Lock)
    • Protects per-session ref_count only, taken only while holding RW *reader*.
    • last_activity: written without session lock (timestamp races acceptable for cleanup heuristic).

    Ordering & awaits:
    - Always acquire RWLock (reader/writer) before SessionData.lock; never the reverse.
    - Never await network I/O under the writer (client creation is the one intentional exception).
    - Client close happens after releasing the writer.

    Cleanup:
    - Under writer: find inactive (ref_count == 0 and idle > max_age), pop from _sessions, stash clients.
    - After writer: await client.__aexit__() for each stashed client.
    - TOCTOU race: cleanup may read ref_count==0 then a usage increments it; accepted, yields None gracefully.

    Invariants:
    - ref_count > 0 prevents cleanup.
    - Usage context increments ref_count before yielding and decrements on exit.
    - If a session disappears between ensure/use, callers return "Tool temporarily unavailable".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MCP client attributes with proper typing
        self._mcp_client = None  # Will be set to the actual MCP client instance
        self._mcp_client_server_name: str | None = None
        self._mcp_client_transport: str | None = None

        # Session management - consolidated data structure
        self._sessions: dict[str, SessionData] = {}

        # Use RWLock for better concurrency: multiple readers (tool calls) can access
        # existing sessions simultaneously, while writers (create/delete) get exclusive access
        self._session_rwlock = aiorwlock.RWLock()
        # Throttled cleanup control
        self._last_cleanup_check: datetime = datetime.now()
        self._cleanup_check_interval: timedelta = timedelta(minutes=5)

        # Shared components for session client creation
        self._shared_auth_provider: AuthProviderBase | None = None
        self._client_config: MCPClientConfig | None = None

        # Auth provider config defaults (set when auth provider is assigned)
        self._default_user_id: str | None = None
        self._allow_default_user_id_for_tool_calls: bool = True

        # Use random session id for testing only
        self._use_random_session_id_for_testing: bool = False

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
        return len(self._sessions)

    @property
    def session_limit(self) -> int:
        """Maximum allowed sessions."""
        return self._client_config.max_sessions if self._client_config else 100

    def _get_random_session_id(self) -> str:
        """Get a random session ID."""
        import uuid
        return str(uuid.uuid4())

    def _get_session_id_from_context(self) -> str | None:
        """Get the session ID from the current context."""
        try:
            from nat.builder.context import Context as _Ctx

            # Get session id from context, authentication is done per-websocket session for tool calls
            session_id = None
            # get session id from cookies if session_aware_tools is enabled
            if self._client_config and self._client_config.session_aware_tools:
                cookies = getattr(_Ctx.get().metadata, "cookies", None)
                if cookies:
                    if self._use_random_session_id_for_testing:
                        # This path is for testing only and should not be used in production
                        session_id = self._get_random_session_id()
                    else:
                        session_id = cookies.get("nat-session")

            if not session_id:
                # use default user id if allowed
                if self._shared_auth_provider and self._allow_default_user_id_for_tool_calls:
                    session_id = self._default_user_id
            return session_id
        except Exception:
            return None

    async def cleanup_sessions(self, max_age: timedelta | None = None) -> int:
        """
        Manually trigger cleanup of inactive sessions.

        Args:
            max_age: Maximum age for sessions before cleanup. If None, uses configured timeout.

        Returns:
            Number of sessions cleaned up.
        """
        sessions_before = len(self._sessions)
        await self._cleanup_inactive_sessions(max_age)
        sessions_after = len(self._sessions)
        return sessions_before - sessions_after

    async def _cleanup_inactive_sessions(self, max_age: timedelta | None = None):
        """Remove clients for sessions inactive longer than max_age.

        This method uses the RWLock writer to ensure thread-safe cleanup.
        """
        if max_age is None:
            max_age = self._client_config.session_idle_timeout if self._client_config else timedelta(hours=1)

        to_close: list[tuple[str, SessionData]] = []

        async with self._session_rwlock.writer:
            current_time = datetime.now()
            inactive_sessions = []

            for session_id, session_data in self._sessions.items():
                # Skip cleanup if session is actively being used
                if session_data.ref_count > 0:
                    continue

                if current_time - session_data.last_activity > max_age:
                    inactive_sessions.append(session_id)

            for session_id in inactive_sessions:
                try:
                    logger.info("Cleaning up inactive session client: %s", truncate_session_id(session_id))
                    session_data = self._sessions[session_id]
                    # Close the client connection
                    if session_data:
                        to_close.append((session_id, session_data))
                except Exception as e:
                    logger.warning("Error cleaning up session client %s: %s", truncate_session_id(session_id), e)
                finally:
                    # Always remove from tracking to prevent leaks, even if close failed
                    self._sessions.pop(session_id, None)
                    logger.info("Cleaned up session tracking for: %s", truncate_session_id(session_id))
                    logger.info(" Total sessions: %d", len(self._sessions))

        # Close sessions outside the writer lock to avoid deadlock
        for session_id, sdata in to_close:
            try:
                if sdata.stop_event and sdata.lifetime_task:
                    if not sdata.lifetime_task.done():
                        # Instead of directly exiting the task, set the stop event
                        # and wait for the task to exit. This ensures the cancel scope
                        # is entered and exited in the same task.
                        sdata.stop_event.set()
                        await sdata.lifetime_task  # __aexit__ runs in that task
                    else:
                        logger.debug("Session client %s lifetime task already done", truncate_session_id(session_id))
                else:
                    # add fallback to ensure we clean up the client
                    logger.warning("Session client %s lifetime task not found, cleaning up client",
                                   truncate_session_id(session_id))
                    await sdata.client.__aexit__(None, None, None)
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
        if self._shared_auth_provider:
            if self._default_user_id and session_id == self._default_user_id:
                return self.mcp_client

        # Fast path: check if session already exists (reader lock for concurrent access)
        async with self._session_rwlock.reader:
            if session_id in self._sessions:
                # Update last activity for existing client
                self._sessions[session_id].last_activity = datetime.now()
                return self._sessions[session_id].client

        # Check session limit before creating new client (outside writer lock to avoid deadlock)
        if self._client_config and len(self._sessions) >= self._client_config.max_sessions:
            # Try cleanup first to free up space
            await self._cleanup_inactive_sessions()

        # Slow path: create session with writer lock for exclusive access
        async with self._session_rwlock.writer:
            # Double-check after acquiring writer lock (another coroutine might have created it)
            if session_id in self._sessions:
                self._sessions[session_id].last_activity = datetime.now()
                return self._sessions[session_id].client

            # Re-check session limit inside writer lock
            if self._client_config and len(self._sessions) >= self._client_config.max_sessions:
                logger.warning("Session limit reached (%d), rejecting new session: %s",
                               self._client_config.max_sessions,
                               truncate_session_id(session_id))
                raise RuntimeError(f"Tool unavailable: Maximum concurrent sessions "
                                   f"({self._client_config.max_sessions}) exceeded.")

            # Create session client lazily
            logger.info("Creating new MCP client for session: %s", truncate_session_id(session_id))
            session_client, stop_event, lifetime_task = await self._create_session_client(session_id)
            session_data = SessionData(
                client=session_client,
                last_activity=datetime.now(),
                ref_count=0,
                stop_event=stop_event,
                lifetime_task=lifetime_task,
            )

            # Cache the session data
            self._sessions[session_id] = session_data
            logger.info(" Total sessions: %d", len(self._sessions))
            return session_client

    @asynccontextmanager
    async def _session_usage_context(self, session_id: str):
        """Context manager to track active session usage and prevent cleanup."""
        # Ensure session exists - create it if it doesn't
        if session_id not in self._sessions:
            # Create session client first
            await self._get_session_client(session_id)  # START read phase: bump ref_count under reader + session lock

        async with self._session_rwlock.reader:
            sdata = self._sessions.get(session_id)
            if not sdata:
                # this can happen if the session is cleaned up between the check and the lock
                # this is rare and we can just return that the tool is temporarily unavailable
                yield None
                return
            async with sdata.lock:
                sdata.ref_count += 1
                client = sdata.client  # capture
        # END read phase (release reader before long await)

        try:
            yield client
        finally:
            # Brief read phase to decrement ref_count and touch activity
            async with self._session_rwlock.reader:
                sdata = self._sessions.get(session_id)
                if sdata:
                    async with sdata.lock:
                        sdata.ref_count -= 1
                        sdata.last_activity = datetime.now()

    async def _create_session_client(self, session_id: str) -> tuple[MCPBaseClient, asyncio.Event, asyncio.Task]:
        """Create a new MCP client instance for the session."""
        from nat.plugins.mcp.client_base import MCPStreamableHTTPClient

        config = self._client_config
        if not config:
            raise RuntimeError("Client config not initialized")

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

        ready = asyncio.Event()
        stop_event = asyncio.Event()

        async def _lifetime():
            """
            Create a lifetime task to respect task boundaries and ensure the
            cancel scope is entered and exited in the same task.
            """
            try:
                async with client:
                    ready.set()
                    await stop_event.wait()
            except Exception:
                ready.set()  # Ensure we don't hang the waiter
                raise

        task = asyncio.create_task(_lifetime(), name=f"mcp-session-{truncate_session_id(session_id)}")

        # Wait for initialization with timeout to prevent infinite hangs
        timeout = config.tool_call_timeout.total_seconds() if config else 300
        try:
            await asyncio.wait_for(ready.wait(), timeout=timeout)
        except TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.error("Session client initialization timed out after %ds for %s",
                         timeout,
                         truncate_session_id(session_id))
            raise RuntimeError(f"Session client initialization timed out after {timeout}s")

        # Check if initialization failed before ready was set
        if task.done():
            try:
                await task  # Re-raise exception if the task failed
            except Exception as e:
                logger.error("Failed to initialize session client for %s: %s", truncate_session_id(session_id), e)
                raise RuntimeError(f"Failed to initialize session client: {e}") from e

        logger.info("Created session client for session: %s", truncate_session_id(session_id))
        # NOTE: caller will place client into SessionData and attach stop_event/task
        return client, stop_event, task


def mcp_session_tool_function(tool, function_group: MCPFunctionGroup):
    """Create a session-aware NAT function for an MCP tool.

    Routes each invocation to the appropriate per-session MCP client while
    preserving the original tool input schema, converters, and description.
    """
    from nat.builder.function import FunctionInfo

    def _convert_from_str(input_str: str) -> tool.input_schema:
        return tool.input_schema.model_validate_json(input_str)

    async def _response_fn(tool_input: BaseModel | None = None, **kwargs) -> str:
        """Response function for the session-aware tool."""
        try:
            # Route to the appropriate session client
            session_id = function_group._get_session_id_from_context()

            # If no session is available and default-user fallback is disabled, deny the call
            if function_group._shared_auth_provider and session_id is None:
                return "User not authorized to call the tool"

            # Check if this is the default user - if so, use base client directly
            if (not function_group._shared_auth_provider or session_id == function_group._default_user_id):
                # Use base client directly for default user
                client = function_group.mcp_client
                session_tool = await client.get_tool(tool.name)
            else:
                # Use session usage context to prevent cleanup during tool execution
                async with function_group._session_usage_context(session_id) as client:
                    if client is None:
                        return "Tool temporarily unavailable. Try again."
                    session_tool = await client.get_tool(tool.name)

            # Preserve original calling convention
            if tool_input:
                args = tool_input.model_dump()
                return await session_tool.acall(args)

            _ = session_tool.input_schema.model_validate(kwargs)
            return await session_tool.acall(kwargs)
        except Exception as e:
            logger.warning("Error calling tool %s", tool.name, exc_info=True)
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
        # For interactive OAuth2: from config. For service accounts: defaults to server URL
        base_user_id = getattr(auth_provider.config, 'default_user_id', str(
            config.server.url)) if auth_provider else None
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

    # Set auth provider config defaults
    # For interactive OAuth2: use config values
    # For service accounts: default_user_id = server URL, allow_default_user_id_for_tool_calls = True
    if auth_provider:
        group._default_user_id = getattr(auth_provider.config, 'default_user_id', str(config.server.url))
        group._allow_default_user_id_for_tool_calls = getattr(auth_provider.config,
                                                              'allow_default_user_id_for_tool_calls',
                                                              True)
    else:
        group._default_user_id = None
        group._allow_default_user_id_for_tool_calls = True

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
            tool_fn = mcp_session_tool_function(tool, group)

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
