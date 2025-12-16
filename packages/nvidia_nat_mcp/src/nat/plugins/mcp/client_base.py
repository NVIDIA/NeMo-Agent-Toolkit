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

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncGenerator
from collections.abc import Callable
from contextlib import AsyncExitStack
from contextlib import asynccontextmanager
from datetime import timedelta

import anyio
import httpx

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent
from nat.authentication.interfaces import AuthenticatedContext
from nat.authentication.interfaces import AuthFlowType
from nat.authentication.interfaces import AuthProviderBase
from nat.plugins.mcp.exception_handler import convert_to_mcp_error
from nat.plugins.mcp.exception_handler import format_mcp_error
from nat.plugins.mcp.exception_handler import mcp_exception_handler
from nat.plugins.mcp.exceptions import MCPError
from nat.plugins.mcp.exceptions import MCPToolNotFoundError
from nat.plugins.mcp.utils import model_from_mcp_schema
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


class AuthAdapter(httpx.Auth):
    """
    httpx.Auth adapter for authentication providers.
    Converts AuthProviderBase to httpx.Auth interface for dynamic token management.
    """

    def __init__(self, auth_provider: AuthProviderBase, user_id: str | None = None):
        self.auth_provider = auth_provider
        self.user_id = user_id  # Session-specific user ID for cache isolation
        # each adapter instance has its own lock to avoid unnecessary delays for multiple clients
        self._lock = anyio.Lock()
        # Track whether we're currently in an interactive authentication flow
        self.is_authenticating = False

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Add authentication headers to the request using NAT auth provider."""
        async with self._lock:
            try:
                # Get auth headers from the NAT auth provider:
                # 1. If discovery is yet to done this will return None and request will be sent without auth header.
                # 2. If discovery is done, this will return the auth header from cache if the token is still valid
                auth_headers = await self._get_auth_headers(request=request, response=None)
                request.headers.update(auth_headers)
            except Exception as e:
                logger.info("Failed to get auth headers: %s", e)
                # Continue without auth headers if auth fails

            response = yield request

            # Handle 401 responses by retrying with fresh auth
            if response.status_code == 401:
                try:
                    # 401 can happen if:
                    # 1. The request was sent without auth header
                    # 2. The auth headers are invalid
                    # 3. The auth headers are expired
                    # 4. The auth headers are revoked
                    # 5. Auth config on the MCP server has changed
                    # In this case we attempt to re-run discovery and authentication

                    # Signal that we're entering interactive auth flow
                    self.is_authenticating = True
                    logger.debug("Starting authentication flow due to 401 response")

                    auth_headers = await self._get_auth_headers(request=request, response=response)
                    request.headers.update(auth_headers)
                    yield request  # Retry the request
                except Exception as e:
                    logger.info("Failed to refresh auth after 401: %s", e)
                    raise
                finally:
                    # Signal that auth flow is complete
                    self.is_authenticating = False
                    logger.debug("Authentication flow completed")
        return

    async def _get_auth_headers(self,
                                request: httpx.Request | None = None,
                                response: httpx.Response | None = None) -> dict[str, str]:
        """Get authentication headers from the NAT auth provider."""
        try:
            # Use the user_id passed to this AuthAdapter instance
            auth_result = await self.auth_provider.authenticate(user_id=self.user_id, response=response)

            # Build headers from credentials
            from nat.data_models.authentication import BearerTokenCred
            from nat.data_models.authentication import HeaderCred
            headers = {}

            for cred in auth_result.credentials:
                if isinstance(cred, BearerTokenCred):
                    # Standard Bearer token
                    token = cred.token.get_secret_value()
                    headers["Authorization"] = f"Bearer {token}"
                elif isinstance(cred, HeaderCred):
                    # Generic header credential (supports custom formats and service accounts)
                    headers[cred.name] = cred.value.get_secret_value()

            return headers
        except Exception as e:
            logger.warning("Failed to get auth token: %s", e)
            return {}


class MCPBaseClient(ABC):
    """
    Base client for creating a MCP transport session and connecting to an MCP server

    Args:
        transport (str): The type of client to use ('sse', 'stdio', or 'streamable-http')
        auth_provider (AuthProviderBase | None): Optional authentication provider for Bearer token injection
        tool_call_timeout (timedelta): Timeout for tool calls when authentication is not required
        auth_flow_timeout (timedelta): Extended timeout for tool calls that may require interactive authentication
        reconnect_enabled (bool): Whether to automatically reconnect on connection failures
        reconnect_max_attempts (int): Maximum number of reconnection attempts
        reconnect_initial_backoff (float): Initial backoff delay in seconds for reconnection attempts
        reconnect_max_backoff (float): Maximum backoff delay in seconds for reconnection attempts
    """

    def __init__(self,
                 transport: str = 'streamable-http',
                 auth_provider: AuthProviderBase | None = None,
                 user_id: str | None = None,
                 tool_call_timeout: timedelta = timedelta(seconds=60),
                 auth_flow_timeout: timedelta = timedelta(seconds=300),
                 reconnect_enabled: bool = True,
                 reconnect_max_attempts: int = 2,
                 reconnect_initial_backoff: float = 0.5,
                 reconnect_max_backoff: float = 50.0):
        self._tools = None
        self._transport = transport.lower()
        if self._transport not in ['sse', 'stdio', 'streamable-http']:
            raise ValueError("transport must be either 'sse', 'stdio' or 'streamable-http'")

        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None  # Main session
        self._connection_established = False
        self._initial_connection = False

        # Convert auth provider to AuthAdapter
        self._auth_provider = auth_provider
        # Use provided user_id or fall back to auth provider's default_user_id
        effective_user_id = user_id or (auth_provider.config.default_user_id if auth_provider else None)
        self._httpx_auth = AuthAdapter(auth_provider, effective_user_id) if auth_provider else None

        self._tool_call_timeout = tool_call_timeout
        self._auth_flow_timeout = auth_flow_timeout

        # Reconnect configuration
        self._reconnect_enabled = reconnect_enabled
        self._reconnect_max_attempts = reconnect_max_attempts
        self._reconnect_initial_backoff = reconnect_initial_backoff
        self._reconnect_max_backoff = reconnect_max_backoff
        self._reconnect_lock: asyncio.Lock = asyncio.Lock()

    @property
    def auth_provider(self) -> AuthProviderBase | None:
        return self._auth_provider

    @property
    def transport(self) -> str:
        return self._transport

    async def __aenter__(self):
        if self._exit_stack:
            raise RuntimeError("MCPBaseClient already initialized. Use async with to initialize.")

        self._exit_stack = AsyncExitStack()

        # Establish connection with httpx.Auth
        self._session = await self._exit_stack.enter_async_context(self.connect_to_server())

        self._initial_connection = True
        self._connection_established = True

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._exit_stack:
            # Close session
            await self._exit_stack.aclose()
            self._session = None
            self._exit_stack = None

        self._connection_established = False
        self._tools = None

    @property
    def server_name(self):
        """
        Provide server name for logging
        """
        return self._transport

    @abstractmethod
    @asynccontextmanager
    async def connect_to_server(self) -> AsyncGenerator[ClientSession, None]:
        """
        Establish a session with an MCP server within an async context
        """
        yield

    async def _reconnect(self):
        """
        Attempt to reconnect by tearing down and re-establishing the session.
        """
        async with self._reconnect_lock:
            backoff = self._reconnect_initial_backoff
            attempt = 0
            last_error: Exception | None = None

            while attempt in range(0, self._reconnect_max_attempts):
                attempt += 1
                try:
                    # Close the existing stack and ClientSession
                    if self._exit_stack:
                        await self._exit_stack.aclose()
                    # Create a fresh stack and session
                    self._exit_stack = AsyncExitStack()
                    self._session = await self._exit_stack.enter_async_context(self.connect_to_server())

                    self._connection_established = True
                    self._tools = None

                    logger.info("Reconnected to MCP server (%s) on attempt %d", self.server_name, attempt)
                    return

                except Exception as e:
                    last_error = e
                    logger.warning("Reconnect attempt %d failed for %s: %s", attempt, self.server_name, e)
                    await asyncio.sleep(min(backoff, self._reconnect_max_backoff))
                    backoff = min(backoff * 2, self._reconnect_max_backoff)

            # All attempts failed
            self._connection_established = False
            if last_error:
                raise last_error

    async def _with_reconnect(self, coro):
        """
        Execute an awaited operation, reconnecting once on errors.
        Does not reconnect if the error occurs during an active authentication flow.
        """
        try:
            return await coro()
        except Exception as e:
            # Check if error happened during active authentication flow
            if self._httpx_auth and self._httpx_auth.is_authenticating:
                # Provide specific error message for authentication timeouts
                if isinstance(e, TimeoutError):
                    logger.error("Timeout during user authentication flow - user may have abandoned authentication")
                    raise RuntimeError(
                        "Authentication timed out. User did not complete authentication in browser within "
                        f"{self._auth_flow_timeout.total_seconds()} seconds.") from e
                else:
                    logger.error("Error during authentication flow: %s", e)
                    raise

            # Normal error - attempt reconnect if enabled
            if self._reconnect_enabled:
                try:
                    await self._reconnect()
                except Exception as reconnect_err:
                    logger.error("MCP Client reconnect attempt failed: %s", reconnect_err)
                    raise
                return await coro()
            raise

    async def _has_cached_auth_token(self) -> bool:
        """
        Check if we have a cached, non-expired authentication token.

        Returns:
            bool: True if we have a valid cached token, False if authentication may be needed
        """
        if not self._auth_provider:
            return True  # No auth needed

        try:
            # Check if OAuth2 provider has tokens cached
            if hasattr(self._auth_provider, '_auth_code_provider'):
                provider = self._auth_provider._auth_code_provider
                if provider and hasattr(provider, '_authenticated_tokens'):
                    # Check if we have at least one non-expired token
                    for auth_result in provider._authenticated_tokens.values():
                        if not auth_result.is_expired():
                            return True

            return False
        except Exception:
            # If we can't check, assume we need auth to be safe
            return False

    async def _get_tool_call_timeout(self) -> timedelta:
        """
        Determine the appropriate timeout for a tool call based on authentication state.

        Returns:
            timedelta: auth_flow_timeout if authentication may be needed, tool_call_timeout otherwise
        """
        if self._auth_provider:
            has_token = await self._has_cached_auth_token()
            timeout = self._tool_call_timeout if has_token else self._auth_flow_timeout
            if not has_token:
                logger.debug("Using extended timeout (%s) for potential interactive authentication", timeout)
            return timeout
        else:
            return self._tool_call_timeout

    @mcp_exception_handler
    async def get_tools(self) -> dict[str, MCPToolClient]:
        """
        Retrieve a dictionary of all tools served by the MCP server.
        Uses unauthenticated session for discovery.
        """

        async def _get_tools():
            session = self._session
            try:
                # Add timeout to the list_tools call.
                # This is needed because MCP SDK does not support timeout for list_tools()
                with anyio.fail_after(self._tool_call_timeout.total_seconds()):
                    tools = await session.list_tools()
            except TimeoutError as e:
                from nat.plugins.mcp.exceptions import MCPTimeoutError
                raise MCPTimeoutError(self.server_name, e)

            return tools

        try:
            response = await self._with_reconnect(_get_tools)
        except Exception as e:
            logger.warning("Failed to get tools: %s", e)
            raise

        return {
            tool.name:
                MCPToolClient(session=self._session,
                              tool_name=tool.name,
                              tool_description=tool.description,
                              tool_input_schema=tool.inputSchema,
                              parent_client=self)
            for tool in response.tools
        }

    @mcp_exception_handler
    async def get_tool(self, tool_name: str) -> MCPToolClient:
        """
        Get an MCP Tool by name.

        Args:
            tool_name (str): Name of the tool to load.

        Returns:
            MCPToolClient for the configured tool.

        Raises:
            MCPToolNotFoundError: If no tool is available with that name.
        """
        if not self._exit_stack:
            raise RuntimeError("MCPBaseClient not initialized. Use async with to initialize.")

        if not self._tools:
            self._tools = await self.get_tools()

        tool = self._tools.get(tool_name)
        if not tool:
            raise MCPToolNotFoundError(tool_name, self.server_name)
        return tool

    def set_user_auth_callback(self, auth_callback: Callable[[AuthFlowType], AuthenticatedContext]):
        """Set the user authentication callback."""
        if self._auth_provider and hasattr(self._auth_provider, "_set_custom_auth_callback"):
            self._auth_provider._set_custom_auth_callback(auth_callback)

    @mcp_exception_handler
    async def call_tool(self, tool_name: str, tool_args: dict | None):

        async def _call_tool():
            session = self._session
            timeout = await self._get_tool_call_timeout()
            return await session.call_tool(tool_name, tool_args, read_timeout_seconds=timeout)

        return await self._with_reconnect(_call_tool)


class MCPSSEClient(MCPBaseClient):
    """
    Client for creating a session and connecting to an MCP server using SSE

    Args:
      url (str): The url of the MCP server
    """

    def __init__(self,
                 url: str,
                 tool_call_timeout: timedelta = timedelta(seconds=60),
                 auth_flow_timeout: timedelta = timedelta(seconds=300),
                 reconnect_enabled: bool = True,
                 reconnect_max_attempts: int = 2,
                 reconnect_initial_backoff: float = 0.5,
                 reconnect_max_backoff: float = 50.0):
        super().__init__("sse",
                         tool_call_timeout=tool_call_timeout,
                         auth_flow_timeout=auth_flow_timeout,
                         reconnect_enabled=reconnect_enabled,
                         reconnect_max_attempts=reconnect_max_attempts,
                         reconnect_initial_backoff=reconnect_initial_backoff,
                         reconnect_max_backoff=reconnect_max_backoff)
        self._url = url

    @property
    def url(self) -> str:
        return self._url

    @property
    def server_name(self):
        return f"sse:{self._url}"

    @asynccontextmanager
    @override
    async def connect_to_server(self):
        """
        Establish a session with an MCP SSE server within an async context
        """
        async with sse_client(url=self._url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


class MCPStdioClient(MCPBaseClient):
    """
    Client for creating a session and connecting to an MCP server using stdio.
    This is a local transport that spawns the MCP server process and communicates
    with it over stdin/stdout.

    Args:
      command (str): The command to run
      args (list[str] | None): Additional arguments for the command
      env (dict[str, str] | None): Environment variables to set for the process
    """

    def __init__(self,
                 command: str,
                 args: list[str] | None = None,
                 env: dict[str, str] | None = None,
                 tool_call_timeout: timedelta = timedelta(seconds=60),
                 auth_flow_timeout: timedelta = timedelta(seconds=300),
                 reconnect_enabled: bool = True,
                 reconnect_max_attempts: int = 2,
                 reconnect_initial_backoff: float = 0.5,
                 reconnect_max_backoff: float = 50.0):
        super().__init__("stdio",
                         tool_call_timeout=tool_call_timeout,
                         auth_flow_timeout=auth_flow_timeout,
                         reconnect_enabled=reconnect_enabled,
                         reconnect_max_attempts=reconnect_max_attempts,
                         reconnect_initial_backoff=reconnect_initial_backoff,
                         reconnect_max_backoff=reconnect_max_backoff)
        self._command = command
        self._args = args
        self._env = env

    @property
    def command(self) -> str:
        return self._command

    @property
    def server_name(self):
        return f"stdio:{self._command}"

    @property
    def args(self) -> list[str] | None:
        return self._args

    @property
    def env(self) -> dict[str, str] | None:
        return self._env

    @asynccontextmanager
    @override
    async def connect_to_server(self):
        """
        Establish a session with an MCP server via stdio within an async context
        """

        server_params = StdioServerParameters(command=self._command, args=self._args or [], env=self._env)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


class MCPStreamableHTTPClient(MCPBaseClient):
    """
    Client for creating a session and connecting to an MCP server using streamable-http

    Args:
      url (str): The url of the MCP server
      auth_provider (AuthProviderBase | None): Optional authentication provider for Bearer token injection
    """

    def __init__(self,
                 url: str,
                 auth_provider: AuthProviderBase | None = None,
                 user_id: str | None = None,
                 tool_call_timeout: timedelta = timedelta(seconds=60),
                 auth_flow_timeout: timedelta = timedelta(seconds=300),
                 reconnect_enabled: bool = True,
                 reconnect_max_attempts: int = 2,
                 reconnect_initial_backoff: float = 0.5,
                 reconnect_max_backoff: float = 50.0):
        super().__init__("streamable-http",
                         auth_provider=auth_provider,
                         user_id=user_id,
                         tool_call_timeout=tool_call_timeout,
                         auth_flow_timeout=auth_flow_timeout,
                         reconnect_enabled=reconnect_enabled,
                         reconnect_max_attempts=reconnect_max_attempts,
                         reconnect_initial_backoff=reconnect_initial_backoff,
                         reconnect_max_backoff=reconnect_max_backoff)
        self._url = url

    @property
    def url(self) -> str:
        return self._url

    @property
    def server_name(self):
        return f"streamable-http:{self._url}"

    @asynccontextmanager
    @override
    async def connect_to_server(self):
        """
        Establish a session with an MCP server via streamable-http within an async context
        """
        # Use httpx.Auth for authentication
        async with streamablehttp_client(url=self._url, auth=self._httpx_auth) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


class MCPToolClient:
    """
    Client wrapper used to call an MCP tool. This assumes that the MCP transport session
    has already been setup.

    Args:
        session (ClientSession): The MCP client session
        tool_name (str): The name of the tool to wrap
        tool_description (str): The description of the tool provided by the MCP server.
        tool_input_schema (dict): The input schema for the tool.
        parent_client (MCPBaseClient): The parent MCP client for auth management.
    """

    def __init__(self,
                 session: ClientSession,
                 parent_client: MCPBaseClient,
                 tool_name: str,
                 tool_description: str | None,
                 tool_input_schema: dict | None = None):
        self._session = session
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._input_schema = (model_from_mcp_schema(self._tool_name, tool_input_schema) if tool_input_schema else None)
        self._parent_client = parent_client

        if self._parent_client is None:
            raise RuntimeError("MCPToolClient initialized without a parent client.")

    @property
    def name(self):
        """Returns the name of the tool."""
        return self._tool_name

    @property
    def description(self):
        """
        Returns the tool's description. If none was provided. Provides a simple description using the tool's name
        """
        if not self._tool_description:
            return f"MCP Tool {self._tool_name}"
        return self._tool_description

    @property
    def input_schema(self):
        """
        Returns the tool's input_schema.
        """
        return self._input_schema

    def set_description(self, description: str):
        """
        Manually define the tool's description using the provided string.
        """
        self._tool_description = description

    async def acall(self, tool_args: dict) -> str:
        """
        Call the MCP tool with the provided arguments.
        Session context is now handled at the client level, eliminating the need for metadata injection.

        Args:
            tool_args (dict[str, Any]): A dictionary of key value pairs to serve as inputs for the MCP tool.
        """
        if self._session is None:
            raise RuntimeError("No session available for tool call")

        try:
            # Simple tool call - session context is already in the client instance
            logger.info("Calling tool %s", self._tool_name)
            result = await self._parent_client.call_tool(self._tool_name, tool_args)

            output = []
            for res in result.content:
                if isinstance(res, TextContent):
                    output.append(res.text)
                else:
                    # Log non-text content for now
                    logger.warning("Got not-text output from %s of type %s", self.name, type(res))
            result_str = "\n".join(output)

            if result.isError:
                mcp_error: MCPError = convert_to_mcp_error(RuntimeError(result_str), self._parent_client.server_name)
                raise mcp_error

        except MCPError as e:
            format_mcp_error(e, include_traceback=False)
            result_str = f"MCPToolClient tool call failed: {e.original_exception}"

        return result_str
