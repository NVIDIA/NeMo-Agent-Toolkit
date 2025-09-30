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
import json
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

    def __init__(self, auth_provider: AuthProviderBase):
        self.auth_provider = auth_provider
        # each adapter instance has its own lock to avoid unnecessary delays for multiple clients
        self._lock = anyio.Lock()

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
                    auth_headers = await self._get_auth_headers(request=request, response=response)
                    request.headers.update(auth_headers)
                    yield request  # Retry the request
                except Exception as e:
                    logger.info("Failed to refresh auth after 401: %s", e)
        return

    def _get_session_id_from_tool_call_request(self, request: httpx.Request) -> tuple[str | None, bool]:
        """Check if this is a tool call request based on the request body.
        Return the session id if it exists and a boolean indicating if it is a tool call request
        """
        try:
            # Check if the request body contains a tool call
            if request.content:
                body = json.loads(request.content.decode('utf-8'))
                # Check if it's a JSON-RPC request with method "tools/call"
                if (isinstance(body, dict) and body.get("method") == "tools/call"):
                    session_id = body.get("params").get("_meta").get("session_id")
                    return session_id, True
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            # If we can't parse the body, assume it's not a tool call
            pass
        return None, False

    async def _get_auth_headers(self,
                                request: httpx.Request | None = None,
                                response: httpx.Response | None = None) -> dict[str, str]:
        """Get authentication headers from the NAT auth provider."""
        try:
            session_id = None
            is_tool_call = False
            if request:
                session_id, is_tool_call = self._get_session_id_from_tool_call_request(request)

            if is_tool_call:
                # Tool call requests should use the session id
                user_id = session_id
            else:
                # Non-tool call requests should use the session id if it exists and fallback to default user id
                user_id = session_id or self.auth_provider.config.default_user_id

            auth_result = await self.auth_provider.authenticate(user_id=user_id, response=response)

            # Check if we have BearerTokenCred
            from nat.data_models.authentication import BearerTokenCred
            if auth_result.credentials and isinstance(auth_result.credentials[0], BearerTokenCred):
                token = auth_result.credentials[0].token.get_secret_value()
                return {"Authorization": f"Bearer {token}"}
            else:
                logger.info("Auth provider did not return BearerTokenCred")
                return {}
        except Exception as e:
            logger.warning("Failed to get auth token: %s", e)
            return {}


class MCPBaseClient(ABC):
    """
    Base client for creating a MCP transport session and connecting to an MCP server

    Args:
        transport (str): The type of client to use ('sse', 'stdio', or 'streamable-http')
        auth_provider (AuthProviderBase | None): Optional authentication provider for Bearer token injection
    """

    def __init__(self,
                 transport: str = 'streamable-http',
                 auth_provider: AuthProviderBase | None = None,
                 tool_call_timeout: timedelta = timedelta(seconds=5),
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
        self._httpx_auth = AuthAdapter(auth_provider) if auth_provider else None

        self._tool_call_timeout = tool_call_timeout

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
        """
        try:
            return await coro()
        except Exception as e:
            if self._reconnect_enabled:
                logger.warning("MCP Client operation failed. Attempting reconnect: %s", e)
                try:
                    await self._reconnect()
                except Exception as reconnect_err:
                    logger.error("MCP Client reconnect attempt failed: %s", reconnect_err)
                    raise
                return await coro()
            raise

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
                              parent_client=self,
                              tool_call_timeout=self._tool_call_timeout)
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
    async def call_tool_with_meta(self, tool_name: str, args: dict, session_id: str):
        from mcp.types import CallToolRequest
        from mcp.types import CallToolRequestParams
        from mcp.types import CallToolResult
        from mcp.types import ClientRequest

        if not self._session:
            raise RuntimeError("MCPBaseClient not initialized. Use async with to initialize.")

        async def _call_tool_with_meta():
            params = CallToolRequestParams(name=tool_name, arguments=args, **{"_meta": {"session_id": session_id}})
            req = ClientRequest(CallToolRequest(params=params))
            # We will increase the timeout to 5 minutes if the tool call timeout is less than 5 min and
            # auth is enabled.
            if self._auth_provider and self._tool_call_timeout.total_seconds() < 300:
                timeout = timedelta(seconds=300)
            else:
                timeout = self._tool_call_timeout
            return await self._session.send_request(req, CallToolResult, request_read_timeout_seconds=timeout)

        return await self._with_reconnect(_call_tool_with_meta)

    @mcp_exception_handler
    async def call_tool(self, tool_name: str, tool_args: dict | None):

        async def _call_tool():
            session = self._session
            return await session.call_tool(tool_name, tool_args, read_timeout_seconds=self._tool_call_timeout)

        return await self._with_reconnect(_call_tool)


class MCPSSEClient(MCPBaseClient):
    """
    Client for creating a session and connecting to an MCP server using SSE

    Args:
      url (str): The url of the MCP server
    """

    def __init__(self,
                 url: str,
                 tool_call_timeout: timedelta = timedelta(seconds=5),
                 reconnect_enabled: bool = True,
                 reconnect_max_attempts: int = 2,
                 reconnect_initial_backoff: float = 0.5,
                 reconnect_max_backoff: float = 50.0):
        super().__init__("sse",
                         tool_call_timeout=tool_call_timeout,
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
                 tool_call_timeout: timedelta = timedelta(seconds=5),
                 reconnect_enabled: bool = True,
                 reconnect_max_attempts: int = 2,
                 reconnect_initial_backoff: float = 0.5,
                 reconnect_max_backoff: float = 50.0):
        super().__init__("stdio",
                         tool_call_timeout=tool_call_timeout,
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
                 tool_call_timeout: timedelta = timedelta(seconds=5),
                 reconnect_enabled: bool = True,
                 reconnect_max_attempts: int = 2,
                 reconnect_initial_backoff: float = 0.5,
                 reconnect_max_backoff: float = 50.0):
        super().__init__("streamable-http",
                         auth_provider=auth_provider,
                         tool_call_timeout=tool_call_timeout,
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
                 tool_input_schema: dict | None = None,
                 tool_call_timeout: timedelta = timedelta(seconds=5)):
        self._session = session
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._input_schema = (model_from_mcp_schema(self._tool_name, tool_input_schema) if tool_input_schema else None)
        self._parent_client = parent_client
        self._tool_call_timeout = tool_call_timeout

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

    def _get_session_id(self) -> str | None:
        """
        Get the session id from the context.
        """
        from nat.builder.context import Context as _Ctx

        # get auth callback (for example: WebSocketAuthenticationFlowHandler). this is lazily set in the client
        # on first tool call
        auth_callback = _Ctx.get().user_auth_callback
        if auth_callback and self._parent_client:
            # set custom auth callback
            self._parent_client.set_user_auth_callback(auth_callback)

        # get session id from context, authentication is done per-websocket session for tool calls
        session_id = None
        cookies = getattr(_Ctx.get().metadata, "cookies", None)
        if cookies:
            session_id = cookies.get("nat-session")

        if not session_id:
            # use default user id if allowed
            if self._parent_client.auth_provider and \
                self._parent_client.auth_provider.config.allow_default_user_id_for_tool_calls:
                session_id = self._parent_client.auth_provider.config.default_user_id
        return session_id

    async def acall(self, tool_args: dict) -> str:
        """
        Call the MCP tool with the provided arguments.

        Args:
            tool_args (dict[str, Any]): A dictionary of key value pairs to serve as inputs for the MCP tool.
        """
        if self._session is None:
            raise RuntimeError("No session available for tool call")

        # Extract context information
        try:
            session_id = self._get_session_id()
        except Exception:
            session_id = None

        try:
            # if auth is enabled and session id is not available return user is not authorized to call the tool
            if self._parent_client.auth_provider and not session_id:
                result_str = "User is not authorized to call the tool"
                mcp_error: MCPError = convert_to_mcp_error(RuntimeError(result_str), self._parent_client.server_name)
                raise mcp_error

            if session_id:
                logger.info("Calling tool %s with arguments %s for a user session", self._tool_name, tool_args)
                result = await self._parent_client.call_tool_with_meta(self._tool_name, tool_args, session_id)
            else:
                logger.info("Calling tool %s with arguments %s", self._tool_name, tool_args)
                result = await self._session.call_tool(self._tool_name, tool_args)

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
