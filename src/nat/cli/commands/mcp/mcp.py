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
import json
import logging
import time
from typing import Any
from typing import Literal
from typing import cast

import click
from pydantic import BaseModel

from nat.cli.commands.start import start_command

logger = logging.getLogger(__name__)


@click.group(name=__name__, invoke_without_command=False, help="MCP-related commands.")
def mcp_command():
    """
    MCP-related commands.
    """
    return None


# nat mcp serve: reuses the start/mcp frontend command
mcp_command.add_command(start_command.get_command(None, "mcp"), name="serve")  # type: ignore

# Suppress verbose logs from mcp.client.sse and httpx
logging.getLogger("mcp.client.sse").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

try:
    from nat.plugins.mcp.exception_handler import format_mcp_error
    from nat.plugins.mcp.exceptions import MCPError
except ImportError:
    # Fallback for when MCP client package is not installed
    MCPError = Exception

    def format_mcp_error(error, include_traceback=False):
        click.echo(f"Error: {error}", err=True)


def validate_transport_cli_args(transport: str, command: str | None, args: str | None, env: str | None) -> bool:
    """
    Validate transport and parameter combinations, returning False if invalid.

    Args:
        transport: The transport type ('sse', 'stdio', or 'streamable-http')
        command: Command for stdio transport
        args: Arguments for stdio transport
        env: Environment variables for stdio transport

    Returns:
        bool: True if valid, False if invalid (error message already displayed)
    """
    if transport == 'stdio':
        if not command:
            click.echo("--command is required when using stdio client type", err=True)
            return False
    elif transport in ['sse', 'streamable-http']:
        if command or args or env:
            click.echo("--command, --args, and --env are not allowed when using sse or streamable-http client type",
                       err=True)
            return False
    return True


class MCPPingResult(BaseModel):
    """Result of an MCP server ping request.

    Attributes:
        url (str): The MCP server URL that was pinged
        status (str): Health status - 'healthy', 'unhealthy', or 'unknown'
        response_time_ms (float | None): Response time in milliseconds, None if request failed completely
        error (str | None): Error message if the ping failed, None if successful
    """
    url: str
    status: str
    response_time_ms: float | None
    error: str | None


def format_tool(tool: Any) -> dict[str, str | None]:
    """Format an MCP tool into a dictionary for display.

    Extracts name, description, and input schema from various MCP tool object types
    and normalizes them into a consistent dictionary format for CLI display.

    Args:
        tool (Any): MCPToolClient or raw MCP Tool object (uses Any due to different types)

    Returns:
        dict[str, str | None]: Dictionary with name, description, and input_schema as keys
    """
    name = getattr(tool, 'name', None)
    description = getattr(tool, 'description', '')
    input_schema = getattr(tool, 'input_schema', None) or getattr(tool, 'inputSchema', None)

    # Normalize schema to JSON string
    if input_schema is None:
        return {
            "name": name,
            "description": description,
            "input_schema": None,
        }
    elif hasattr(input_schema, "schema_json"):
        schema_str = input_schema.schema_json(indent=2)
    elif hasattr(input_schema, "model_json_schema"):
        schema_str = json.dumps(input_schema.model_json_schema(), indent=2)
    elif isinstance(input_schema, dict):
        schema_str = json.dumps(input_schema, indent=2)
    else:
        # Final fallback: attempt to dump stringified version wrapped as JSON string
        schema_str = json.dumps({"raw": str(input_schema)}, indent=2)

    return {
        "name": name,
        "description": description,
        "input_schema": schema_str,
    }


def print_tool(tool_dict: dict[str, str | None], detail: bool = False) -> None:
    """Print a formatted tool to the console with optional detailed information.

    Outputs tool information in a user-friendly format to stdout. When detail=True
    or when description/schema are available, shows full information with separator.

    Args:
        tool_dict (dict[str, str | None]): Dictionary containing tool information with name, description, and
        input_schema as keys
        detail (bool, optional): Whether to force detailed output. Defaults to False.
    """
    click.echo(f"Tool: {tool_dict.get('name', 'Unknown')}")
    if detail or tool_dict.get('input_schema') or tool_dict.get('description'):
        click.echo(f"Description: {tool_dict.get('description', 'No description available')}")
        if tool_dict.get("input_schema"):
            click.echo("Input Schema:")
            click.echo(tool_dict.get("input_schema"))
        else:
            click.echo("Input Schema: None")
        click.echo("-" * 60)


def _set_auth_defaults(auth: bool,
                       url: str | None,
                       auth_redirect_uri: str | None,
                       auth_user_id: str | None,
                       auth_scopes: str | None) -> tuple[str | None, str | None, list[str] | None]:
    """Set default auth values when --auth flag is used.

    Args:
        auth: Whether --auth flag was used
        url: MCP server URL
        auth_redirect_uri: OAuth2 redirect URI
        auth_user_id: User ID for authentication
        auth_scopes: OAuth2 scopes (comma-separated string)

    Returns:
        Tuple of (auth_redirect_uri, auth_user_id, auth_scopes_list) with defaults applied
    """
    if auth:
        auth_redirect_uri = auth_redirect_uri or "http://localhost:8000/auth/redirect"
        auth_user_id = auth_user_id or url
        auth_scopes = auth_scopes or ""

    # Convert comma-separated string to list, stripping whitespace
    auth_scopes_list = [scope.strip() for scope in auth_scopes.split(',')] if auth_scopes else None

    return auth_redirect_uri, auth_user_id, auth_scopes_list


async def _create_mcp_client_config(
    builder,
    server_cfg,
    url: str | None,
    transport: str,
    auth_redirect_uri: str | None,
    auth_user_id: str | None,
    auth_scopes: list[str] | None,
):
    from nat.plugins.mcp.client_config import MCPClientConfig

    if url and transport == "streamable-http" and auth_redirect_uri:
        try:
            from nat.plugins.mcp.auth.auth_provider_config import MCPOAuth2ProviderConfig
            auth_config = MCPOAuth2ProviderConfig(
                server_url=url,
                redirect_uri=auth_redirect_uri,
                default_user_id=auth_user_id or url,
                scopes=auth_scopes or [],
            )
            auth_provider_name = "mcp_oauth2_cli"
            await builder.add_auth_provider(auth_provider_name, auth_config)
            server_cfg.auth_provider = auth_provider_name
        except ImportError:
            click.echo(
                "[WARNING] MCP OAuth2 authentication requires nvidia-nat-mcp package.",
                err=True,
            )

    return MCPClientConfig(server=server_cfg)


async def list_tools_via_function_group(
    command: str | None,
    url: str | None,
    tool_name: str | None = None,
    transport: str = 'sse',
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    auth_redirect_uri: str | None = None,
    auth_user_id: str | None = None,
    auth_scopes: list[str] | None = None,
) -> list[dict[str, str | None]]:
    """List tools by constructing the mcp_client function group and introspecting functions.

    Mirrors the behavior of list_mcp.py but routes through the registered function group to ensure
    parity with workflow configuration.
    """
    try:
        # Ensure the registration side-effects are loaded
        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.plugins.mcp.client_config import MCPClientConfig
        from nat.plugins.mcp.client_config import MCPServerConfig
    except ImportError:
        click.echo(
            "MCP client functionality requires nvidia-nat-mcp package. Install with: uv pip install nvidia-nat-mcp",
            err=True)
        return []

    if args is None:
        args = []

    # Build server config according to transport
    server_cfg = MCPServerConfig(
        transport=cast(Literal["stdio", "sse", "streamable-http"], transport),
        url=cast(Any, url) if transport in ('sse', 'streamable-http') else None,
        command=command if transport == 'stdio' else None,
        args=args if transport == 'stdio' else None,
        env=env if transport == 'stdio' else None,
    )

    group_cfg = MCPClientConfig(server=server_cfg)

    tools: list[dict[str, str | None]] = []

    async with WorkflowBuilder() as builder:  # type: ignore
        # Add auth provider if url is provided and auth_redirect_uri is given (only for streamable-http)
        group_cfg = await _create_mcp_client_config(builder,
                                                    server_cfg,
                                                    url,
                                                    transport,
                                                    auth_redirect_uri,
                                                    auth_user_id,
                                                    auth_scopes)
        group = await builder.add_function_group("mcp_client", group_cfg)

        # Access functions exposed by the group
        fns = await group.get_accessible_functions()

        def to_tool_entry(full_name: str, fn_obj) -> dict[str, str | None]:
            # full_name like "mcp_client.<tool>"
            name = full_name.split(".", 1)[1] if "." in full_name else full_name
            schema = getattr(fn_obj, 'input_schema', None)
            if schema is None:
                schema_str = None
            elif hasattr(schema, "schema_json"):
                schema_str = schema.schema_json(indent=2)
            elif hasattr(schema, "model_json_schema"):
                try:
                    schema_str = json.dumps(schema.model_json_schema(), indent=2)
                except Exception:
                    schema_str = None
            else:
                schema_str = None
            return {"name": name, "description": getattr(fn_obj, 'description', ''), "input_schema": schema_str}

        if tool_name:
            full = f"mcp_client.{tool_name}"
            fn = fns.get(full)
            if fn is not None:
                tools.append(to_tool_entry(full, fn))
        else:
            for full, fn in fns.items():
                tools.append(to_tool_entry(full, fn))

    return tools


async def list_tools_direct(command, url, tool_name=None, transport='sse', args=None, env=None):
    """List MCP tools using direct MCP protocol with structured exception handling.

    Bypasses MCPBuilder and uses raw MCP ClientSession and SSE client directly.
    Converts raw exceptions to structured MCPErrors for consistent user experience.
    Used when --direct flag is specified in CLI.

    Args:
        url (str): MCP server URL to connect to
        tool_name (str | None, optional): Specific tool name to retrieve.
        If None, retrieves all available tools. Defaults to None.

    Returns:
        list[dict[str, str | None]]: List of formatted tool dictionaries, each containing name, description, and
        input_schema as keys

    Note:
        This function handles ExceptionGroup by extracting the most relevant exception
        and converting it to MCPError for consistent error reporting.
    """
    if args is None:
        args = []
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    try:
        if transport == 'stdio':

            def get_stdio_client():
                return stdio_client(server=StdioServerParameters(command=command, args=args, env=env))

            client = get_stdio_client
        elif transport == 'streamable-http':

            def get_streamable_http_client():
                return streamablehttp_client(url=url)

            client = get_streamable_http_client
        else:

            def get_sse_client():
                return sse_client(url=url)

            client = get_sse_client

        async with client() as ctx:
            read, write = (ctx[0], ctx[1]) if isinstance(ctx, tuple) else ctx
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()

        tools = []
        for tool in response.tools:
            if tool_name:
                if tool.name == tool_name:
                    tools.append(format_tool(tool))
            else:
                tools.append(format_tool(tool))

        if tool_name and not tools:
            click.echo(f"[INFO] Tool '{tool_name}' not found.")
        return tools
    except Exception as e:
        # Convert raw exceptions to structured MCPError for consistency
        try:
            from nat.plugins.mcp.exception_handler import convert_to_mcp_error
            from nat.plugins.mcp.exception_handler import extract_primary_exception
        except ImportError:
            # Fallback when MCP client package is not installed
            def convert_to_mcp_error(exception, url):
                return Exception(f"Error connecting to {url}: {exception}")

            def extract_primary_exception(exceptions):
                return exceptions[0] if exceptions else Exception("Unknown error")

        if isinstance(e, ExceptionGroup):
            primary_exception = extract_primary_exception(list(e.exceptions))
            mcp_error = convert_to_mcp_error(primary_exception, url)
        else:
            mcp_error = convert_to_mcp_error(e, url)

        format_mcp_error(mcp_error, include_traceback=False)
        return []


async def ping_mcp_server(url: str,
                          timeout: int,
                          transport: str = 'streamable-http',
                          command: str | None = None,
                          args: list[str] | None = None,
                          env: dict[str, str] | None = None,
                          auth_redirect_uri: str | None = None,
                          auth_user_id: str | None = None,
                          auth_scopes: list[str] | None = None) -> MCPPingResult:
    """Ping an MCP server to check if it's responsive.

    Args:
        url (str): MCP server URL to ping
        timeout (int): Timeout in seconds for the ping request

    Returns:
        MCPPingResult: Structured result with status, response_time, and any error info
    """
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    async def _ping_operation():
        # Select transport
        if transport == 'stdio':
            stdio_args_local: list[str] = args or []
            if not command:
                raise RuntimeError("--command is required for stdio transport")
            client_ctx = stdio_client(server=StdioServerParameters(command=command, args=stdio_args_local, env=env))
        elif transport == 'sse':
            client_ctx = sse_client(url)
        else:  # streamable-http
            client_ctx = streamablehttp_client(url=url)

        async with client_ctx as ctx:
            read, write = (ctx[0], ctx[1]) if isinstance(ctx, tuple) else ctx
            async with ClientSession(read, write) as session:
                await session.initialize()

                start_time = time.time()
                await session.send_ping()
                end_time = time.time()
                response_time_ms = round((end_time - start_time) * 1000, 2)

                return MCPPingResult(url=url, status="healthy", response_time_ms=response_time_ms, error=None)

    try:
        # Apply timeout to the entire ping operation
        return await asyncio.wait_for(_ping_operation(), timeout=timeout)

    except TimeoutError:
        return MCPPingResult(url=url,
                             status="unhealthy",
                             response_time_ms=None,
                             error=f"Timeout after {timeout} seconds")

    except Exception as e:
        return MCPPingResult(url=url, status="unhealthy", response_time_ms=None, error=str(e))


@mcp_command.group(name="client", invoke_without_command=False, help="MCP client commands.")
def mcp_client_command():
    """
    MCP client commands.
    """
    try:
        from nat.runtime.loader import PluginTypes
        from nat.runtime.loader import discover_and_register_plugins
        discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)
    except ImportError:
        click.echo("[WARNING] MCP client functionality requires nvidia-nat-mcp package.", err=True)
        pass


@mcp_client_command.group(name="tool", invoke_without_command=False, help="Inspect and call MCP tools.")
def mcp_client_tool_group():
    """
    MCP client tool commands.
    """
    return None


@mcp_client_tool_group.command(name="list", help="List tool names (default), or show details with --detail or --tool.")
@click.option('--direct', is_flag=True, help='Bypass MCPBuilder and use direct MCP protocol')
@click.option(
    '--url',
    default='http://localhost:9901/mcp',
    show_default=True,
    help='MCP server URL (e.g. http://localhost:8080/mcp for streamable-http, http://localhost:8080/sse for sse)')
@click.option('--transport',
              type=click.Choice(['sse', 'stdio', 'streamable-http']),
              default='streamable-http',
              show_default=True,
              help='Type of client to use (default: streamable-http, backwards compatible with sse)')
@click.option('--command', help='For stdio: The command to run (e.g. mcp-server)')
@click.option('--args', help='For stdio: Additional arguments for the command (space-separated)')
@click.option('--env', help='For stdio: Environment variables in KEY=VALUE format (space-separated)')
@click.option('--tool', default=None, help='Get details for a specific tool by name')
@click.option('--detail', is_flag=True, help='Show full details for all tools')
@click.option('--json-output', is_flag=True, help='Output tool metadata in JSON format')
@click.option('--auth',
              is_flag=True,
              help='Enable OAuth2 authentication with default settings (streamable-http only, not with --direct)')
@click.option('--auth-redirect-uri',
              help='OAuth2 redirect URI for authentication (streamable-http only, not with --direct)')
@click.option('--auth-user-id', help='User ID for authentication (streamable-http only, not with --direct)')
@click.option('--auth-scopes', help='OAuth2 scopes (comma-separated, streamable-http only, not with --direct)')
@click.pass_context
def mcp_client_tool_list(ctx,
                         direct,
                         url,
                         transport,
                         command,
                         args,
                         env,
                         tool,
                         detail,
                         json_output,
                         auth,
                         auth_redirect_uri,
                         auth_user_id,
                         auth_scopes):
    """List MCP tool names (default) or show detailed tool information.

    Use --detail for full output including descriptions and input schemas.
    If --tool is provided, always shows full output for that specific tool.
    Use --direct to bypass MCPBuilder and use raw MCP protocol.
    Use --json-output to get structured JSON data instead of formatted text.
    Use --auth to enable auth with default settings (streamable-http only, not with --direct).
    Use --auth-redirect-uri to enable auth for protected MCP servers (streamable-http only, not with --direct).

    Args:
        ctx (click.Context): Click context object for command invocation
        direct (bool): Whether to bypass MCPBuilder and use direct MCP protocol
        url (str): MCP server URL to connect to (default: http://localhost:9901/mcp)
        tool (str | None): Optional specific tool name to retrieve detailed info for
        detail (bool): Whether to show full details (description + schema) for all tools
        json_output (bool): Whether to output tool metadata in JSON format instead of text
        auth (bool): Whether to enable OAuth2 authentication (streamable-http only, not with --direct)
        auth_redirect_uri (str | None): redirect URI for auth (streamable-http only, not with --direct)
        auth_user_id (str | None): User ID for authentication (streamable-http only, not with --direct)
        auth_scopes (str | None): OAuth2 scopes (comma-separated, streamable-http only, not with --direct)

    Examples:
        nat mcp client tool list                           # List tool names only
        nat mcp client tool list --detail                  # Show all tools with full details
        nat mcp client tool list --tool my_tool            # Show details for specific tool
        nat mcp client tool list --json-output             # Get JSON format output
        nat mcp client tool list --direct --url http://... # Use direct protocol with custom URL (no auth)
        nat mcp client tool list --url https://example.com/mcp/ --auth # With auth using defaults
        nat mcp client tool list --url https://example.com/mcp/ --transport streamable-http \
            --auth-redirect-uri http://localhost:8000/auth/redirect # With custom auth settings
        nat mcp client tool list --url https://example.com/mcp/ --transport streamable-http \
            --auth-redirect-uri http://localhost:8000/auth/redirect --auth-user-id myuser # With auth and user ID
    """
    if ctx.invoked_subcommand is not None:
        return

    if not validate_transport_cli_args(transport, command, args, env):
        return

    if transport in ['sse', 'streamable-http']:
        if not url:
            click.echo("[ERROR] --url is required when using sse or streamable-http client type", err=True)
            return

    # Set auth defaults if --auth flag is used
    auth_redirect_uri, auth_user_id, auth_scopes_list = _set_auth_defaults(
        auth, url, auth_redirect_uri, auth_user_id, auth_scopes
    )

    stdio_args = args.split() if args else []
    stdio_env = dict(var.split('=', 1) for var in env.split()) if env else None

    if direct:
        tools = asyncio.run(
            list_tools_direct(command, url, tool_name=tool, transport=transport, args=stdio_args, env=stdio_env))
    else:
        tools = asyncio.run(
            list_tools_via_function_group(command,
                                          url,
                                          tool_name=tool,
                                          transport=transport,
                                          args=stdio_args,
                                          env=stdio_env,
                                          auth_redirect_uri=auth_redirect_uri,
                                          auth_user_id=auth_user_id,
                                          auth_scopes=auth_scopes_list))

    if json_output:
        click.echo(json.dumps(tools, indent=2))
    elif tool:
        for tool_dict in (tools or []):
            print_tool(tool_dict, detail=True)
    elif detail:
        for tool_dict in (tools or []):
            print_tool(tool_dict, detail=True)
    else:
        for tool_dict in (tools or []):
            click.echo(tool_dict.get('name', 'Unknown tool'))


@mcp_client_command.command(name="ping", help="Ping an MCP server to check if it's responsive.")
@click.option(
    '--url',
    default='http://localhost:9901/mcp',
    show_default=True,
    help='MCP server URL (e.g. http://localhost:8080/mcp for streamable-http, http://localhost:8080/sse for sse)')
@click.option('--transport',
              type=click.Choice(['sse', 'stdio', 'streamable-http']),
              default='streamable-http',
              show_default=True,
              help='Type of client to use for ping')
@click.option('--command', help='For stdio: The command to run (e.g. mcp-server)')
@click.option('--args', help='For stdio: Additional arguments for the command (space-separated)')
@click.option('--env', help='For stdio: Environment variables in KEY=VALUE format (space-separated)')
@click.option('--timeout', default=60, show_default=True, help='Timeout in seconds for ping request')
@click.option('--json-output', is_flag=True, help='Output ping result in JSON format')
@click.option('--auth-redirect-uri',
              help='OAuth2 redirect URI for authentication (streamable-http only, not with --direct)')
@click.option('--auth-user-id', help='User ID for authentication (streamable-http only, not with --direct)')
@click.option('--auth-scopes', help='OAuth2 scopes (comma-separated, streamable-http only, not with --direct)')
def mcp_client_ping(url: str,
                    transport: str,
                    command: str | None,
                    args: str | None,
                    env: str | None,
                    timeout: int,
                    json_output: bool,
                    auth_redirect_uri: str | None,
                    auth_user_id: str | None,
                    auth_scopes: str | None) -> None:
    """Ping an MCP server to check if it's responsive.

    This command sends a ping request to the MCP server and measures the response time.
    It's useful for health checks and monitoring server availability.

    Args:
        url (str): MCP server URL to ping (default: http://localhost:9901/mcp)
        timeout (int): Timeout in seconds for the ping request (default: 60)
        json_output (bool): Whether to output the result in JSON format
        auth_redirect_uri (str | None): redirect URI for auth (streamable-http only, not with --direct)
        auth_user_id (str | None): User ID for auth (streamable-http only, not with --direct)
        auth_scopes (str | None): OAuth2 scopes (comma-separated, streamable-http only, not with --direct)

    Examples:
        nat mcp client ping                                    # Ping default server
        nat mcp client ping --url http://custom-server:9901/mcp # Ping custom server
        nat mcp client ping --timeout 10                      # Use 10 second timeout
        nat mcp client ping --json-output                     # Get JSON format output
        nat mcp client ping --url https://example.com/mcp/ --transport streamable-http --auth # With auth
    """
    # Validate combinations similar to list command
    if not validate_transport_cli_args(transport, command, args, env):
        return

    stdio_args = args.split() if args else []
    stdio_env = dict(var.split('=', 1) for var in env.split()) if env else None

    # Auth validation: if user_id or scopes provided, require redirect_uri
    if (auth_user_id or auth_scopes) and not auth_redirect_uri:
        click.echo("[ERROR] --auth-redirect-uri is required when using --auth-user-id or --auth-scopes", err=True)
        return

    # Parse auth scopes, stripping whitespace
    auth_scopes_list = [scope.strip() for scope in auth_scopes.split(',')] if auth_scopes else None

    result = asyncio.run(
        ping_mcp_server(url,
                        timeout,
                        transport,
                        command,
                        stdio_args,
                        stdio_env,
                        auth_redirect_uri,
                        auth_user_id,
                        auth_scopes_list))

    if json_output:
        click.echo(result.model_dump_json(indent=2))
    elif result.status == "healthy":
        click.echo(f"Server at {result.url} is healthy (response time: {result.response_time_ms}ms)")
    else:
        click.echo(f"Server at {result.url} {result.status}: {result.error}")


async def call_tool_direct(command: str | None,
                           url: str | None,
                           tool_name: str,
                           transport: str,
                           args: list[str] | None,
                           env: dict[str, str] | None,
                           tool_args: dict[str, Any] | None) -> str:
    """Call an MCP tool directly via the selected transport.

    Bypasses the WorkflowBuilder and talks to the MCP server using the raw
    protocol client for the given transport. Aggregates tool outputs into a
    plain string suitable for terminal display. Converts transport/protocol
    exceptions into a structured MCPError for consistency.

    Args:
        command (str | None): For ``stdio`` transport, the command to execute.
        url (str | None): For ``sse`` or ``streamable-http`` transports, the server URL.
        tool_name (str): Name of the tool to call.
        transport (str): One of ``'stdio'``, ``'sse'``, or ``'streamable-http'``.
        args (list[str] | None): For ``stdio`` transport, additional command arguments.
        env (dict[str, str] | None): For ``stdio`` transport, environment variables.
        tool_args (dict[str, Any] | None): JSON-serializable arguments passed to the tool.

    Returns:
        str: Concatenated textual output from the tool invocation.

    Raises:
        MCPError: If the connection, initialization, or tool call fails. When the
            MCP client package is not installed, a generic ``Exception`` is raised
            with an MCP-like error message.
        RuntimeError: If required parameters for the chosen transport are missing
            or if the tool returns an error response.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import TextContent

    try:
        if transport == 'stdio':
            if not command:
                raise RuntimeError("--command is required for stdio transport")

            def get_stdio_client():
                return stdio_client(server=StdioServerParameters(command=command, args=args or [], env=env))

            client = get_stdio_client
        elif transport == 'streamable-http':

            def get_streamable_http_client():
                if not url:
                    raise RuntimeError("--url is required for streamable-http transport")
                return streamablehttp_client(url=url)

            client = get_streamable_http_client
        else:

            def get_sse_client():
                if not url:
                    raise RuntimeError("--url is required for sse transport")
                return sse_client(url=url)

            client = get_sse_client

        async with client() as ctx:
            read, write = (ctx[0], ctx[1]) if isinstance(ctx, tuple) else ctx
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_args or {})

        outputs: list[str] = []
        for content in result.content:
            if isinstance(content, TextContent):
                outputs.append(content.text)
            else:
                outputs.append(str(content))

        # If the result indicates an error, raise to surface in CLI
        if getattr(result, "isError", False):
            raise RuntimeError("\n".join(outputs) or f"Tool call '{tool_name}' returned an error")

        return "\n".join(outputs)
    except Exception as e:
        # Convert raw exceptions to structured MCPError for consistency
        try:
            from nat.plugins.mcp.exception_handler import convert_to_mcp_error
            from nat.plugins.mcp.exception_handler import extract_primary_exception
        except ImportError:
            # Fallback when MCP client package is not installed
            def convert_to_mcp_error(exception: Exception, url: str):
                return Exception(f"Error connecting to {url}: {exception}")

            def extract_primary_exception(exceptions):
                return exceptions[0] if exceptions else Exception("Unknown error")

        endpoint = url or (f"stdio:{command}" if transport == 'stdio' else "unknown")
        if isinstance(e, ExceptionGroup):
            primary_exception = extract_primary_exception(list(e.exceptions))
            mcp_error = convert_to_mcp_error(primary_exception, endpoint)
        else:
            mcp_error = convert_to_mcp_error(e, endpoint)
        raise mcp_error from e


async def call_tool_and_print(command: str | None,
                              url: str | None,
                              tool_name: str,
                              transport: str,
                              args: list[str] | None,
                              env: dict[str, str] | None,
                              tool_args: dict[str, Any] | None,
                              direct: bool,
                              auth_redirect_uri: str | None = None,
                              auth_user_id: str | None = None,
                              auth_scopes: list[str] | None = None) -> str:
    """Call an MCP tool either directly or via the function group and return output.

    When ``direct`` is True, uses the raw MCP protocol client (bypassing the
    builder). Otherwise, constructs the ``mcp_client`` function group and
    invokes the corresponding function, mirroring workflow configuration.

    Args:
        command (str | None): For ``stdio`` transport, the command to execute.
        url (str | None): For ``sse`` or ``streamable-http`` transports, the server URL.
        tool_name (str): Name of the tool to call.
        transport (str): One of ``'stdio'``, ``'sse'``, or ``'streamable-http'``.
        args (list[str] | None): For ``stdio`` transport, additional command arguments.
        env (dict[str, str] | None): For ``stdio`` transport, environment variables.
        tool_args (dict[str, Any] | None): JSON-serializable arguments passed to the tool.
        direct (bool): If True, bypass WorkflowBuilder and use direct MCP client.

    Returns:
        str: Stringified tool output suitable for terminal display. May be an
        empty string when the MCP client package is not installed and ``direct``
        is False.

    Raises:
        RuntimeError: If the tool is not found when using the function group.
        MCPError: Propagated from ``call_tool_direct`` when direct mode fails.
    """
    if direct:
        return await call_tool_direct(command, url, tool_name, transport, args, env, tool_args)

    try:
        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.plugins.mcp.client_config import MCPClientConfig
        from nat.plugins.mcp.client_config import MCPServerConfig
    except ImportError:
        click.echo(
            "MCP client functionality requires nvidia-nat-mcp package. Install with: uv pip install nvidia-nat-mcp",
            err=True)
        return ""

    server_cfg = MCPServerConfig(
        transport=cast(Literal["stdio", "sse", "streamable-http"], transport),
        url=cast(Any, url) if transport in ('sse', 'streamable-http') else None,
        command=command if transport == 'stdio' else None,
        args=args if transport == 'stdio' else None,
        env=env if transport == 'stdio' else None,
    )

    group_cfg = MCPClientConfig(server=server_cfg)

    async with WorkflowBuilder() as builder:  # type: ignore
        # Add auth provider if url is provided and auth_redirect_uri is given (only for streamable-http)
        if url and transport == 'streamable-http' and auth_redirect_uri:
            try:
                group_cfg = await _create_mcp_client_config(builder,
                                                            server_cfg,
                                                            url,
                                                            transport,
                                                            auth_redirect_uri,
                                                            auth_user_id,
                                                            auth_scopes)
            except ImportError:
                click.echo("[WARNING] MCP OAuth2 authentication requires nvidia-nat-mcp package.", err=True)

        group = await builder.add_function_group("mcp_client", group_cfg)
        fns = await group.get_accessible_functions()
        full = f"mcp_client.{tool_name}"
        fn = fns.get(full)
        if fn is None:
            raise RuntimeError(f"Tool '{tool_name}' not found")
        # The group exposes a Function that we can invoke with kwargs
        result = await fn.acall_invoke(**(tool_args or {}))
        # Ensure string output for terminal
        return str(result)


@mcp_client_tool_group.command(name="call", help="Call a tool by name with optional arguments.")
@click.argument('tool_name', nargs=1, required=True)
@click.option('--direct', is_flag=True, help='Bypass MCPBuilder and use direct MCP protocol')
@click.option(
    '--url',
    default='http://localhost:9901/mcp',
    show_default=True,
    help='MCP server URL (e.g. http://localhost:8080/mcp for streamable-http, http://localhost:8080/sse for sse)')
@click.option('--transport',
              type=click.Choice(['sse', 'stdio', 'streamable-http']),
              default='streamable-http',
              show_default=True,
              help='Type of client to use (default: streamable-http, backwards compatible with sse)')
@click.option('--command', help='For stdio: The command to run (e.g. mcp-server)')
@click.option('--args', help='For stdio: Additional arguments for the command (space-separated)')
@click.option('--env', help='For stdio: Environment variables in KEY=VALUE format (space-separated)')
@click.option('--json-args', default=None, help='Pass tool args as a JSON object string')
@click.option('--auth',
              is_flag=True,
              help='Enable OAuth2 authentication with default settings (streamable-http only, not with --direct)')
@click.option('--auth-redirect-uri',
              help='OAuth2 redirect URI for authentication (streamable-http only, not with --direct)')
@click.option('--auth-user-id', help='User ID for authentication (streamable-http only, not with --direct)')
@click.option('--auth-scopes', help='OAuth2 scopes (comma-separated, streamable-http only, not with --direct)')
def mcp_client_tool_call(tool_name: str,
                         direct: bool,
                         url: str | None,
                         transport: str,
                         command: str | None,
                         args: str | None,
                         env: str | None,
                         json_args: str | None,
                         auth: bool,
                         auth_redirect_uri: str | None,
                         auth_user_id: str | None,
                         auth_scopes: str | None) -> None:
    """Call an MCP tool by name with optional JSON arguments.

    Validates transport parameters, parses ``--json-args`` into a dictionary,
    invokes the tool (either directly or via the function group), and prints
    the resulting output to stdout. Errors are formatted consistently with
    other MCP CLI commands.

    Args:
        tool_name (str): Name of the tool to call.
        direct (bool): If True, bypass WorkflowBuilder and use the direct MCP client.
        url (str | None): For ``sse`` or ``streamable-http`` transports, the server URL.
        transport (str): One of ``'stdio'``, ``'sse'``, or ``'streamable-http'``.
        command (str | None): For ``stdio`` transport, the command to execute.
        args (str | None): For ``stdio`` transport, space-separated command arguments.
        env (str | None): For ``stdio`` transport, space-separated ``KEY=VALUE`` pairs.
        json_args (str | None): JSON object string with tool arguments (e.g. '{"q": "hello"}').
        auth_redirect_uri (str | None): redirect URI for auth (streamable-http only, not with --direct)
        auth_user_id (str | None): User ID for authentication (streamable-http only, not with --direct)
        auth_scopes (str | None): OAuth2 scopes (comma-separated, streamable-http only, not with --direct)

    Examples:
        nat mcp client tool call echo --json-args '{"text": "Hello"}'
        nat mcp client tool call search --direct --url http://localhost:9901/mcp \
            --json-args '{"query": "NVIDIA"}' # Direct mode (no auth)
        nat mcp client tool call run --transport stdio --command mcp-server \
            --args "--flag1 --flag2" --env "ENV1=V1 ENV2=V2" --json-args '{}'
        nat mcp client tool call search --url https://example.com/mcp/ --auth \
            --json-args '{"query": "test"}' # With auth using defaults
        nat mcp client tool call search --url https://example.com/mcp/ \
            --transport streamable-http --json-args '{"query": "test"}' --auth
    """
    # Validate transport args
    if not validate_transport_cli_args(transport, command, args, env):
        return

    # Parse stdio params
    stdio_args = args.split() if args else []
    stdio_env = dict(var.split('=', 1) for var in env.split()) if env else None

    # Set auth defaults if --auth flag is used
    auth_redirect_uri, auth_user_id, auth_scopes_list = _set_auth_defaults(
        auth, url, auth_redirect_uri, auth_user_id, auth_scopes
    )

    # Parse tool args
    arg_obj: dict[str, Any] = {}
    if json_args:
        try:
            parsed = json.loads(json_args)
            if not isinstance(parsed, dict):
                click.echo("[ERROR] --json-args must be a JSON object", err=True)
                return
            arg_obj.update(parsed)
        except json.JSONDecodeError as e:
            click.echo(f"[ERROR] Failed to parse --json-args: {e}", err=True)
            return

    try:
        output = asyncio.run(
            call_tool_and_print(
                command=command,
                url=url,
                tool_name=tool_name,
                transport=transport,
                args=stdio_args,
                env=stdio_env,
                tool_args=arg_obj,
                direct=direct,
                auth_redirect_uri=auth_redirect_uri,
                auth_user_id=auth_user_id,
                auth_scopes=auth_scopes_list,
            ))
        if output:
            click.echo(output)
    except MCPError as e:
        format_mcp_error(e, include_traceback=False)
    except Exception as e:
        click.echo(f"[ERROR] {e}", err=True)
