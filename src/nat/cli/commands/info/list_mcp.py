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

import click
from pydantic import BaseModel

from nat.tool.mcp.exceptions import MCPError
from nat.tool.mcp.mcp_client import MCPBuilder
from nat.utils.exception_handlers.mcp import format_mcp_error

# Suppress verbose logs from mcp.client.sse and httpx
logging.getLogger("mcp.client.sse").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


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

    schema_str = None
    if input_schema:
        if hasattr(input_schema, "schema_json"):
            schema_str = input_schema.schema_json(indent=2)
        else:
            schema_str = str(input_schema)

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


async def list_tools_and_schemas(url: str, tool_name: str | None = None) -> list[dict[str, str | None]]:
    """List MCP tools using MCPBuilder with structured exception handling.

    Args:
        url (str): MCP server URL to connect to
        tool_name (str | None, optional): Specific tool name to retrieve.
        If None, retrieves all available tools. Defaults to None.

    Returns:
        list[dict[str, str | None]]: List of formatted tool dictionaries, each containing name, description, and
        input_schema as keys

    Raises:
        MCPError: Caught internally and logged, returns empty list instead
    """
    builder = MCPBuilder(url=url)
    try:
        if tool_name:
            tool = await builder.get_tool(tool_name)
            return [format_tool(tool)]
        tools = await builder.get_tools()
        return [format_tool(tool) for tool in tools.values()]
    except MCPError as e:
        format_mcp_error(e, include_traceback=False)
        return []


async def list_tools_direct(url: str, tool_name: str | None = None) -> list[dict[str, str | None]]:
    """List MCP tools using direct MCP protocol with exception conversion.

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
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    try:
        async with sse_client(url=url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()

                tools = []
                for tool in response.tools:
                    if tool_name:
                        if tool.name == tool_name:
                            return [format_tool(tool)]
                    else:
                        tools.append(format_tool(tool))
                if tool_name and not tools:
                    click.echo(f"[INFO] Tool '{tool_name}' not found.")
                return tools
    except Exception as e:
        # Convert raw exceptions to structured MCPError for consistency
        from nat.utils.exception_handlers.mcp import convert_to_mcp_error
        from nat.utils.exception_handlers.mcp import extract_primary_exception

        if isinstance(e, ExceptionGroup):  # noqa: F821
            primary_exception = extract_primary_exception(list(e.exceptions))
            mcp_error = convert_to_mcp_error(primary_exception, url)
        else:
            mcp_error = convert_to_mcp_error(e, url)

        format_mcp_error(mcp_error, include_traceback=False)
        return []


async def ping_mcp_server(url: str, timeout: int) -> MCPPingResult:
    """Ping an MCP server to check if it's responsive.

    Args:
        url (str): MCP server URL to ping
        timeout (int): Timeout in seconds for the ping request

    Returns:
        MCPPingResult: Structured result with status, response_time, and any error info
    """
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client

    async def _ping_operation():
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()

                # Record start time just before ping
                start_time = time.time()
                # Send ping request
                await session.send_ping()

                end_time = time.time()
                response_time_ms = round((end_time - start_time) * 1000, 2)

                return MCPPingResult(url=url, status="healthy", response_time_ms=response_time_ms, error=None)

    try:
        # Apply timeout to the entire ping operation
        return await asyncio.wait_for(_ping_operation(), timeout=timeout)

    except asyncio.TimeoutError:
        return MCPPingResult(url=url,
                             status="unreachable",
                             response_time_ms=None,
                             error=f"Timeout after {timeout} seconds")

    except Exception as e:
        return MCPPingResult(url=url, status="unhealthy", response_time_ms=None, error=str(e))


@click.group(invoke_without_command=True, help="List tool names (default), or show details with --detail or --tool.")
@click.option('--direct', is_flag=True, help='Bypass MCPBuilder and use direct MCP protocol')
@click.option('--url', default='http://localhost:9901/sse', show_default=True, help='MCP server URL')
@click.option('--tool', default=None, help='Get details for a specific tool by name')
@click.option('--detail', is_flag=True, help='Show full details for all tools')
@click.option('--json-output', is_flag=True, help='Output tool metadata in JSON format')
@click.pass_context
def list_mcp(ctx: click.Context, direct: bool, url: str, tool: str | None, detail: bool, json_output: bool) -> None:
    """List MCP tool names (default) or show detailed tool information.

    Use --detail for full output including descriptions and input schemas.
    If --tool is provided, always shows full output for that specific tool.
    Use --direct to bypass MCPBuilder and use raw MCP protocol.
    Use --json-output to get structured JSON data instead of formatted text.

    Args:
        ctx (click.Context): Click context object for command invocation
        direct (bool): Whether to bypass MCPBuilder and use direct MCP protocol
        url (str): MCP server URL to connect to (default: http://localhost:9901/sse)
        tool (str | None): Optional specific tool name to retrieve detailed info for
        detail (bool): Whether to show full details (description + schema) for all tools
        json_output (bool): Whether to output tool metadata in JSON format instead of text

    Examples:
        nat info mcp                           # List tool names only
        nat info mcp --detail                  # Show all tools with full details
        nat info mcp --tool my_tool            # Show details for specific tool
        nat info mcp --json-output             # Get JSON format output
        nat info mcp --direct --url http://...  # Use direct protocol with custom URL
    """
    if ctx.invoked_subcommand is not None:
        return
    fetcher = list_tools_direct if direct else list_tools_and_schemas
    tools = asyncio.run(fetcher(url, tool))

    if json_output:
        click.echo(json.dumps(tools, indent=2))
    elif tool:
        for tool_dict in tools:
            print_tool(tool_dict, detail=True)
    elif detail:
        for tool_dict in tools:
            print_tool(tool_dict, detail=True)
    else:
        for tool_dict in tools:
            click.echo(tool_dict.get('name', 'Unknown tool'))


@list_mcp.command()
@click.option('--url', default='http://localhost:9901/sse', show_default=True, help='MCP server URL')
@click.option('--timeout', default=60, show_default=True, help='Timeout in seconds for ping request')
@click.option('--json-output', is_flag=True, help='Output ping result in JSON format')
def ping(url: str, timeout: int, json_output: bool) -> None:
    """Ping an MCP server to check if it's responsive.

    This command sends a ping request to the MCP server and measures the response time.
    It's useful for health checks and monitoring server availability.

    Args:
        url (str): MCP server URL to ping (default: http://localhost:9901/sse)
        timeout (int): Timeout in seconds for the ping request (default: 60)
        json_output (bool): Whether to output the result in JSON format

    Examples:
        nat info mcp ping                                    # Ping default server
        nat info mcp ping --url http://custom-server:9901/sse # Ping custom server
        nat info mcp ping --timeout 10                      # Use 10 second timeout
        nat info mcp ping --json-output                     # Get JSON format output
    """
    result = asyncio.run(ping_mcp_server(url, timeout))

    if json_output:
        click.echo(result.model_dump_json(indent=2))
    else:
        if result.status == "healthy":
            click.echo(f"Server at {result.url} is healthy (response time: {result.response_time_ms}ms)")
        else:
            click.echo(f"Server at {result.url} {result.status}: {result.error}")
