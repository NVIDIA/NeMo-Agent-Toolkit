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

import json
import logging

import anyio
import click

# Suppress verbose logs from mcp.client.sse and httpx
logging.getLogger("mcp.client.sse").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def format_tool(tool):
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


def print_tool(tool_dict, detail=False):
    click.echo(f"Tool: {tool_dict['name']}")
    if detail or tool_dict.get('input_schema') or tool_dict.get('description'):
        click.echo(f"Description: {tool_dict['description']}")
        if tool_dict["input_schema"]:
            click.echo("Input Schema:")
            click.echo(tool_dict["input_schema"])
        else:
            click.echo("Input Schema: None")
        click.echo("-" * 60)


async def list_tools_and_schemas(command, url, tool_name=None, transport='sse', args=None, env=None):
    if args is None:
        args = []
    from aiq.tool.mcp.mcp_client_base import MCPSSEClient
    from aiq.tool.mcp.mcp_client_base import MCPStdioClient
    from aiq.tool.mcp.mcp_client_base import MCPStreamableHTTPClient

    try:
        if transport == 'stdio':
            client = MCPStdioClient(command=command, args=args, env=env)
        elif transport == 'streamable-http':
            client = MCPStreamableHTTPClient(url=url)
        else:  # sse
            client = MCPSSEClient(url=url)

        async with client:
            if tool_name:
                tool = await client.get_tool(tool_name)
                return [format_tool(tool)]
            else:
                tools = await client.get_tools()
                return [format_tool(tool) for tool in tools.values()]
    except Exception as e:
        click.echo(f"[ERROR] Failed to fetch tools via MCP client: {e}", err=True)
        return []


async def list_tools_direct(command, url, tool_name=None, transport='sse', args=None, env=None):
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
        else:  # transport

            def get_sse_client():
                return sse_client(url=url)

            client = get_sse_client

        async with client() as (read, write):
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
        click.echo(f"[ERROR] Failed to fetch tools via direct protocol: {e}", err=True)
        return []


@click.group(invoke_without_command=True, help="List tool names (default), or show details with --detail or --tool.")
@click.option('--direct', is_flag=True, help='Bypass MCPBuilder and use direct MCP protocol')
@click.option('--url',
              default='http://localhost:9901/sse',
              show_default=True,
              help='For SSE/StreamableHTTP: MCP server URL (e.g. http://localhost:8080/sse)')
@click.option('--transport',
              type=click.Choice(['sse', 'stdio', 'streamable-http']),
              default='sse',
              show_default=True,
              help='Type of client to use')
@click.option('--command', help='For stdio: The command to run (e.g. mcp-server)')
@click.option('--args', help='For stdio: Additional arguments for the command (space-separated)')
@click.option('--env', help='For stdio: Environment variables in KEY=VALUE format (space-separated)')
@click.option('--tool', default=None, help='Get details for a specific tool by name')
@click.option('--detail', is_flag=True, help='Show full details for all tools')
@click.option('--json-output', is_flag=True, help='Output tool metadata in JSON format')
@click.pass_context
def list_mcp(ctx, direct, url, transport, command, args, env, tool, detail, json_output):
    """
    List tool names (default). Use --detail for full output. If --tool is provided,
    always show full output for that tool.
    """
    if ctx.invoked_subcommand is not None:
        return

    if transport == 'stdio':
        if not command:
            click.echo("[ERROR] --command is required when using stdio client type", err=True)
            return

    if transport in ['sse', 'streamable-http']:
        if not url:
            click.echo("[ERROR] --url is required when using sse or streamable-http client type", err=True)
            return
        if command or args or env:
            click.echo(
                "[ERROR] --command, --args, and --env are not allowed when using sse or streamable-http client type",
                err=True)
            return

    stdio_args = args.split() if args else []
    stdio_env = dict(var.split('=', 1) for var in env.split()) if env else None

    fetcher = list_tools_direct if direct else list_tools_and_schemas
    tools = anyio.run(fetcher, command, url, tool, transport, stdio_args, stdio_env)

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
            click.echo(tool_dict['name'])
