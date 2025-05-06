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

import click

from aiq.tool.mcp.mcp_client import MCPBuilder


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


def print_tool(tool_dict):
    click.echo(f"Tool: {tool_dict['name']}")
    click.echo(f"Description: {tool_dict['description']}")
    if tool_dict["input_schema"]:
        click.echo("Input Schema:")
        click.echo(tool_dict["input_schema"])
    else:
        click.echo("Input Schema: None")
    click.echo("-" * 60)


async def list_tools_and_schemas(url, tool_name=None):
    builder = MCPBuilder(url=url)
    try:
        if tool_name:
            tool = await builder.get_tool(tool_name)
            return [format_tool(tool)]
        else:
            tools = await builder.get_tools()
            return [format_tool(tool) for tool in tools.values()]
    except Exception as e:
        click.echo(f"[ERROR] Failed to fetch tools via MCPBuilder: {e}", err=True)
        return []


async def list_tools_direct(url, tool_name=None):
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
        click.echo(f"[ERROR] Failed to fetch tools via direct protocol: {e}", err=True)
        return []


@click.group(invoke_without_command=True, help="Interact with tools on an MCP server.")
@click.pass_context
def list_mcp(ctx):
    """Interact with tools on an MCP server."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(list_tools)


@list_mcp.command("list", help="List tools and their descriptions from an MCP server.")
@click.option('--direct', is_flag=True, help='Bypass MCPBuilder and use direct MCP protocol')
@click.option('--url', default='http://localhost:8080/sse', show_default=True, help='MCP server URL')
@click.option('--tool', default=None, help='Get details for a specific tool by name')
@click.option('--json-output', is_flag=True, help='Output tool metadata in JSON format')
def list_tools(direct, url, tool, json_output):
    fetcher = list_tools_direct if direct else list_tools_and_schemas
    tools = asyncio.run(fetcher(url, tool))

    if json_output:
        click.echo(json.dumps(tools, indent=2))
    else:
        for tool_dict in tools:
            print_tool(tool_dict)
