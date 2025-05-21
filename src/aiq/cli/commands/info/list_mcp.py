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

from aiq.tool.mcp.mcp_client import MCPBuilder

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


@click.group(invoke_without_command=True, help="List tool names (default), or show details with --detail or --tool.")
@click.option('--direct', is_flag=True, help='Bypass MCPBuilder and use direct MCP protocol')
@click.option('--url', default='http://localhost:9901/sse', show_default=True, help='MCP server URL')
@click.option('--tool', default=None, help='Get details for a specific tool by name')
@click.option('--detail', is_flag=True, help='Show full details for all tools')
@click.option('--json-output', is_flag=True, help='Output tool metadata in JSON format')
@click.pass_context
def list_mcp(ctx, direct, url, tool, detail, json_output):
    """
    List tool names (default). Use --detail for full output. If --tool is provided,
    always show full output for that tool.
    """
    if ctx.invoked_subcommand is not None:
        return
    fetcher = list_tools_direct if direct else list_tools_and_schemas
    tools = anyio.run(fetcher, url, tool)

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
