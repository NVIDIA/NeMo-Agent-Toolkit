#!/usr/bin/env python3
# Minimal MCP "list tools" script without Click.
# Python 3.11+/3.12. Requires: modelcontextprotocol (MCP) installed.

import argparse
import asyncio
import json
import sys
from typing import Any

# MCP imports
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import StdioServerParameters, stdio_client


def _format_tool(tool: Any) -> dict[str, str | None]:
    """Normalize tool fields (name, description, input_schema) to strings."""
    name = getattr(tool, "name", None)
    description = getattr(tool, "description", "") or ""
    input_schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)

    if input_schema is None:
        schema_str = None
    elif hasattr(input_schema, "schema_json"):
        try:
            schema_str = input_schema.schema_json(indent=2)
        except Exception:
            schema_str = json.dumps({"raw": str(input_schema)}, indent=2)
    elif isinstance(input_schema, dict):
        schema_str = json.dumps(input_schema, indent=2)
    else:
        schema_str = json.dumps({"raw": str(input_schema)}, indent=2)

    return {
        "name": name,
        "description": description,
        "input_schema": schema_str,
    }


async def list_tools_direct(
    *,
    url: str,
    transport: str = "streamable-http",
    tool_name: str | None = None,
    stdio_command: str | None = None,
    stdio_args: list[str] | None = None,
    stdio_env: dict[str, str] | None = None,
) -> list[dict[str, str | None]]:
    """
    Open the chosen MCP transport, create a ClientSession, initialize, and call list_tools().
    Mirrors the SDK's recommended 'async with transport -> async with ClientSession' pattern.
    """
    if transport == "stdio":
        if not stdio_command:
            raise ValueError("--stdio-command is required when transport=stdio")
        client_ctx = stdio_client(
            server=StdioServerParameters(command=stdio_command, args=stdio_args or [], env=stdio_env)
        )
    elif transport == "sse":
        client_ctx = sse_client(url=url)
    elif transport == "streamable-http":
        client_ctx = streamablehttp_client(url=url)
    else:
        raise ValueError(f"Unknown transport: {transport}")

    async with client_ctx as ctx:
        # Some transports yield (read, write, extra), others yield (read, write)
        if isinstance(ctx, tuple):
            read, write = ctx[0], ctx[1]
        else:
            read, write = ctx

        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()

    out: list[dict[str, str | None]] = []
    for tool in response.tools:
        if tool_name is None or tool.name == tool_name:
            out.append(_format_tool(tool))

    return out


def _parse_env_kv(pairs: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"Invalid --stdio-env '{item}', expected KEY=VALUE")
        k, v = item.split("=", 1)
        result[k] = v
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal MCP list-tools (no Click).")
    p.add_argument("--url", default="http://localhost:9901/mcp", help="MCP server URL")
    p.add_argument(
        "--transport",
        choices=["streamable-http", "sse", "stdio"],
        default="streamable-http",
        help="Transport type",
    )
    p.add_argument("--tool", dest="tool_name", default=None, help="Filter for a specific tool name")
    p.add_argument("--json", action="store_true", help="Output JSON")
    # stdio extras
    p.add_argument("--stdio-command", help="Command to run for stdio transport")
    p.add_argument("--stdio-arg", action="append", default=[], help="Repeatable: extra args for stdio")
    p.add_argument("--stdio-env", action="append", default=[], help="Repeatable: KEY=VALUE env pairs for stdio")

    args = p.parse_args()

    stdio_env = _parse_env_kv(args.stdio_env) if args.stdio_env else None

    try:
        tools = asyncio.run(
            list_tools_direct(
                url=args.url,
                transport=args.transport,
                tool_name=args.tool_name,
                stdio_command=args.stdio_command,
                stdio_args=args.stdio_arg,
                stdio_env=stdio_env,
            )
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(tools, indent=2))
        return

    # pretty text output
    if args.tool_name:
        if not tools:
            print(f"[INFO] Tool '{args.tool_name}' not found.")
            return
        for t in tools:
            print(f"Tool: {t.get('name')}")
            print(f"Description: {t.get('description') or 'No description available'}")
            if t.get("input_schema"):
                print("Input Schema:")
                print(t["input_schema"])
            else:
                print("Input Schema: None")
            print("-" * 60)
    else:
        # names only
        for t in tools:
            print(t.get("name", "Unknown tool"))


if __name__ == "__main__":
    main()
