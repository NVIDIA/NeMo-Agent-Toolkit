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
import argparse
import asyncio
import os

import pytest
import uvicorn
from mcp.server.fastmcp.server import FastMCP
from mcp.types import TextContent

from nat.tool.mcp.mcp_client_base import MCPBaseClient
from nat.tool.mcp.mcp_client_base import MCPSSEClient
from nat.tool.mcp.mcp_client_base import MCPStdioClient
from nat.tool.mcp.mcp_client_base import MCPStreamableHTTPClient


def _create_test_mcp_server(port: int) -> FastMCP:
    s = FastMCP(name="Test Server", port=port)

    @s.tool()
    async def return_42(param: str):
        return f"{param} 42 {os.environ.get('TEST', '')}"

    @s.tool()
    async def throw_error(param: str):
        raise RuntimeError(f"Error message: {param}")

    return s


async def _wait_for_uvicorn_server(server: uvicorn.Server):
    # wait up to 50s for server.started to flip True
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(1)
    else:
        pytest.fail("Server failed to start within timeout")


@pytest.fixture(name="mcp_client", params=["stdio", "sse", "streamable-http"])
async def mcp_client_fixture(request: pytest.FixtureRequest, unused_tcp_port_factory):
    os.environ["TEST"] = "env value"  # shared for in-process servers

    server_task: asyncio.Task | None = None
    server: uvicorn.Server | None = None

    transport = request.param

    if transport == "stdio":
        # Launch this file as a stdio server in a child process.
        client = MCPStdioClient(
            command="python",
            args=[
                "-u",
                os.path.abspath(__file__),
                "--transport",
                "stdio",
            ],
            env={
                **os.environ,  # inherit so imports work in CI
                "TEST": os.environ["TEST"],
            },
        )
        # no uvicorn for stdio; nothing to wait for

    elif transport == "sse":
        port = unused_tcp_port_factory()
        mcp_server = _create_test_mcp_server(port=port)
        config = uvicorn.Config(
            app=mcp_server.sse_app(),
            host=mcp_server.settings.host,
            port=port,
            log_level=mcp_server.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        await _wait_for_uvicorn_server(server)
        client = MCPSSEClient(url=f"http://localhost:{port}/sse")

    elif transport == "streamable-http":
        port = unused_tcp_port_factory()
        mcp_server = _create_test_mcp_server(port=port)
        config = uvicorn.Config(
            app=mcp_server.streamable_http_app(),
            host=mcp_server.settings.host,
            port=port,
            log_level=mcp_server.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        await _wait_for_uvicorn_server(server)
        client = MCPStreamableHTTPClient(url=f"http://localhost:{port}/mcp")

    else:
        raise ValueError(f"Invalid transport: {transport}")

    try:
        yield client
    finally:
        # Graceful shutdowns, transport-specific
        if isinstance(client, MCPStdioClient):
            # context manager in tests will close it; nothing else needed here
            pass

        if server is not None:
            server.should_exit = True
        if server_task is not None:
            try:
                await server_task
            except asyncio.CancelledError:
                pass


@pytest.mark.skip(reason="Temporarily disabled while debugging MCP server hang")
async def test_mcp_client_base_methods(mcp_client: MCPBaseClient):

    async with mcp_client:

        # Test get_tools
        tools = await mcp_client.get_tools()
        assert len(tools) == 2
        assert "return_42" in tools

        # Test get_tool
        tool = await mcp_client.get_tool("return_42")
        assert tool.name == "return_42"

        # Test call_tool
        result = await mcp_client.call_tool("return_42", {"param": "value"})

        value = result.content[0]

        assert isinstance(value, TextContent)
        assert value.text == f"value 42 {os.environ['TEST']}"


@pytest.mark.skip(reason="Temporarily disabled while debugging MCP server hang")
async def test_error_handling(mcp_client: MCPBaseClient):
    async with mcp_client:

        tool = await mcp_client.get_tool("throw_error")

        with pytest.raises(RuntimeError) as e:
            await tool.acall({"param": "value"})

        assert "Error message: value" in str(e.value)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MCP Server")
    parser.add_argument("--transport", type=str, default="stdio", help="Transport to use for the server")

    args = parser.parse_args()

    _create_test_mcp_server(port=8122).run(transport=args.transport)
