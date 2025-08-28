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
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from pydantic import BaseModel

from mcp.types import TextContent
from nat.builder.workflow_builder import WorkflowBuilder
from nat.tool.mcp.mcp_client_impl import MCPClientConfig
from nat.tool.mcp.mcp_client_impl import MCPServerConfig
from nat.tool.mcp.mcp_client_impl import MCPSingleToolConfig
from nat.tool.mcp.mcp_client_impl import ToolOverrideConfig
from nat.tool.mcp.mcp_client_impl import _filter_and_configure_tools
from nat.tool.mcp.mcp_client_base import MCPBaseClient


class _InputSchema(BaseModel):
    param: str


class _FakeTool:
    def __init__(self, name: str, description: str = "desc") -> None:
        self.name = name
        self.description = description
        self.input_schema = _InputSchema

    async def acall(self, args: dict[str, Any]) -> str:
        return f"ok {args['param']}"

    def set_description(self, description: str | None) -> None:
        if description is not None:
            self.description = description


class _ErrorTool(_FakeTool):
    async def acall(self, args: dict[str, Any]) -> str:  # type: ignore[override]
        raise RuntimeError("boom")


class _FakeSession:
    def __init__(self, tools: dict[str, _FakeTool]) -> None:
        self._tools = tools

    class _ToolInfo:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description
            # Provide a trivial input schema compatible with MCPToolClient expectations
            self.inputSchema = {"type": "object", "properties": {"param": {"type": "string"}}, "required": ["param"]}

    class _ListToolsResponse:
        def __init__(self, tools: list["_FakeSession._ToolInfo"]) -> None:
            self.tools = tools

    async def list_tools(self) -> "_FakeSession._ListToolsResponse":
        infos = [self._ToolInfo(name=t.name, description=t.description) for t in self._tools.values()]
        return self._ListToolsResponse(tools=infos)

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]):
        tool = self._tools[tool_name]
        class _Result:
            def __init__(self, text: str):
                self.content = [TextContent(type="text", text=text)]
                self.isError = False
        return _Result(await tool.acall(tool_args))


class _FakeMCPClient(MCPBaseClient):
    def __init__(self, *, tools: dict[str, _FakeTool], transport: str = "streamable-http", url: str | None = None,
                 command: str | None = None) -> None:
        super().__init__(transport)
        self._tools_map = tools
        self._url = url
        self._command = command

    @property
    def url(self) -> str | None:
        return self._url

    @property
    def command(self) -> str | None:
        return self._command

    @asynccontextmanager
    async def connect_to_server(self):
        yield _FakeSession(self._tools_map)

    async def get_tools(self) -> dict[str, _FakeTool]:  # type: ignore[override]
        return self._tools_map

    async def get_tool(self, tool_name: str) -> _FakeTool:  # type: ignore[override]
        return self._tools_map[tool_name]


async def test_mcp_single_tool_happy_path_kwargs():
    client = _FakeMCPClient(tools={"echo": _FakeTool("echo", "Echo tool")})

    async with WorkflowBuilder() as builder:
        fn = await builder.add_function(
            name="echo_fn",
            config=MCPSingleToolConfig(client=client, tool_name="echo", tool_description="Overridden desc"),
        )

        # Validate invocation path using kwargs
        result = await fn.acall_invoke(param="value")
        assert result == "ok value"


async def test_mcp_single_tool_returns_error_string_on_exception():
    client = _FakeMCPClient(tools={"err": _ErrorTool("err", "Err tool")})

    async with WorkflowBuilder() as builder:
        fn = await builder.add_function(
            name="err_fn",
            config=MCPSingleToolConfig(client=client, tool_name="err"),
        )

        result = await fn.acall_invoke(param="value")
        assert "boom" in result


def test_filter_and_configure_tools_none_filter_returns_all():
    tools = {"a": _FakeTool("a", "da"), "b": _FakeTool("b", "db")}
    out = _filter_and_configure_tools(tools, tool_filter=None)
    assert out == {
        "a": {"function_name": "a", "description": "da"},
        "b": {"function_name": "b", "description": "db"},
    }


def test_filter_and_configure_tools_list_filter_subsets():
    tools = {"a": _FakeTool("a", "da"), "b": _FakeTool("b", "db"), "c": _FakeTool("c", "dc")}
    out = _filter_and_configure_tools(tools, tool_filter=["b", "c"])  # type: ignore[arg-type]
    assert out == {
        "b": {"function_name": "b", "description": "db"},
        "c": {"function_name": "c", "description": "dc"},
    }


def test_filter_and_configure_tools_dict_overrides_alias_and_description(caplog):
    tools = {"raw": _FakeTool("raw", "original")}
    overrides = {"raw": ToolOverrideConfig(alias="alias", description="new desc")}
    out = _filter_and_configure_tools(tools, tool_filter=overrides)  # type: ignore[arg-type]
    assert out == {"raw": {"function_name": "alias", "description": "new desc"}}


@patch("nat.tool.mcp.mcp_client_base.MCPSSEClient")
@patch("nat.tool.mcp.mcp_client_base.MCPStdioClient")
@patch("nat.tool.mcp.mcp_client_base.MCPStreamableHTTPClient")
async def test_mcp_client_function_handler_registers_tools(mock_mcp_streamable_http_client, mock_mcp_stdio_client, mock_mcp_sse_client):
    # Prepare fake client classes to be used by the handler
    fake_tools = {"t1": _FakeTool("t1", "d1"), "t2": _FakeTool("t2", "d2")}

    def _mk_client(_: str):
        return _FakeMCPClient(tools=fake_tools, transport="sse", url="http://x")

    # Set up the mock side effects
    mock_mcp_sse_client.side_effect = lambda url: _mk_client(url)
    mock_mcp_stdio_client.side_effect = lambda command, args, env: _FakeMCPClient(tools=fake_tools, transport="stdio", command=command)
    mock_mcp_streamable_http_client.side_effect = lambda url: _mk_client(url)

    server_cfg = MCPServerConfig(transport="sse", url=cast(Any, "http://fake"))
    client_cfg = MCPClientConfig(server=server_cfg, tool_filter=["t1"])  # only expose t1

    async with WorkflowBuilder() as builder:
        # Adding the handler function triggers discovery and registering of tools
        await builder.add_function(name="mcp_client", config=client_cfg)

        # Confirm that the filtered tool has been registered as a function and is invokable
        fn = builder.get_function("t1")
        out = await fn.acall_invoke(param="v")
        assert out == "ok v"
