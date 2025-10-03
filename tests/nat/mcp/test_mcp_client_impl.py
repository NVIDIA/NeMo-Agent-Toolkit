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

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

from pydantic import BaseModel

from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.mcp.client_base import MCPBaseClient
from nat.plugins.mcp.client_config import MCPClientConfig
from nat.plugins.mcp.client_config import MCPServerConfig
from nat.plugins.mcp.client_config import MCPToolOverrideConfig
from nat.plugins.mcp.client_impl import mcp_apply_tool_alias_and_description
from nat.plugins.mcp.client_impl import mcp_client_function_group


class _InputSchema(BaseModel):
    """Input schema for fake tools used in testing."""
    param: str


class _FakeTool:
    """Fake tool class for testing MCP tool functionality."""

    def __init__(self, name: str, description: str = "desc") -> None:
        self.name = name
        self.description = description
        self.input_schema = _InputSchema

    async def acall(self, args: dict[str, Any]) -> str:
        """Simulate tool execution by returning a formatted response."""
        return f"ok {args['param']}"

    def set_description(self, description: str) -> None:
        """Allow description to be updated for testing purposes."""
        if description is not None:
            self.description = description


class _FakeMCPClient(MCPBaseClient):
    """Fake MCP client for testing client-server interactions."""

    def __init__(self,
                 *,
                 tools: dict[str, _FakeTool],
                 url: str | None = None,
                 command: str | None = None,
                 args: list[str] | None = None) -> None:
        super().__init__("stdio")
        self._tools = tools
        self.url = url
        self.command = command

    async def get_tool(self, name: str) -> _FakeTool:
        """Retrieve a tool by name."""
        return self._tools[name]

    async def get_tools(self) -> dict[str, _FakeTool]:
        """Retrieve all tools."""
        return self._tools

    @asynccontextmanager
    async def connect_to_server(self):
        """Support async context manager for testing."""
        yield self


def test_mcp_apply_tool_alias_and_description_none_returns_empty():
    """If no overrides are provided, helper returns empty mapping."""
    tools = {"a": _FakeTool("a", "da"), "b": _FakeTool("b", "db")}
    out = mcp_apply_tool_alias_and_description(tools, tool_overrides=None)
    assert out == {}


def test_mcp_apply_tool_alias_and_description_filters_to_existing():
    """Only keep overrides for tools that exist in discovery list."""
    tools = {"a": _FakeTool("a", "da")}
    overrides = {"a": MCPToolOverrideConfig(alias=None, description=None), "missing": MCPToolOverrideConfig()}
    out = mcp_apply_tool_alias_and_description(tools, tool_overrides=overrides)
    assert set(out.keys()) == {"a"}


def test_mcp_apply_tool_alias_and_description_applies_alias_and_desc(caplog):
    """Alias and description are applied when provided."""
    tools = {"raw": _FakeTool("raw", "original")}
    overrides = {"raw": MCPToolOverrideConfig(alias="alias", description="new desc")}
    out = mcp_apply_tool_alias_and_description(tools, tool_overrides=overrides)
    assert "raw" in out
    assert out["raw"].alias == "alias"
    assert out["raw"].description == "new desc"


async def test_mcp_client_function_group_includes_respected():
    """Function group exposes only included tools as accessible functions."""
    with patch("nat.plugins.mcp.client_base.MCPStdioClient") as mock_client:
        fake_tools = {
            "fake_tool_1": _FakeTool("fake_tool_1", "A fake tool for testing"),
            "fake_tool_2": _FakeTool("fake_tool_2", "Another fake tool for testing"),
        }

        mock_client.return_value = _FakeMCPClient(tools=fake_tools, command="python", args=["server.py"])

        server_cfg = MCPServerConfig(transport="stdio", command="python", args=["server.py"])
        client_cfg = MCPClientConfig(server=server_cfg, include=["fake_tool_1"])  # only include one tool

        mock_builder = MagicMock(spec=WorkflowBuilder)

        async with mcp_client_function_group(client_cfg, mock_builder) as group:
            accessible = await group.get_accessible_functions()
            assert set(accessible.keys()) == {"mcp_client.fake_tool_1"}


async def test_mcp_client_function_group_applies_overrides():
    with patch("nat.plugins.mcp.client_base.MCPStdioClient") as mock_client:
        fake_tools = {"raw": _FakeTool("raw", "original")}
        mock_client.return_value = _FakeMCPClient(tools=fake_tools, command="python", args=["server.py"])

        server_cfg = MCPServerConfig(transport="stdio", command="python", args=["server.py"])
        client_cfg = MCPClientConfig(
            server=server_cfg,
            include=["alias_raw"],
            tool_overrides={"raw": MCPToolOverrideConfig(alias="alias_raw", description="new desc")},
        )

        mock_builder = MagicMock(spec=WorkflowBuilder)

        async with mcp_client_function_group(client_cfg, mock_builder) as group:
            accessible = await group.get_accessible_functions()
            assert set(accessible.keys()) == {"mcp_client.alias_raw"}
            assert accessible["mcp_client.alias_raw"].description == "new desc"


async def test_mcp_client_function_group_no_include_exposes_all():
    with patch("nat.plugins.mcp.client_base.MCPStdioClient") as mock_client:
        fake_tools = {"a": _FakeTool("a", "da"), "b": _FakeTool("b", "db")}
        mock_client.return_value = _FakeMCPClient(tools=fake_tools, command="python", args=["server.py"])

        server_cfg = MCPServerConfig(transport="stdio", command="python", args=["server.py"])
        client_cfg = MCPClientConfig(server=server_cfg)  # no include/exclude

        mock_builder = MagicMock(spec=WorkflowBuilder)

        async with mcp_client_function_group(client_cfg, mock_builder) as group:
            accessible = await group.get_accessible_functions()
            assert set(accessible.keys()) == {"mcp_client.a", "mcp_client.b"}
