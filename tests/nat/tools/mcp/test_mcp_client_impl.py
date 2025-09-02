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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from pydantic import BaseModel

from nat.builder.workflow_builder import WorkflowBuilder
from nat.tool.mcp.mcp_client_base import MCPBaseClient
from nat.tool.mcp.mcp_client_impl import MCPClientConfig
from nat.tool.mcp.mcp_client_impl import MCPServerConfig
from nat.tool.mcp.mcp_client_impl import MCPSingleToolConfig
from nat.tool.mcp.mcp_client_impl import ToolOverrideConfig
from nat.tool.mcp.mcp_client_impl import mcp_client_function_handler


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


def test_filter_and_configure_tools_none_filter_returns_all():
    """Test that None filter returns all tools."""
    from nat.tool.mcp.mcp_client_impl import _filter_and_configure_tools
    tools = {"a": _FakeTool("a", "da"), "b": _FakeTool("b", "db")}
    out = _filter_and_configure_tools(tools, tool_filter=None)
    assert out == {
        "a": {
            "function_name": "a", "description": "da"
        },
        "b": {
            "function_name": "b", "description": "db"
        },
    }


def test_filter_and_configure_tools_list_filter_subsets():
    """Test that list filter returns subset of tools."""
    from nat.tool.mcp.mcp_client_impl import _filter_and_configure_tools
    tools = {"a": _FakeTool("a", "da"), "b": _FakeTool("b", "db")}
    out = _filter_and_configure_tools(tools, tool_filter=["b"])  # type: ignore[arg-type]
    assert out == {
        "b": {
            "function_name": "b", "description": "db"
        },
    }


def test_filter_and_configure_tools_dict_overrides_alias_and_description(caplog):
    """Test that dict filter with overrides works correctly."""
    tools = {"raw": _FakeTool("raw", "original")}
    overrides = {"raw": ToolOverrideConfig(alias="alias", description="new desc")}
    from nat.tool.mcp.mcp_client_impl import _filter_and_configure_tools
    out = _filter_and_configure_tools(tools, tool_filter=overrides)  # type: ignore[arg-type]
    assert out == {"raw": {"function_name": "alias", "description": "new desc"}}


async def test_mcp_client_function_handler():
    """Test MCP client function handler."""
    with patch("nat.tool.mcp.mcp_client_base.MCPStdioClient") as mock_client:
        fake_tools = {
            "fake_tool_1": _FakeTool("fake_tool_1", "A fake tool for testing"),
            "fake_tool_2": _FakeTool("fake_tool_2", "Another fake tool for testing")
        }

        mock_client.return_value = _FakeMCPClient(tools=fake_tools, command="python", args=["server.py"])

        server_cfg = MCPServerConfig(transport="stdio", command="python", args=["server.py"])
        client_cfg = MCPClientConfig(server=server_cfg, tool_filter=["fake_tool_1"])

        # Mock the WorkflowBuilder
        mock_builder = MagicMock(spec=WorkflowBuilder)
        mock_builder.add_function = AsyncMock()

        # Test the handler function
        async with mcp_client_function_handler(client_cfg, mock_builder) as fn_info:
            # fn_info is the idle FunctionInfo ("MCP client")
            assert fn_info.description == "MCP client"

        # Verify the MCP client was constructed and used
        mock_client.assert_called_once()

        # Verify that add_function was awaited exactly once for the filtered tool
        mock_builder.add_function.assert_awaited_once()
        name, cfg = mock_builder.add_function.await_args.args
        assert name == "fake_tool_1"
        assert isinstance(cfg, MCPSingleToolConfig)
        assert cfg.tool_name == "fake_tool_1"  # original tool name
        assert cfg.tool_description == "A fake tool for testing"  # carried through
