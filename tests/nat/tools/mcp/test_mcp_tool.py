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

import pytest
from pydantic import BaseModel
from pydantic.networks import HttpUrl

from nat.tool.mcp.mcp_client_base import MCPBaseClient
from nat.tool.mcp.mcp_tool import MCPToolConfig
from nat.tool.mcp.mcp_tool import mcp_tool


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

    def __init__(self, *, tools: dict[str, _FakeTool], url: HttpUrl | None = None) -> None:
        super().__init__("streamable-http")
        self._tools = tools
        self.url = url

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


def test_mcp_tool_config_validation_stdio_requires_command():
    """Test that stdio transport requires command parameter."""
    with pytest.raises(ValueError, match="command is required"):
        MCPToolConfig(transport="stdio", mcp_tool_name="test_tool"
                      # Missing command
                      )


def test_mcp_tool_config_validation_stdio_rejects_url():
    """Test that stdio transport rejects URL parameter."""
    with pytest.raises(ValueError, match="url should not be set"):
        MCPToolConfig(
            transport="stdio",
            mcp_tool_name="test_tool",
            command="python",
            url=HttpUrl("http://localhost:8000/mcp")  # Should not be set for stdio
        )


def test_mcp_tool_config_validation_http_requires_url():
    """Test that HTTP transports require URL parameter."""
    with pytest.raises(ValueError, match="url is required"):
        MCPToolConfig(transport="streamable-http", mcp_tool_name="test_tool"
                      # Missing url
                      )


def test_mcp_tool_config_validation_http_rejects_stdio_params():
    """Test that HTTP transports reject stdio parameters."""
    with pytest.raises(ValueError, match="command, args, and env should not be set"):
        MCPToolConfig(
            transport="sse",
            mcp_tool_name="test_tool",
            url=HttpUrl("http://localhost:8000/mcp"),
            command="python",  # Should not be set for HTTP
            args=["script.py"],  # Should not be set for HTTP
            env={"DEBUG": "1"}  # Should not be set for HTTP
        )


def test_mcp_tool_config_defaults():
    """Test that MCPToolConfig has correct defaults."""
    config = MCPToolConfig(mcp_tool_name="test_tool", url=HttpUrl("http://localhost:8000/mcp"))
    assert config.transport == "streamable-http"
    assert config.return_exception is True
    assert config.description is None


def test_mcp_tool_invalid_transport_raises_error():
    """Test that invalid transport type raises ValueError."""
    with pytest.raises(ValueError, match="Input should be 'sse', 'stdio' or 'streamable-http'"):
        MCPToolConfig(
            transport="invalid",  # type: ignore[assignment]
            mcp_tool_name="test_tool",
            url=HttpUrl("http://localhost:8000/mcp"))


async def test_mcp_tool_streamable_http_client_initialization():
    """Test that streamable-http client is properly initialized with URL."""
    with patch("nat.tool.mcp.mcp_client_base.MCPStreamableHTTPClient") as mock_client:
        fake_tool = _FakeTool("test_tool", "test description")
        fake_tools = {"test_tool": fake_tool}

        mock_client.side_effect = lambda url: _FakeMCPClient(tools=fake_tools, url=url)

        config = MCPToolConfig(transport="streamable-http",
                               mcp_tool_name="test_tool",
                               url=HttpUrl("http://localhost:8000/mcp"))

        # Mock builder (we don't need it for this test)
        mock_builder = MagicMock()

        async with mcp_tool(config, mock_builder) as fn_info:
            # Test that the function works
            result = await fn_info.single_fn(_InputSchema(param="test_value"))
            assert result == "ok test_value"
