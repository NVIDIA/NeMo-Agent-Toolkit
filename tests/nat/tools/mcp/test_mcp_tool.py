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

"""
Functional tests for src/nat/tool/mcp/mcp_tool.py
Focus on behavior (config validation, client initialization, tool wrapping, error handling), not class/type basics.
"""

import json
from typing import Any
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from pydantic import BaseModel

from nat.builder.workflow_builder import WorkflowBuilder
from nat.tool.mcp.mcp_tool import MCPToolConfig


class _InputSchema(BaseModel):
    param: str


class _FakeTool:
    def __init__(self, name: str, description: str = "desc") -> None:
        self.name = name
        self.description = description
        self.input_schema = _InputSchema

    async def acall(self, args: dict[str, Any]) -> str:
        return f"ok {args['param']}"

    def set_description(self, description: str) -> None:
        if description is not None:
            self.description = description


class _ErrorTool(_FakeTool):
    async def acall(self, args: dict[str, Any]) -> str:  # type: ignore[override]
        raise RuntimeError("boom")


class _FakeMCPClient:
    def __init__(self, *, tools: dict[str, _FakeTool], transport: str = "sse", url: str | None = None,
                 command: str | None = None) -> None:
        self._tools = tools
        self.transport = transport
        self.url = url
        self.command = command
        self.server_name = f"{transport}:{url or command}"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_tool(self, name: str) -> _FakeTool:
        return self._tools[name]


def test_mcp_tool_config_validation_stdio_requires_command():
    """Test that stdio transport requires command parameter."""
    with pytest.raises(ValueError, match="command is required"):
        MCPToolConfig(
            transport="stdio",
            mcp_tool_name="test_tool"
            # Missing command
        )


def test_mcp_tool_config_validation_stdio_rejects_url():
    """Test that stdio transport rejects URL parameter."""
    with pytest.raises(ValueError, match="url should not be set"):
        MCPToolConfig(
            transport="stdio",
            mcp_tool_name="test_tool",
            command="python",
            url="http://example.com"  # Should not be set for stdio
        )


def test_mcp_tool_config_validation_http_requires_url():
    """Test that HTTP transports require URL parameter."""
    with pytest.raises(ValueError, match="url is required"):
        MCPToolConfig(
            transport="streamable-http",
            mcp_tool_name="test_tool"
            # Missing url
        )


def test_mcp_tool_config_validation_http_rejects_stdio_params():
    """Test that HTTP transports reject stdio parameters."""
    with pytest.raises(ValueError, match="command, args, and env should not be set"):
        MCPToolConfig(
            transport="sse",
            mcp_tool_name="test_tool",
            url="http://example.com",
            command="python",  # Should not be set for HTTP
            args=["script.py"],  # Should not be set for HTTP
            env={"DEBUG": "1"}  # Should not be set for HTTP
        )


def test_mcp_tool_config_validation_valid_stdio():
    """Test that valid stdio config passes validation."""
    config = MCPToolConfig(
        transport="stdio",
        mcp_tool_name="test_tool",
        command="python",
        args=["script.py"],
        env={"DEBUG": "1"}
    )
    assert config.transport == "stdio"
    assert config.command == "python"
    assert config.args == ["script.py"]
    assert config.env == {"DEBUG": "1"}


def test_mcp_tool_config_validation_valid_http():
    """Test that valid HTTP config passes validation."""
    config = MCPToolConfig(
        transport="streamable-http",
        mcp_tool_name="test_tool",
        url="http://example.com"
    )
    assert config.transport == "streamable-http"
    assert config.url == "http://example.com"


def test_mcp_tool_config_defaults():
    """Test that MCPToolConfig has correct defaults."""
    config = MCPToolConfig(
        transport="streamable-http",
        mcp_tool_name="test_tool",
        url="http://example.com"
    )
    assert config.transport == "streamable-http"
    assert config.return_exception is True
    assert config.description is None


@patch("nat.tool.mcp.mcp_client_base.MCPSSEClient")
async def test_mcp_tool_wraps_tool_with_description_override(mock_mcp_sse_client):
    """Test that mcp_tool function wraps tool and applies description override."""
    fake_tool = _FakeTool("test_tool", "original description")

    def mock_client_factory(*args, **kwargs):
        return _FakeMCPClient(tools={"test_tool": fake_tool}, transport="sse", url="http://example.com")

    mock_mcp_sse_client.side_effect = mock_client_factory

    config = MCPToolConfig(
        transport="sse",
        mcp_tool_name="test_tool",
        url="http://example.com",
        description="overridden description"
    )

    async with WorkflowBuilder() as builder:
        fn = await builder.add_function(name="test_fn", config=config)

        # Verify the tool description was overridden
        assert fn.description == "overridden description"

        # Verify the tool is callable
        result = await fn.acall_invoke(param="test_value")
        assert result == "ok test_value"


@patch("nat.tool.mcp.mcp_client_base.MCPSSEClient")
async def test_mcp_tool_returns_exception_string_when_enabled(mock_mcp_sse_client):
    """Test that mcp_tool returns exception string when return_exception=True."""
    error_tool = _ErrorTool("error_tool", "error tool")

    def mock_client_factory(*args, **kwargs):
        return _FakeMCPClient(tools={"error_tool": error_tool}, transport="sse", url="http://example.com")

    mock_mcp_sse_client.side_effect = mock_client_factory

    config = MCPToolConfig(
        transport="sse",
        mcp_tool_name="error_tool",
        url="http://example.com",
        return_exception=True
    )

    async with WorkflowBuilder() as builder:
        fn = await builder.add_function(name="error_fn", config=config)

        # Should return error string instead of raising
        result = await fn.acall_invoke(param="test_value")
        assert "boom" in result


@patch("nat.tool.mcp.mcp_client_base.MCPSSEClient")
async def test_mcp_tool_raises_exception_when_disabled(mock_mcp_sse_client):
    """Test that mcp_tool raises exception when return_exception=False."""
    error_tool = _ErrorTool("error_tool", "error tool")

    def mock_client_factory(*args, **kwargs):
        return _FakeMCPClient(tools={"error_tool": error_tool}, transport="sse", url="http://example.com")

    mock_mcp_sse_client.side_effect = mock_client_factory

    config = MCPToolConfig(
        transport="sse",
        mcp_tool_name="error_tool",
        url="http://example.com",
        return_exception=False
    )

    async with WorkflowBuilder() as builder:
        fn = await builder.add_function(name="error_fn", config=config)

        # Should raise exception instead of returning string
        with pytest.raises(RuntimeError, match="boom"):
            await fn.acall_invoke(param="test_value")


@patch("nat.tool.mcp.mcp_client_base.MCPSSEClient")
async def test_mcp_tool_supports_both_input_methods(mock_mcp_sse_client):
    """Test that mcp_tool supports both tool_input and kwargs input methods."""
    fake_tool = _FakeTool("test_tool", "test description")

    def mock_client_factory(*args, **kwargs):
        return _FakeMCPClient(tools={"test_tool": fake_tool}, transport="sse", url="http://example.com")

    mock_mcp_sse_client.side_effect = mock_client_factory

    config = MCPToolConfig(
        transport="sse",
        mcp_tool_name="test_tool",
        url="http://example.com"
    )

    async with WorkflowBuilder() as builder:
        fn = await builder.add_function(name="test_fn", config=config)

        # Test kwargs input method
        result1 = await fn.acall_invoke(param="value1")
        assert result1 == "ok value1"

        # Test tool_input method (using the converter)
        input_data = _InputSchema(param="value2")
        result2 = await fn.acall_invoke(tool_input=input_data)
        assert result2 == "ok value2"


def test_mcp_tool_invalid_transport_raises_error():
    """Test that invalid transport type raises ValueError."""
    with pytest.raises(ValueError, match="Input should be 'sse', 'stdio' or 'streamable-http'"):
        MCPToolConfig(
            transport="invalid",  # type: ignore[assignment]
            mcp_tool_name="test_tool",
            url="http://example.com"
        )


@patch("nat.tool.mcp.mcp_client_base.MCPStdioClient")
async def test_mcp_tool_stdio_client_initialization(mock_mcp_stdio_client):
    """Test that stdio client is properly initialized with command, args, and env."""
    fake_tool = _FakeTool("test_tool", "test description")

    def mock_stdio_client(command, args, env):
        assert command == "python"
        assert args == ["script.py"]
        assert env == {"DEBUG": "1"}
        return _FakeMCPClient(tools={"test_tool": fake_tool}, transport="stdio", command=command)

    mock_mcp_stdio_client.side_effect = mock_stdio_client

    config = MCPToolConfig(
        transport="stdio",
        mcp_tool_name="test_tool",
        command="python",
        args=["script.py"],
        env={"DEBUG": "1"}
    )

    async with WorkflowBuilder() as builder:
        fn = await builder.add_function(name="test_fn", config=config)

        # Verify the tool works
        result = await fn.acall_invoke(param="test_value")
        assert result == "ok test_value"


@patch("nat.tool.mcp.mcp_client_base.MCPStreamableHTTPClient")
async def test_mcp_tool_streamable_http_client_initialization(mock_mcp_streamable_http_client):
    """Test that streamable-http client is properly initialized with URL."""
    fake_tool = _FakeTool("test_tool", "test description")

    def mock_streamable_http_client(url):
        assert url == "http://example.com"
        return _FakeMCPClient(tools={"test_tool": fake_tool}, transport="streamable-http", url=url)

    mock_mcp_streamable_http_client.side_effect = mock_streamable_http_client

    config = MCPToolConfig(
        transport="streamable-http",
        mcp_tool_name="test_tool",
        url="http://example.com"
    )

    async with WorkflowBuilder() as builder:
        fn = await builder.add_function(name="test_fn", config=config)

        # Verify the tool works
        result = await fn.acall_invoke(param="test_value")
        assert result == "ok test_value"
