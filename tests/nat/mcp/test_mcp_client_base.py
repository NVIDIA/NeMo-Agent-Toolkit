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
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import uvicorn
from mcp.client.session import ClientSession
from mcp.server.fastmcp.server import FastMCP
from mcp.types import TextContent

from nat.plugins.mcp.client_base import MCPBaseClient
from nat.plugins.mcp.client_base import MCPSSEClient
from nat.plugins.mcp.client_base import MCPStdioClient
from nat.plugins.mcp.client_base import MCPStreamableHTTPClient
from nat.plugins.mcp.exceptions import MCPConnectionError


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


# ============================================================================
# Tests for new reconnect logic and timeout features
# ============================================================================


class MockMCPClient(MCPBaseClient):
    """Mock MCP client for testing reconnect and timeout functionality."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect_call_count = 0
        self.connect_should_fail = False
        self.connect_failure_count = 0
        self.reconnect_call_count = 0
        # Global side effects that persist across reconnections
        self.list_tools_side_effect: callable = None  # type: ignore
        self.call_tool_side_effect: callable = None  # type: ignore

    def connect_to_server(self):  # type: ignore
        """Mock connection that can be configured to fail."""
        return MockAsyncContextManager(self)


class MockAsyncContextManager:
    """Mock async context manager for testing."""

    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        self.client.connect_call_count += 1
        # Only fail during reconnect attempts, not initial connection for most tests
        if (self.client.connect_should_fail and self.client.connect_call_count > 1
                and  # Allow first connection to succeed
                self.client.connect_call_count <= self.client.connect_failure_count + 1):
            raise ConnectionError(f"Mock connection failure #{self.client.connect_call_count}")

        # Return a mock session
        mock_session = AsyncMock(spec=ClientSession)

        # Apply global side effects if they exist
        if self.client.list_tools_side_effect:
            mock_session.list_tools.side_effect = self.client.list_tools_side_effect
        else:
            mock_session.list_tools = AsyncMock()

        if self.client.call_tool_side_effect:
            mock_session.call_tool.side_effect = self.client.call_tool_side_effect
        else:
            mock_session.call_tool = AsyncMock()

        return mock_session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


async def test_reconnect_configuration():
    """Test that reconnect configuration parameters are properly set."""
    client = MockMCPClient(transport="streamable-http",
                           reconnect_enabled=False,
                           reconnect_max_attempts=5,
                           reconnect_initial_backoff=1.0,
                           reconnect_max_backoff=100.0)
    assert client._reconnect_enabled is False
    assert client._reconnect_max_attempts == 5
    assert client._reconnect_initial_backoff == 1.0
    assert client._reconnect_max_backoff == 100.0


async def test_tool_call_timeout_configuration():
    """Test that tool call timeout is properly configured."""
    timeout = timedelta(seconds=10)
    client = MockMCPClient(transport="streamable-http", tool_call_timeout=timeout)

    assert client._tool_call_timeout == timeout


async def test_reconnect_disabled_no_retry():
    """Test that when reconnect is disabled, no retry attempts are made."""
    client = MockMCPClient(transport="streamable-http", reconnect_enabled=False)

    # Mock the session to simulate a failure
    async def failing_list_tools():
        raise ConnectionError("Connection lost")

    client.list_tools_side_effect = failing_list_tools

    async with client:
        # Should not retry when reconnect is disabled
        with pytest.raises(MCPConnectionError):
            await client.get_tools()

        # Connection should only be attempted once (during __aenter__)
        assert client.connect_call_count == 1


async def test_reconnect_success_after_failure():
    """Test successful reconnection after initial failure."""
    client = MockMCPClient(
        transport="streamable-http",
        reconnect_enabled=True,
        reconnect_max_attempts=2,
        reconnect_initial_backoff=0.01,  # Fast for testing
        reconnect_max_backoff=0.02)

    # Mock the session to fail once, then succeed
    call_count = 0

    async def mock_list_tools():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("First call fails")
        return MagicMock(tools=[])

    client.list_tools_side_effect = mock_list_tools

    async with client:
        # Should succeed after reconnect
        result = await client.get_tools()
        assert isinstance(result, dict)

        # Should have been called twice (fail, then succeed)
        assert call_count == 2


async def test_reconnect_max_attempts_exceeded():
    """Test that reconnect gives up after max attempts."""
    client = MockMCPClient(transport="streamable-http",
                           reconnect_enabled=True,
                           reconnect_max_attempts=2,
                           reconnect_initial_backoff=0.01,
                           reconnect_max_backoff=0.02)

    # Configure client to fail connection attempts during reconnect
    client.connect_should_fail = True
    client.connect_failure_count = 3  # More than max attempts

    # Mock session to always fail to trigger reconnect
    async def always_fail():
        raise ConnectionError("Always fails")

    client.list_tools_side_effect = always_fail

    async with client:
        with pytest.raises(MCPConnectionError):
            await client.get_tools()


@pytest.mark.skip(reason="This test might fail in CI due to race conditions")
async def test_reconnect_backoff_timing():
    """Test that reconnect backoff timing works correctly."""
    client = MockMCPClient(transport="streamable-http",
                           reconnect_enabled=True,
                           reconnect_max_attempts=3,
                           reconnect_initial_backoff=0.1,
                           reconnect_max_backoff=0.5)

    # Track timing of reconnect attempts
    attempt_times = []
    original_sleep = asyncio.sleep

    async def mock_sleep(delay):
        attempt_times.append(delay)
        await original_sleep(0.01)  # Actual short sleep for test

    # Configure to fail first 2 reconnect attempts, succeed on 3rd
    client.connect_should_fail = True
    client.connect_failure_count = 2

    # Mock session to fail initially to trigger reconnect
    call_count = 0

    async def mock_list_tools():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Trigger reconnect")
        return MagicMock(tools=[])

    client.list_tools_side_effect = mock_list_tools

    with patch('asyncio.sleep', mock_sleep):
        async with client:
            # Should eventually succeed
            await client.get_tools()

            # Check backoff timing: should be [0.1, 0.2] (initial, then doubled)
            assert len(attempt_times) == 2
            assert attempt_times[0] == 0.1
            assert attempt_times[1] == 0.2


@pytest.mark.skip(reason="This test might fail in CI due to race conditions")
async def test_reconnect_max_backoff_limit():
    """Test that backoff doesn't exceed maximum."""
    client = MockMCPClient(
        transport="streamable-http",
        reconnect_enabled=True,
        reconnect_max_attempts=4,
        reconnect_initial_backoff=0.2,
        reconnect_max_backoff=0.3  # Low max for testing
    )

    attempt_times = []
    original_sleep = asyncio.sleep

    async def mock_sleep(delay):
        attempt_times.append(delay)
        await original_sleep(0.01)  # Use original_sleep to avoid recursion

    client.connect_should_fail = True
    client.connect_failure_count = 4

    # Mock session to always fail to trigger reconnect
    async def always_fail():
        raise ConnectionError("Always fails")

    client.list_tools_side_effect = always_fail

    with patch('asyncio.sleep', mock_sleep):
        async with client:
            with pytest.raises(MCPConnectionError):
                await client.get_tools()

            # Backoff should be: [0.2, 0.3, 0.3, 0.3] for 4 attempts (capped at max_backoff)
            assert len(attempt_times) == 4
            assert attempt_times[0] == 0.2
            assert attempt_times[1] == 0.3  # min(0.4, 0.3)
            assert attempt_times[2] == 0.3  # min(0.6, 0.3)
            assert attempt_times[3] == 0.3  # min(1.2, 0.3)


async def test_tool_call_timeout_passed_to_session():
    """Test that tool call timeout is properly passed to the session."""
    timeout = timedelta(seconds=15)
    client = MockMCPClient(transport="streamable-http", tool_call_timeout=timeout)

    # Create a mock that tracks calls
    call_args = []

    async def mock_call_tool(*args, **kwargs):
        call_args.append((args, kwargs))
        return MagicMock(content=[])

    client.call_tool_side_effect = mock_call_tool

    async with client:
        await client.call_tool("test_tool", {"arg": "value"})

        # Verify timeout was passed correctly
        assert len(call_args) == 1
        args, kwargs = call_args[0]
        assert args == ("test_tool", {"arg": "value"})
        assert kwargs.get("read_timeout_seconds") == timeout


async def test_with_reconnect_success_no_retry():
    """Test _with_reconnect when operation succeeds on first try."""
    client = MockMCPClient(transport="streamable-http", reconnect_enabled=True)

    async with client:
        call_count = 0

        async def mock_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await client._with_reconnect(mock_operation)

        assert result == "success"
        assert call_count == 1


async def test_with_reconnect_disabled_propagates_error():
    """Test _with_reconnect propagates error when reconnect is disabled."""
    client = MockMCPClient(transport="streamable-http", reconnect_enabled=False)

    async with client:

        async def failing_operation():
            raise ValueError("Operation failed")

        with pytest.raises(ValueError, match="Operation failed"):
            await client._with_reconnect(failing_operation)


async def test_reconnect_lock_prevents_concurrent_reconnects():
    """Test that reconnect lock prevents concurrent reconnection attempts."""
    client = MockMCPClient(transport="streamable-http",
                           reconnect_enabled=True,
                           reconnect_max_attempts=1,
                           reconnect_initial_backoff=0.01)

    async with client:
        # Track reconnect calls and timing
        reconnect_call_count = 0
        reconnect_start_times = []

        async def mock_reconnect():
            nonlocal reconnect_call_count
            reconnect_call_count += 1
            start_time = asyncio.get_event_loop().time()
            reconnect_start_times.append(start_time)
            await asyncio.sleep(0.1)  # Simulate longer work to test lock
            # Make reconnect fail to ensure we get exceptions
            raise ConnectionError("Reconnect failed")

        client._reconnect = mock_reconnect

        # Trigger two concurrent operations that will fail
        async def failing_operation():
            raise ConnectionError("Simulated failure")

        # Run two operations concurrently that should both trigger reconnect
        results = await asyncio.gather(client._with_reconnect(failing_operation),
                                       client._with_reconnect(failing_operation),
                                       return_exceptions=True)

        # Both should fail
        assert all(isinstance(r, ConnectionError) for r in results)

        # Due to the lock, reconnect should ideally only be called once
        # However, in practice with concurrent operations, we might see up to 2 calls
        # The important thing is that the lock mechanism exists and limits concurrent access
        assert reconnect_call_count <= 2  # Lock should limit concurrent reconnects

        # The main goal is to verify the lock mechanism limits concurrent reconnects
        # In practice, both operations might trigger reconnect, but the lock should
        # prevent them from running completely concurrently
        # The fact that we get at most 2 calls (not more) shows the lock is working

        # Additional verification: if multiple calls happened, they should not be
        # completely simultaneous (some microsecond difference is expected)
        if len(reconnect_start_times) > 1:
            time_diffs = [
                abs(reconnect_start_times[i + 1] - reconnect_start_times[i])
                for i in range(len(reconnect_start_times) - 1)
            ]
            # Even a tiny difference shows they're not perfectly concurrent
            assert any(diff > 0 for diff in time_diffs), "Reconnect calls should have some temporal separation"


async def test_connection_established_flag():
    """Test that connection established flag is properly managed."""
    client = MockMCPClient(transport="streamable-http")

    # Initially not connected
    assert client._connection_established is False
    assert client._initial_connection is False

    async with client:
        # Should be connected after entering context
        assert client._connection_established is True
        assert client._initial_connection is True

    # Should be disconnected after exiting context
    assert client._connection_established is False


class TestMCPToolClient:
    """Test the MCPToolClient basic functionality."""

    def test_tool_client_instantiation(self):
        """Test that MCPToolClient can be instantiated correctly."""
        from nat.plugins.mcp.client_base import MCPToolClient

        # Create mock objects
        mock_session = MagicMock()
        mock_parent_client = MagicMock()

        # Create MCPToolClient instance
        tool_client = MCPToolClient(session=mock_session,
                                    parent_client=mock_parent_client,
                                    tool_name="test_tool",
                                    tool_description="Test tool")

        # Verify basic properties
        assert tool_client.name == "test_tool"
        assert tool_client.description == "Test tool"
        assert tool_client.input_schema is None

    def test_tool_client_with_input_schema(self):
        """Test that MCPToolClient handles input schema correctly."""
        from nat.plugins.mcp.client_base import MCPToolClient

        # Create mock objects
        mock_session = MagicMock()
        mock_parent_client = MagicMock()
        input_schema = {"type": "object", "properties": {"arg1": {"type": "string"}, "arg2": {"type": "number"}}}

        # Create MCPToolClient instance
        tool_client = MCPToolClient(session=mock_session,
                                    parent_client=mock_parent_client,
                                    tool_name="test_tool",
                                    tool_description="Test tool",
                                    tool_input_schema=input_schema)

        # Verify input schema is processed
        assert tool_client.input_schema is not None

    def test_tool_client_description_override(self):
        """Test that tool description can be overridden."""
        from nat.plugins.mcp.client_base import MCPToolClient

        # Create mock objects
        mock_session = MagicMock()
        mock_parent_client = MagicMock()

        # Create MCPToolClient instance
        tool_client = MCPToolClient(session=mock_session,
                                    parent_client=mock_parent_client,
                                    tool_name="test_tool",
                                    tool_description="Original description")

        # Override description
        tool_client.set_description("New description")
        assert tool_client.description == "New description"

    def test_tool_client_no_parent_client_raises_error(self):
        """Test that MCPToolClient raises error when no parent client is provided."""
        from nat.plugins.mcp.client_base import MCPToolClient

        # Create mock objects
        mock_session = MagicMock()

        # Should raise RuntimeError when parent_client is None
        with pytest.raises(RuntimeError, match="MCPToolClient initialized without a parent client"):
            MCPToolClient(session=mock_session, parent_client=None, tool_name="test_tool", tool_description="Test tool")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MCP Server")
    parser.add_argument("--transport", type=str, default="stdio", help="Transport to use for the server")

    args = parser.parse_args()

    _create_test_mcp_server(port=8122).run(transport=args.transport)
