# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from mcp.types import CallToolResult
from mcp.types import TextContent
from pydantic import BaseModel

from nat.builder.function import FunctionGroup
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.mcp.client.client_base import MCPBaseClient
from nat.plugins.mcp.client.client_base import MCPToolClient
from nat.plugins.mcp.client.client_config import MCPClientConfig
from nat.plugins.mcp.client.client_config import MCPServerConfig
from nat.plugins.mcp.client.client_config import MCPToolOverrideConfig
from nat.plugins.mcp.client.client_config import PerUserMCPClientConfig
from nat.plugins.mcp.client.client_impl import MCPFunctionGroup
from nat.plugins.mcp.client.client_impl import mcp_apply_tool_alias_and_description
from nat.plugins.mcp.client.client_impl import mcp_client_function_group
from nat.plugins.mcp.client.client_impl import mcp_per_user_tool_function
from nat.plugins.mcp.client.client_impl import mcp_session_tool_function
from nat.plugins.mcp.client.client_impl import per_user_mcp_client_function_group
from nat.plugins.mcp.exceptions import MCPError
from nat.plugins.mcp.exceptions import MCPErrorCategory
from nat.plugins.mcp.exceptions import MCPToolError


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
    with patch("nat.plugins.mcp.client.client_base.MCPStdioClient") as mock_client:
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
            assert set(accessible.keys()) == {f"mcp_client{FunctionGroup.SEPARATOR}fake_tool_1"}


async def test_mcp_client_function_group_applies_overrides():
    with patch("nat.plugins.mcp.client.client_base.MCPStdioClient") as mock_client:
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
            assert set(accessible.keys()) == {f"mcp_client{FunctionGroup.SEPARATOR}alias_raw"}
            assert accessible[f"mcp_client{FunctionGroup.SEPARATOR}alias_raw"].description == "new desc"


async def test_mcp_client_function_group_no_include_exposes_all():
    with patch("nat.plugins.mcp.client.client_base.MCPStdioClient") as mock_client:
        fake_tools = {"a": _FakeTool("a", "da"), "b": _FakeTool("b", "db")}
        mock_client.return_value = _FakeMCPClient(tools=fake_tools, command="python", args=["server.py"])

        server_cfg = MCPServerConfig(transport="stdio", command="python", args=["server.py"])
        client_cfg = MCPClientConfig(server=server_cfg)  # no include/exclude

        mock_builder = MagicMock(spec=WorkflowBuilder)

        async with mcp_client_function_group(client_cfg, mock_builder) as group:
            accessible = await group.get_accessible_functions()
            sep = FunctionGroup.SEPARATOR
            assert set(accessible.keys()) == {f"mcp_client{sep}a", f"mcp_client{sep}b"}


def _make_group(server_cfg=None, client_cfg=None):
    """Create an MCPFunctionGroup with sensible defaults for unit tests."""
    if server_cfg is None:
        server_cfg = MCPServerConfig(transport="stdio", command="python", args=["server.py"])
    if client_cfg is None:
        client_cfg = MCPClientConfig(server=server_cfg)
    return MCPFunctionGroup(config=client_cfg)


class TestSessionToolDefaultUserPath:
    """Tests for mcp_session_tool_function when routed through the default-user (base-client) path."""

    async def test_returns_unavailable_when_base_client_is_none(self):
        """mcp_client is None -> graceful unavailable message."""
        tool = _FakeTool("health")
        group = _make_group()
        group.mcp_client = None

        fn_info = mcp_session_tool_function(tool, group)
        result = await fn_info.single_fn(_InputSchema(param="x"))

        assert result == "Tool temporarily unavailable. Try again."

    async def test_returns_unavailable_when_base_client_disconnected(self):
        """mcp_client exists but is_connected is False -> graceful unavailable message.

        This is the scenario from the original bug: the client object is non-None
        but _exit_stack is None (e.g. after __aexit__ during shutdown).
        """
        tool = _FakeTool("health")
        group = _make_group()

        fake_client = _FakeMCPClient(tools={"health": _FakeTool("health")})
        group.mcp_client = fake_client
        assert not fake_client.is_connected

        fn_info = mcp_session_tool_function(tool, group)
        result = await fn_info.single_fn(_InputSchema(param="x"))

        assert result == "Tool temporarily unavailable. Try again."

    async def test_invokes_tool_when_connected(self):
        """Connected base client -> tool is invoked and result returned."""
        tool = _FakeTool("health")
        group = _make_group()

        fake_client = _FakeMCPClient(tools={"health": _FakeTool("health")})
        async with fake_client:
            group.mcp_client = fake_client
            assert fake_client.is_connected

            fn_info = mcp_session_tool_function(tool, group)
            result = await fn_info.single_fn(_InputSchema(param="ping"))

            assert result == "ok ping"


class TestSessionToolSessionPath:
    """Tests for mcp_session_tool_function when routed through the per-session path."""

    @pytest.fixture
    def session_group(self):
        """Create a group configured for the session path (auth provider present)."""
        group = _make_group()
        group._shared_auth_provider = MagicMock()
        group._default_user_id = "default-user"
        group._client_config = MagicMock()
        group._client_config.session_aware_tools = False
        return group

    async def test_returns_unavailable_when_session_client_is_none(self, session_group):
        """Session context yields None -> graceful unavailable message."""
        tool = _FakeTool("health")

        @asynccontextmanager
        async def fake_ctx(session_id):
            yield None

        with patch.object(session_group, '_get_session_id_from_context', return_value="sess-1"):
            with patch.object(session_group, '_session_usage_context', fake_ctx):
                fn_info = mcp_session_tool_function(tool, session_group)
                result = await fn_info.single_fn(_InputSchema(param="x"))

        assert result == "Tool temporarily unavailable. Try again."

    async def test_returns_unavailable_when_session_client_disconnected(self, session_group):
        """Session client exists but is_connected is False -> graceful unavailable message."""
        tool = _FakeTool("health")
        disconnected_client = _FakeMCPClient(tools={})

        @asynccontextmanager
        async def fake_ctx(session_id):
            yield disconnected_client

        with patch.object(session_group, '_get_session_id_from_context', return_value="sess-1"):
            with patch.object(session_group, '_session_usage_context', fake_ctx):
                fn_info = mcp_session_tool_function(tool, session_group)
                result = await fn_info.single_fn(_InputSchema(param="x"))

        assert result == "Tool temporarily unavailable. Try again."

    async def test_invokes_tool_when_connected(self, session_group):
        """Connected session client -> tool is invoked and result returned."""
        tool = _FakeTool("health")
        fake_client = _FakeMCPClient(tools={"health": _FakeTool("health")})

        @asynccontextmanager
        async def fake_ctx(session_id):
            async with fake_client:
                yield fake_client

        with patch.object(session_group, '_get_session_id_from_context', return_value="sess-1"):
            with patch.object(session_group, '_session_usage_context', fake_ctx):
                fn_info = mcp_session_tool_function(tool, session_group)
                result = await fn_info.single_fn(_InputSchema(param="ping"))

        assert result == "ok ping"

    async def test_tool_call_executes_inside_session_context(self, session_group):
        """Verify acall runs while the session usage context is still active.

        This guards against the original scoping bug where session_tool.acall()
        was invoked after the context manager had already decremented ref_count.
        """
        tool = _FakeTool("health")
        context_active = False
        acall_was_inside_context = None

        class _TrackingTool:
            name = "health"
            description = "desc"
            input_schema = _InputSchema

            async def acall(self, args):
                nonlocal acall_was_inside_context
                acall_was_inside_context = context_active
                return f"ok {args['param']}"

        fake_client = _FakeMCPClient(tools={"health": _TrackingTool()})

        @asynccontextmanager
        async def tracking_ctx(session_id):
            nonlocal context_active
            async with fake_client:
                context_active = True
                yield fake_client
            context_active = False

        with patch.object(session_group, '_get_session_id_from_context', return_value="sess-1"):
            with patch.object(session_group, '_session_usage_context', tracking_ctx):
                fn_info = mcp_session_tool_function(tool, session_group)
                result = await fn_info.single_fn(_InputSchema(param="ping"))

        assert result == "ok ping"
        assert acall_was_inside_context is True


def _failing_mcp_tool(name: str = "orders") -> MCPToolClient:
    """Create a real MCPToolClient whose server reports every call as failed (`isError` set)."""
    parent_client = MagicMock()
    parent_client.server_name = "stdio:server.py"
    parent_client.call_tool = AsyncMock(return_value=CallToolResult(
        content=[TextContent(type="text", text="Order ORD-404 does not exist")], isError=True))
    return MCPToolClient(session=MagicMock(),
                         parent_client=parent_client,
                         tool_name=name,
                         tool_description="desc",
                         tool_input_schema={
                             "type": "object", "properties": {
                                 "param": {
                                     "type": "string"
                                 }
                             }, "required": ["param"]
                         })


def _raising_server_config() -> MCPServerConfig:
    return MCPServerConfig(transport="stdio", command="python", args=["server.py"])


def _raising_config() -> MCPClientConfig:
    return MCPClientConfig(server=_raising_server_config(), raise_on_tool_error=True)


def test_raise_on_tool_error_defaults_to_false():
    server_cfg = MCPServerConfig(transport="stdio", command="python", args=["server.py"])
    assert MCPClientConfig(server=server_cfg).raise_on_tool_error is False
    assert PerUserMCPClientConfig(server=server_cfg).raise_on_tool_error is False


class TestSessionToolErrorHandling:
    """Tests for how mcp_session_tool_function reports failed tool calls."""

    async def test_tool_error_returned_as_text_by_default(self):
        group = _make_group()
        fake_client = _FakeMCPClient(tools={"orders": _failing_mcp_tool()})

        async with fake_client:
            group.mcp_client = fake_client
            fn_info = mcp_session_tool_function(_FakeTool("orders"), group)
            result = await fn_info.single_fn(_InputSchema(param="x"))

        assert result == "MCPToolClient tool call failed: Order ORD-404 does not exist"

    async def test_tool_error_raised_when_configured(self):
        group = _make_group(client_cfg=_raising_config())
        fake_client = _FakeMCPClient(tools={"orders": _failing_mcp_tool()})

        async with fake_client:
            group.mcp_client = fake_client
            fn_info = mcp_session_tool_function(_FakeTool("orders"), group)
            with pytest.raises(MCPToolError, match="Order ORD-404 does not exist") as exc_info:
                await fn_info.single_fn(_InputSchema(param="x"))

        assert "stdio:server.py" not in str(exc_info.value)

    async def test_unavailable_client_raised_when_configured(self):
        group = _make_group(client_cfg=_raising_config())
        group.mcp_client = None

        fn_info = mcp_session_tool_function(_FakeTool("orders"), group)
        with pytest.raises(MCPError, match="Tool temporarily unavailable") as exc_info:
            await fn_info.single_fn(_InputSchema(param="x"))

        assert exc_info.value.category == MCPErrorCategory.CONNECTION

    async def test_unavailable_session_client_raised_when_configured(self):
        group = _make_group(client_cfg=_raising_config())
        group._shared_auth_provider = MagicMock()
        group._default_user_id = "default-user"

        @asynccontextmanager
        async def no_client(session_id):
            yield None

        with patch.object(group, '_get_session_id_from_context', return_value="sess-1"):
            with patch.object(group, '_session_usage_context', no_client):
                fn_info = mcp_session_tool_function(_FakeTool("orders"), group)
                with pytest.raises(MCPError, match="Tool temporarily unavailable") as exc_info:
                    await fn_info.single_fn(_InputSchema(param="x"))

        assert exc_info.value.category == MCPErrorCategory.CONNECTION

    async def test_missing_session_raised_as_authentication_error_when_configured(self):
        group = _make_group(client_cfg=_raising_config())
        group._shared_auth_provider = MagicMock()

        with patch.object(group, '_get_session_id_from_context', return_value=None):
            fn_info = mcp_session_tool_function(_FakeTool("orders"), group)
            with pytest.raises(MCPError, match="User not authorized") as exc_info:
                await fn_info.single_fn(_InputSchema(param="x"))

        assert exc_info.value.category == MCPErrorCategory.AUTHENTICATION

    async def test_function_group_applies_raise_on_tool_error(self):
        """The config flag reaches the functions that mcp_client_function_group adds to the group."""
        with patch("nat.plugins.mcp.client.client_base.MCPStdioClient") as mock_client:
            mock_client.return_value = _FakeMCPClient(tools={"orders": _failing_mcp_tool()},
                                                      command="python",
                                                      args=["server.py"])

            async with mcp_client_function_group(_raising_config(), MagicMock(spec=WorkflowBuilder)) as group:
                accessible = await group.get_accessible_functions()
                fn = accessible[f"mcp_client{FunctionGroup.SEPARATOR}orders"]
                with pytest.raises(MCPToolError):
                    await fn.acall_invoke(param="x")


class TestPerUserToolErrorHandling:
    """Tests for how mcp_per_user_tool_function reports failed tool calls."""

    async def test_tool_error_returned_as_text_by_default(self):
        fn_info = mcp_per_user_tool_function(_FakeTool("orders"), _FakeMCPClient(tools={"orders": _failing_mcp_tool()}))

        result = await fn_info.single_fn(_InputSchema(param="x"))

        assert result == "MCPToolClient tool call failed: Order ORD-404 does not exist"

    async def test_tool_error_raised_when_configured(self):
        fn_info = mcp_per_user_tool_function(_FakeTool("orders"),
                                             _FakeMCPClient(tools={"orders": _failing_mcp_tool()}),
                                             raise_on_tool_error=True)

        with pytest.raises(MCPToolError, match="Order ORD-404 does not exist"):
            await fn_info.single_fn(_InputSchema(param="x"))

    async def test_function_group_applies_raise_on_tool_error(self):
        """The config flag reaches the functions that per_user_mcp_client_function_group adds to the group."""
        with patch("nat.plugins.mcp.client.client_base.MCPStdioClient") as mock_client:
            mock_client.return_value = _FakeMCPClient(tools={"orders": _failing_mcp_tool()},
                                                      command="python",
                                                      args=["server.py"])
            config = PerUserMCPClientConfig(server=_raising_server_config(), raise_on_tool_error=True)

            async with per_user_mcp_client_function_group(config, MagicMock(spec=WorkflowBuilder)) as group:
                accessible = await group.get_accessible_functions()
                fn = next(f for name, f in accessible.items() if name.endswith(f"{FunctionGroup.SEPARATOR}orders"))
                with pytest.raises(MCPToolError):
                    await fn.acall_invoke(param="x")
