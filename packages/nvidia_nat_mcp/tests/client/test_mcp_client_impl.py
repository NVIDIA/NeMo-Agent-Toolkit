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

from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.builder.function import FunctionGroup
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.mcp.client.client_base import MCPBaseClient
from nat.plugins.mcp.client.client_config import MCPClientConfig
from nat.plugins.mcp.client.client_config import MCPServerConfig
from nat.plugins.mcp.client.client_config import MCPToolOverrideConfig
from nat.plugins.mcp.client.client_impl import MCPFunctionGroup
from nat.plugins.mcp.client.client_impl import mcp_apply_tool_alias_and_description
from nat.plugins.mcp.client.client_impl import mcp_client_function_group
from nat.plugins.mcp.client.client_impl import mcp_session_tool_function


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


_NULLABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "issue_id": {
            "type": "string"
        },
        "assignee": {
            "type": ["string", "null"]
        },
        "note": {
            "type": "string"
        },
        "filter": {
            "type": "object",
            "properties": {
                "label": {
                    "anyOf": [{
                        "type": "string"
                    }, {
                        "type": "null"
                    }]
                }, "limit": {
                    "type": "integer"
                }
            },
            "required": ["label"],
        },
        "rows": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "v": {
                            "type": "integer"
                        }, "note": {
                            "type": "string"
                        }
                    },
                    "required": ["v"],
                },
            },
        },
    },
    "required": ["issue_id", "assignee"],
}

# Required fields that also declare a default, at the top level and in a nested object.
_REQUIRED_WITH_DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "issue_id": {
            "type": "string"
        },
        "reviewer": {
            "type": ["string", "null"], "default": None
        },
        "filter": {
            "type": "object",
            "properties": {
                "label": {
                    "type": ["string", "null"], "default": None
                },
                "limit": {
                    "type": ["integer", "null"]
                },
            },
            "required": ["label"],
        },
    },
    "required": ["issue_id", "reviewer"],
}

_REF_SCHEMA = {
    "type": "object",
    "$defs": {
        "Filter": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string"
                },
                "limit": {
                    "type": "integer"
                },
            },
            "required": ["label"],
        }
    },
    "properties": {
        "issue_id": {
            "type": "string"
        }, "filter": {
            "$ref": "#/$defs/Filter"
        }
    },
    "required": ["issue_id", "filter"],
}

_DISCRIMINATED_UNION_SCHEMA = {
    "type": "object",
    "$defs": {
        "Cat": {
            "type": "object",
            "properties": {
                "pet_type": {
                    "const": "cat", "type": "string"
                },
                "name": {
                    "type": "string"
                },
                "lives": {
                    "default": 9, "type": "integer"
                }
            },
            "required": ["pet_type", "name"],
            "additionalProperties": False,
        },
        "Dog": {
            "type": "object",
            "properties": {
                "pet_type": {
                    "const": "dog", "type": "string"
                },
                "name": {
                    "type": "string"
                },
                "good_boy": {
                    "default": True, "type": "boolean"
                },
                "lives": {
                    "default": 1, "type": "integer"
                },
            },
            "required": ["pet_type", "name"],
            "additionalProperties": False,
        },
    },
    "properties": {
        "pet": {
            "discriminator": {
                "mapping": {
                    "cat": "#/$defs/Cat", "dog": "#/$defs/Dog"
                }, "propertyName": "pet_type"
            },
            "oneOf": [{
                "$ref": "#/$defs/Cat"
            }, {
                "$ref": "#/$defs/Dog"
            }],
        }
    },
    "required": ["pet"],
}


class _RecordingTool:
    """Fake MCP tool that records the arguments sent to the server."""

    def __init__(self, schema: dict[str, Any]) -> None:
        from nat.plugins.mcp.utils import model_from_mcp_schema

        self.name = "update_issue"
        self.description = "desc"
        self.input_schema = model_from_mcp_schema("update_issue", schema)
        self.sent: dict[str, Any] | None = None

    async def acall(self, args: dict[str, Any]) -> str:
        self.sent = args
        return "ok"


async def _call_tool_fn(fn_info, tool: _RecordingTool, tool_args: dict[str, Any]) -> None:
    # Same path as an agent tool call: the arguments are validated into the input schema, and FunctionInfo
    # unpacks the model into keyword arguments for the tool function.
    result = await fn_info.single_fn(tool.input_schema.model_validate(tool_args))
    assert result == "ok"


async def _send_through_per_user_tool(tool_args: dict[str, Any],
                                      schema: dict[str, Any] = _NULLABLE_SCHEMA) -> dict[str, Any] | None:
    from nat.plugins.mcp.client.client_impl import mcp_per_user_tool_function

    tool = _RecordingTool(schema)
    client = MagicMock()

    async def get_tool(name):
        return tool

    client.get_tool = get_tool
    await _call_tool_fn(mcp_per_user_tool_function(tool, client), tool, tool_args)
    return tool.sent


async def _send_through_session_tool(tool_args: dict[str, Any],
                                     schema: dict[str, Any] = _NULLABLE_SCHEMA) -> dict[str, Any] | None:
    tool = _RecordingTool(schema)
    group = _make_group()
    fake_client = _FakeMCPClient(tools={"update_issue": tool})
    async with fake_client:
        group.mcp_client = fake_client
        await _call_tool_fn(mcp_session_tool_function(tool, group), tool, tool_args)
    return tool.sent


@pytest.fixture(name="send_tool_args", params=["per_user", "session"])
def fixture_send_tool_args(request):
    return _send_through_per_user_tool if request.param == "per_user" else _send_through_session_tool


async def test_required_nullable_argument_keeps_explicit_null(send_tool_args):
    """A required field must be present in the call even when its value is null (JSON Schema `required`)."""
    sent = await send_tool_args({"issue_id": "ISS-1", "assignee": None})
    assert sent == {"issue_id": "ISS-1", "assignee": None}


async def test_nested_required_nullable_argument_keeps_explicit_null(send_tool_args):
    sent = await send_tool_args({"issue_id": "ISS-1", "assignee": "bob", "filter": {"label": None}})
    assert sent == {"issue_id": "ISS-1", "assignee": "bob", "filter": {"label": None}}


async def test_optional_arguments_left_unset_are_not_sent(send_tool_args):
    """Optional fields the model did not fill are left out, as many MCP servers reject extra nulls."""
    sent = await send_tool_args({"issue_id": "ISS-1", "assignee": "bob", "filter": {"label": "bug"}})
    assert sent == {"issue_id": "ISS-1", "assignee": "bob", "filter": {"label": "bug"}}


async def test_optional_arguments_left_unset_in_nested_arrays_are_not_sent(send_tool_args):
    sent = await send_tool_args({"issue_id": "ISS-1", "assignee": "bob", "rows": [[{"v": 1}], [{"v": 2}]]})
    assert sent == {"issue_id": "ISS-1", "assignee": "bob", "rows": [[{"v": 1}], [{"v": 2}]]}


async def test_required_argument_with_default_is_sent_when_left_unset(send_tool_args):
    """A schema default does not lift `required`: the key is still sent, carrying the default."""
    sent = await send_tool_args({"issue_id": "ISS-1"}, _REQUIRED_WITH_DEFAULT_SCHEMA)
    assert sent == {"issue_id": "ISS-1", "reviewer": None}


async def test_nested_required_argument_with_default_keeps_explicit_null(send_tool_args):
    tool_args = {"issue_id": "ISS-1", "reviewer": None, "filter": {"label": None}}
    sent = await send_tool_args(tool_args, _REQUIRED_WITH_DEFAULT_SCHEMA)
    assert sent == {"issue_id": "ISS-1", "reviewer": None, "filter": {"label": None}}


async def test_referenced_nested_argument_is_sent_as_an_object(send_tool_args: Callable[...,
                                                                                        Awaitable[dict[str, Any]
                                                                                                  | None]]):
    """A `$ref` argument (FastMCP nested model) goes out as a plain object, without the unset optional field."""
    sent = await send_tool_args({"issue_id": "ISS-1", "filter": {"label": "bug"}}, _REF_SCHEMA)
    assert sent == {"issue_id": "ISS-1", "filter": {"label": "bug"}}


async def test_discriminated_union_argument_is_sent_as_the_matching_variant(
        send_tool_args: Callable[..., Awaitable[dict[str, Any] | None]]):
    """The `const` tag selects the variant, so the call carries that variant's fields and defaults."""
    sent = await send_tool_args({"pet": {"pet_type": "dog", "name": "Rex"}}, _DISCRIMINATED_UNION_SCHEMA)
    assert sent == {"pet": {"pet_type": "dog", "name": "Rex", "good_boy": True, "lives": 1}}
