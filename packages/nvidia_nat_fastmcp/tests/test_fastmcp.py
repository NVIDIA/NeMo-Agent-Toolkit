# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for FastMCP CLI and server wiring."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp import FastMCP
from fastmcp.server.auth import RemoteAuthProvider
from pydantic import BaseModel
from pydantic import Field
from pydantic import SecretStr
from starlette.testclient import TestClient
from watchfiles import Change

from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from nat.builder.function_base import FunctionBase
from nat.data_models.authentication import TokenValidationResult
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.plugins.fastmcp.cli.commands import fastmcp_command  # pylint: disable=import-error,no-name-in-module
from nat.plugins.fastmcp.cli.utils import _filter_change_set
from nat.plugins.fastmcp.cli.utils import iter_file_changes
from nat.plugins.fastmcp.server.front_end_config import FastMCPFrontEndConfig
from nat.plugins.fastmcp.server.front_end_plugin_worker import FastMCPFrontEndPluginWorker
from nat.plugins.fastmcp.server.token_verifier import NATFastMCPTokenVerifier


class _MockTestSchema(BaseModel):
    text: str | None = None
    number: int = 42


class _ChatRequestSchema(BaseModel):
    messages: list = Field(default_factory=list)
    model: str | None = None


class _RegularFunction(FunctionBase[str, str, str]):
    description = "Regular function description"

    def __init__(self):
        super().__init__(input_schema=_MockTestSchema)

    async def _ainvoke(self, value: str) -> str:
        return value

    async def _astream(self, value: str):
        yield value


class _ChatRequestFunction(FunctionBase[str, str, str]):
    description = "Chat request function description"

    def __init__(self):
        super().__init__(input_schema=_ChatRequestSchema)

    async def _ainvoke(self, value: str) -> str:
        return value

    async def _astream(self, value: str):
        yield value


class _NoSchemaFunction(FunctionBase[str, str, str]):
    description = "Function without schema"

    def __init__(self):
        super().__init__(input_schema=None)

    async def _ainvoke(self, value: str) -> str:
        return value

    async def _astream(self, value: str):
        yield value


def test_fastmcp_cli_groups() -> None:
    """Ensure FastMCP CLI groups and commands are registered."""
    assert "server" in fastmcp_command.commands
    assert "serve" in fastmcp_command.commands

    server_group = fastmcp_command.commands["server"]
    assert "dev" in server_group.commands
    assert "install" in server_group.commands
    assert "run" in server_group.commands

    dev_cmd = server_group.commands["dev"]
    option_names = {option.name for option in dev_cmd.params if hasattr(option, "name")}
    assert "reload_include_glob" in option_names
    assert "reload_exclude_glob" in option_names


def test_filter_change_set_excludes_noisy_files_by_glob() -> None:
    changes = {
        (Change.modified, "/tmp/server.log"),
        (Change.modified, "/tmp/__pycache__"),
        (Change.modified, "/tmp/__pycache__/worker.pyc"),
        (Change.modified, "/tmp/worker.py"),
    }
    filtered = _filter_change_set(changes,
                                  include_globs=(),
                                  exclude_globs=("*.log", "*__pycache__", "*__pycache__/*", "*.pyc"))
    assert filtered == {(Change.modified, "/tmp/worker.py")}


def test_iter_file_changes_applies_include_and_exclude_globs(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted_changes = iter([
        {(Change.modified, "/tmp/server.log")},
        {(Change.modified, "/tmp/workflow.py")},
    ])

    def fake_watch(*_args, **_kwargs):
        return emitted_changes

    monkeypatch.setattr("nat.plugins.fastmcp.cli.utils.watch", fake_watch)

    change_iterator = iter_file_changes(paths=[Path("/tmp")],
                                        debounce_ms=50,
                                        include_globs=("*.py", ),
                                        exclude_globs=("*.log", ))
    assert next(change_iterator) == {(Change.modified, "/tmp/workflow.py")}
    with pytest.raises(StopIteration):
        next(change_iterator)


def test_iter_file_changes_include_glob_can_match_default_excluded_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted_changes = iter([
        {(Change.modified, "/tmp/server.log")},
    ])

    def fake_watch(*_args, **_kwargs):
        return emitted_changes

    monkeypatch.setattr("nat.plugins.fastmcp.cli.utils.watch", fake_watch)

    change_iterator = iter_file_changes(paths=[Path("/tmp")], debounce_ms=50, include_globs=("*.log", ))
    assert next(change_iterator) == {(Change.modified, "/tmp/server.log")}
    with pytest.raises(StopIteration):
        next(change_iterator)


async def test_fastmcp_auth_disabled():
    config = Config(general=GeneralConfig(front_end=FastMCPFrontEndConfig()))
    worker = FastMCPFrontEndPluginWorker(config)

    mcp = await worker.create_mcp_server()

    assert mcp.auth is None


async def test_fastmcp_auth_introspection_exposes_metadata():
    server_auth = OAuth2ResourceServerConfig(
        issuer_url="http://localhost:8080/realms/master",
        introspection_endpoint="http://localhost:8080/realms/master/protocol/openid-connect/token/introspect",
        client_id="test-client",
        client_secret=SecretStr("secret"),
        scopes=["calculator_mcp_execute"],
    )
    front_end = FastMCPFrontEndConfig(server_auth=server_auth, host="0.0.0.0", port=9902)
    config = Config(general=GeneralConfig(front_end=front_end))
    worker = FastMCPFrontEndPluginWorker(config)

    mcp = await worker.create_mcp_server()

    assert isinstance(mcp.auth, RemoteAuthProvider)
    assert isinstance(mcp.auth.token_verifier, NATFastMCPTokenVerifier)

    routes = mcp.auth.get_well_known_routes(mcp_path="/mcp")
    assert any(route.path.startswith("/.well-known/oauth-protected-resource") for route in routes)


async def test_fastmcp_nat_token_verifier_adapts_active_result(monkeypatch: pytest.MonkeyPatch):
    server_auth = OAuth2ResourceServerConfig(
        issuer_url="http://localhost:8080/realms/master",
        introspection_endpoint="http://localhost:8080/realms/master/protocol/openid-connect/token/introspect",
        client_id="test-client",
        client_secret=SecretStr("secret"),
        scopes=["calculator_mcp_execute"],
    )
    verifier = NATFastMCPTokenVerifier(server_auth, base_url="http://localhost:9902")
    verify = AsyncMock(return_value=TokenValidationResult(
        client_id="client-abc",
        scopes=["calculator_mcp_execute"],
        expires_at=1234567890,
        audience=["test-client"],
        subject="user-123",
        issuer="http://localhost:8080/realms/master",
        token_type="at+jwt",
        active=True,
    ))
    monkeypatch.setattr(verifier._bearer_token_validator, "verify", verify)

    access_token = await verifier.verify_token("token-value")

    assert access_token is not None
    assert access_token.client_id == "client-abc"
    assert access_token.scopes == ["calculator_mcp_execute"]
    assert access_token.expires_at == 1234567890
    assert access_token.claims["aud"] == ["test-client"]


async def test_fastmcp_nat_token_verifier_rejects_inactive_result(monkeypatch: pytest.MonkeyPatch):
    server_auth = OAuth2ResourceServerConfig(
        issuer_url="http://localhost:8080/realms/master",
        introspection_endpoint="http://localhost:8080/realms/master/protocol/openid-connect/token/introspect",
        client_id="test-client",
        client_secret=SecretStr("secret"),
        scopes=["calculator_mcp_execute"],
    )
    verifier = NATFastMCPTokenVerifier(server_auth, base_url="http://localhost:9902")
    verify = AsyncMock(return_value=TokenValidationResult(client_id="", token_type="bearer", active=False))
    monkeypatch.setattr(verifier._bearer_token_validator, "verify", verify)

    assert await verifier.verify_token("token-value") is None


async def test_fastmcp_function_workflow_config_without_alias_uses_type():
    config = Config(general=GeneralConfig(front_end=FastMCPFrontEndConfig()))
    worker = FastMCPFrontEndPluginWorker(config)
    workflow = SimpleNamespace(functions={},
                               function_groups={},
                               config=SimpleNamespace(workflow=SimpleNamespace(type="langgraph_wrapper")))

    functions = await worker._get_all_functions(workflow)

    assert functions == {"langgraph_wrapper": workflow}


def test_fastmcp_debug_route_lists_tools():
    config = Config(general=GeneralConfig(front_end=FastMCPFrontEndConfig()))
    worker = FastMCPFrontEndPluginWorker(config)
    mcp = FastMCP("Test Server")
    functions = {
        "regular_tool": _RegularFunction(),
        "chat_tool": _ChatRequestFunction(),
        "no_schema_tool": _NoSchemaFunction(),
    }
    worker._setup_debug_endpoints(mcp, functions)

    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        resp = client.get("/debug/tools/list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == len(functions)
        tool_names = {tool["name"] for tool in data["tools"]}
        assert tool_names == set(functions.keys())


def test_fastmcp_debug_route_detail_schema():
    config = Config(general=GeneralConfig(front_end=FastMCPFrontEndConfig()))
    worker = FastMCPFrontEndPluginWorker(config)
    mcp = FastMCP("Test Server")
    functions = {
        "regular_tool": _RegularFunction(),
        "chat_tool": _ChatRequestFunction(),
        "no_schema_tool": _NoSchemaFunction(),
    }
    worker._setup_debug_endpoints(mcp, functions)

    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        resp = client.get("/debug/tools/list?name=regular_tool&detail=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "input_schema" in data["tools"][0]

        resp = client.get("/debug/tools/list?name=chat_tool&detail=true")
        assert resp.status_code == 200
        chat_schema = resp.json()["tools"][0]["input_schema"]
        assert "properties" in chat_schema
        assert "query" in chat_schema["properties"]

        resp = client.get("/debug/tools/list?name=no_schema_tool&detail=true")
        assert resp.status_code == 200
        assert "input_schema" in resp.json()["tools"][0]


def test_fastmcp_health_endpoint():
    config = Config(general=GeneralConfig(front_end=FastMCPFrontEndConfig()))
    worker = FastMCPFrontEndPluginWorker(config)
    mcp = FastMCP("Test Server")
    worker._setup_health_endpoint(mcp)

    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["server_name"] == "Test Server"
