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

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.lowlevel.server import request_ctx
from pydantic import BaseModel
from starlette.requests import Request
from starlette.testclient import TestClient

from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.fastmcp.server.front_end_config import FastMCPFrontEndConfig
from nat.plugins.fastmcp.server.front_end_plugin import FastMCPFrontEndPlugin
from nat.plugins.fastmcp.server.front_end_plugin_worker import FastMCPFrontEndPluginWorker
from nat.runtime.session import SESSION_COOKIE_NAME
from nat.runtime.session import PerUserBuilderInfo
from nat.runtime.session import SessionManager
from nat.runtime.user_manager import UserManager


class _Input(BaseModel):
    message: str


class _Output(BaseModel):
    result: str


class PerUserFastMCPWorkflowConfig(FunctionBaseConfig, name="per_user_fastmcp_test_workflow"):
    """Per-user workflow config for FastMCP front-end tests."""

    workflow_alias: str | None = None


class SharedFastMCPWorkflowConfig(FunctionBaseConfig, name="shared_fastmcp_test_workflow"):
    """Shared workflow config for FastMCP front-end tests."""


@pytest.fixture(name="registered_workflows", scope="module")
def fixture_registered_workflows():
    """Register test workflows in a pushed registry so they do not leak."""

    @register_per_user_function(config_type=PerUserFastMCPWorkflowConfig, input_type=_Input, single_output_type=_Output)
    async def _build_per_user(_config: PerUserFastMCPWorkflowConfig, _builder: Builder):

        async def _impl(inp: _Input) -> _Output:
            return _Output(result=f"per-user: {inp.message}")

        yield FunctionInfo.from_fn(_impl)

    @register_function(config_type=SharedFastMCPWorkflowConfig)
    async def _build_shared(_config: SharedFastMCPWorkflowConfig, _builder: Builder):

        async def _impl(inp: _Input) -> _Output:
            return _Output(result=f"shared: {inp.message}")

        yield FunctionInfo.from_fn(_impl)


def _config(workflow) -> Config:
    return Config(
        general=GeneralConfig(front_end=FastMCPFrontEndConfig(
            name="Test FastMCP Server",
            host="localhost",
            port=9903,
            debug=False,
            log_level="INFO",
        )),
        workflow=workflow,
    )


@pytest.fixture(name="per_user_config")
def fixture_per_user_config(registered_workflows) -> Config:
    config = _config(PerUserFastMCPWorkflowConfig())
    config.general.front_end.server_auth = OAuth2ResourceServerConfig(issuer_url="https://issuer.example",
                                                                      audience="test-api")
    return config


@pytest.fixture(name="shared_config")
def fixture_shared_config(registered_workflows) -> Config:
    return _config(SharedFastMCPWorkflowConfig())


class TestPerUserWorkflowStartup:
    """The FastMCP front end must serve per-user workflows, not die building them."""

    async def test_per_user_workflow_server_starts(self, per_user_config, monkeypatch):
        """Startup used to raise "Must set a workflow before building"."""

        async def _no_serve(*_args, **_kwargs):
            return None

        monkeypatch.setattr("fastmcp.server.server.FastMCP.run_async", _no_serve)

        await FastMCPFrontEndPlugin(full_config=per_user_config).run()

    async def test_shared_workflow_still_builds(self, shared_config, monkeypatch):
        captured = {}

        original_create = SessionManager.create

        async def _capture_create(*args, **kwargs):
            session_manager = await original_create(*args, **kwargs)
            if not session_manager.is_workflow_per_user:
                captured["workflow"] = session_manager.workflow
            return session_manager

        async def _no_serve(*_args, **_kwargs):
            return None

        monkeypatch.setattr(SessionManager, "create", _capture_create)
        monkeypatch.setattr("fastmcp.server.server.FastMCP.run_async", _no_serve)

        await FastMCPFrontEndPlugin(full_config=shared_config).run()

        assert captured["workflow"] is not None

    async def test_per_user_session_manager_reaps_and_shuts_down(self, per_user_config):
        worker = FastMCPFrontEndPluginWorker(per_user_config)

        async with WorkflowBuilder.from_config(config=per_user_config) as builder:
            mcp = await worker.create_mcp_server()
            await worker._default_add_routes(mcp, builder)

            assert len(worker._session_managers) == 1
            session_manager = worker._session_managers[0]
            assert session_manager.is_workflow_per_user
            assert session_manager._per_user_builders_cleanup_task is not None

            cleanup_task = session_manager._per_user_builders_cleanup_task
            await worker.cleanup()
            assert cleanup_task.done()

    async def test_register_function_skips_shared_workflow_lookup(self, per_user_config, monkeypatch):
        from nat.plugins.fastmcp.server import tool_converter

        worker = FastMCPFrontEndPluginWorker(per_user_config)

        async with WorkflowBuilder.from_config(config=per_user_config) as builder:
            session_manager = await SessionManager.create(config=per_user_config, shared_builder=builder)
            mcp = await worker.create_mcp_server()

            get_schema = MagicMock(return_value=_Input)
            monkeypatch.setattr(session_manager, "get_workflow_input_schema", get_schema)

            tool_converter.register_function_with_mcp(mcp, "per_user_fastmcp_test_workflow", session_manager)

            get_schema.assert_called_once()


class TestWorkerCleanup:
    """Worker cleanup must shut down every manager and clear tracking."""

    async def test_cleanup_shuts_down_all_managers_after_shutdown_failure(self, per_user_config):
        worker = FastMCPFrontEndPluginWorker(per_user_config)
        failing = MagicMock()
        failing.shutdown = AsyncMock(side_effect=RuntimeError("boom"))
        succeeding = MagicMock()
        succeeding.shutdown = AsyncMock()
        worker._session_managers = [failing, succeeding]

        with pytest.raises(RuntimeError, match="boom"):
            await worker.cleanup()

        failing.shutdown.assert_awaited_once()
        succeeding.shutdown.assert_awaited_once()
        assert worker._session_managers == []


class TestPerUserRequestIdentity:
    """Per-user tool calls must reach session() with a resolved user id."""

    async def test_run_through_session_manager_uses_context_user_id(self, monkeypatch):
        from nat.plugins.fastmcp.server.tool_converter import _run_through_session_manager

        session_manager = MagicMock()
        session_manager.is_workflow_per_user = True
        session_manager.session = MagicMock()
        session = MagicMock()
        runner = MagicMock()
        runner.result = AsyncMock(return_value="ok")
        session.run.return_value.__aenter__ = AsyncMock(return_value=runner)
        session.run.return_value.__aexit__ = AsyncMock(return_value=False)
        session_manager.session.return_value.__aenter__ = AsyncMock(return_value=session)
        session_manager.session.return_value.__aexit__ = AsyncMock(return_value=False)

        context = SimpleNamespace(user_id="alice")
        monkeypatch.setattr("nat.builder.context.Context.get", lambda: context)

        payload = _Input(message="hello")
        result = await _run_through_session_manager(session_manager, payload)

        assert result == "ok"
        session_manager.session.assert_called_once_with(user_id="alice", http_connection=None)

    async def test_run_through_session_manager_resolves_user_from_mcp_request(self, monkeypatch):
        from nat.plugins.fastmcp.server.tool_converter import _run_through_session_manager

        session_manager = MagicMock()
        session_manager.is_workflow_per_user = True
        session_manager.session = MagicMock()
        session = MagicMock()
        runner = MagicMock()
        runner.result = AsyncMock(return_value="ok")
        session.run.return_value.__aenter__ = AsyncMock(return_value=runner)
        session.run.return_value.__aexit__ = AsyncMock(return_value=False)
        session_manager.session.return_value.__aenter__ = AsyncMock(return_value=session)
        session_manager.session.return_value.__aexit__ = AsyncMock(return_value=False)

        context = SimpleNamespace(user_id=None)
        monkeypatch.setattr("nat.builder.context.Context.get", lambda: context)

        request = _authenticated_request(_unsigned_jwt({"sub": "bob"}), cookie=None)
        ctx = SimpleNamespace(request_context=SimpleNamespace(request=request))

        payload = _Input(message="hello")
        result = await _run_through_session_manager(session_manager, payload, ctx=ctx)

        assert result == "ok"
        session_manager.session.assert_called_once_with(user_id=UserManager._user_info_from_jwt({
            "sub": "bob"
        }).get_user_id(),
                                                        http_connection=request)


class TestPerUserToolStreamableHttp:
    """Per-user tool calls over streamable-http must receive injected request context."""

    async def test_call_tool_resolves_user_from_bearer_token(self, per_user_config, monkeypatch):
        from nat.plugins.fastmcp.server import tool_converter

        worker = FastMCPFrontEndPluginWorker(per_user_config)

        async with WorkflowBuilder.from_config(config=per_user_config) as builder:
            session_manager = await SessionManager.create(config=per_user_config, shared_builder=builder)
            mcp = await worker.create_mcp_server()
            tool_converter.register_function_with_mcp(mcp, "per_user_fastmcp_test_workflow", session_manager)

            session = MagicMock()
            runner = MagicMock()
            runner.result = AsyncMock(return_value="ok")
            session.run.return_value.__aenter__ = AsyncMock(return_value=runner)
            session.run.return_value.__aexit__ = AsyncMock(return_value=False)

            captured: dict[str, object] = {}
            session_cm = MagicMock()
            session_cm.__aenter__ = AsyncMock(return_value=session)
            session_cm.__aexit__ = AsyncMock(return_value=False)

            def _capture_session(user_id=None, http_connection=None):
                captured["user_id"] = user_id
                captured["http_connection"] = http_connection
                return session_cm

            monkeypatch.setattr(session_manager, "session", _capture_session)

            monkeypatch.setattr("nat.builder.context.Context.get", lambda: SimpleNamespace(user_id=None))
            request = _authenticated_request(_unsigned_jwt({"sub": "bob"}), cookie=None)

            with TestClient(mcp.http_app(transport="streamable-http")):
                token = request_ctx.set(SimpleNamespace(request=request, session=MagicMock()))
                try:
                    await mcp.call_tool("per_user_fastmcp_test_workflow", {"message": "hello"})
                finally:
                    request_ctx.reset(token)

            assert captured["user_id"] == UserManager._user_info_from_jwt({"sub": "bob"}).get_user_id()
            assert captured["http_connection"] is request


def _unsigned_jwt(claims: dict) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}."


def _authenticated_request(token: str, cookie: str | None) -> Request:
    headers = [(b"authorization", f"Bearer {token}".encode())]
    if cookie is not None:
        headers.append((b"cookie", f"{SESSION_COOKIE_NAME}={cookie}".encode()))
    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "root_path": "",
        "user": AuthenticatedUser(AccessToken(token=token, client_id="alice-client", scopes=["execute"])),
    })


class TestVerifiedPrincipalIsolation:
    """A valid bearer token must not select another user's builder via nat-session."""

    async def test_alice_token_does_not_build_bob_workflow(self, per_user_config, monkeypatch):
        from nat.plugins.fastmcp.server.tool_converter import _run_through_session_manager

        alice_token = _unsigned_jwt({"sub": "alice"})
        alice_id = UserManager._user_info_from_jwt({"sub": "alice"}).get_user_id()
        built: list[str] = []

        async def _install_builder(user_id: str):
            built.append(user_id)
            workflow = MagicMock()
            runner = MagicMock()
            runner.result = AsyncMock(return_value="ok")
            workflow.run.return_value.__aenter__ = AsyncMock(return_value=runner)
            workflow.run.return_value.__aexit__ = AsyncMock(return_value=False)
            builder = MagicMock()
            builder.__aexit__ = AsyncMock()
            session_manager._per_user_builders[user_id] = PerUserBuilderInfo(
                builder=builder,
                workflow=workflow,
                semaphore=asyncio.Semaphore(1),
            )
            return builder, workflow

        monkeypatch.setattr("nat.builder.context.Context.get", lambda: SimpleNamespace(user_id=None))

        async with WorkflowBuilder.from_config(config=per_user_config) as builder:
            session_manager = await SessionManager.create(config=per_user_config, shared_builder=builder)
            monkeypatch.setattr(session_manager, "_get_or_create_per_user_builder", _install_builder)
            try:
                assert session_manager._identity_header is None

                conflict = SimpleNamespace(request_context=SimpleNamespace(
                    request=_authenticated_request(alice_token, cookie="bob")))
                with pytest.raises(ValueError, match="nat-session cookie does not match"):
                    await _run_through_session_manager(session_manager, _Input(message="hello"), ctx=conflict)
                assert built == []

                allowed = SimpleNamespace(request_context=SimpleNamespace(
                    request=_authenticated_request(alice_token, cookie=None)))
                result = await _run_through_session_manager(session_manager, _Input(message="hello"), ctx=allowed)
            finally:
                await session_manager.shutdown()

        assert result == "ok"
        assert built == [alice_id]
        assert alice_id != UserManager._user_info_from_jwt({"sub": "bob"}).get_user_id()


class TestPerUserSecurityPolicy:
    """Reject unsupported exposure and unverified request identities before building a user workflow."""

    @pytest.mark.parametrize("missing_auth,transport", [(True, "streamable-http"), (False, "sse")])
    async def test_rejects_unauthenticated_transport(self, per_user_config, missing_auth, transport):
        per_user_config.general.front_end.transport = transport
        if missing_auth:
            per_user_config.general.front_end.server_auth = None
        worker = FastMCPFrontEndPluginWorker(per_user_config)
        async with WorkflowBuilder.from_config(config=per_user_config) as builder:
            mcp = await worker.create_mcp_server()
            with pytest.raises(ValueError, match="require streamable-http transport with server_auth"):
                await worker._default_add_routes(mcp, builder)
            assert await mcp.list_tools() == []
            assert worker._session_managers == []

    @pytest.mark.parametrize("tool_names", [["unrelated"], ["per_user_fastmcp_test_workflow", "unrelated"]])
    async def test_rejects_other_tool_selections(self, per_user_config, tool_names):
        per_user_config.general.front_end.tool_names = tool_names
        worker = FastMCPFrontEndPluginWorker(per_user_config)
        async with WorkflowBuilder.from_config(config=per_user_config) as builder:
            mcp = await worker.create_mcp_server()
            with pytest.raises(ValueError, match="can only expose the workflow tool"):
                await worker._default_add_routes(mcp, builder)
            assert await mcp.list_tools() == []
            assert worker._session_managers == []

    @pytest.mark.parametrize("alias", [None, "custom_workflow"])
    async def test_accepts_workflow_tool_selection(self, per_user_config, alias):
        per_user_config.workflow.workflow_alias = alias
        tool_name = alias or per_user_config.workflow.type
        per_user_config.general.front_end.tool_names = [tool_name]
        worker = FastMCPFrontEndPluginWorker(per_user_config)
        async with WorkflowBuilder.from_config(config=per_user_config) as builder:
            try:
                mcp = await worker.create_mcp_server()
                await worker._default_add_routes(mcp, builder)
                assert [tool.name for tool in await mcp.list_tools()] == [tool_name]
            finally:
                await worker.cleanup()

    @pytest.mark.parametrize("credential", ["cookie", "unsigned_jwt", "opaque_token", "missing"])
    async def test_unverified_identity_never_creates_a_session(self, per_user_config, monkeypatch, credential):
        from nat.plugins.fastmcp.server.tool_converter import _run_through_session_manager

        request = _authenticated_request(_unsigned_jwt({"sub": "bob"}), cookie=None)
        request.scope.pop("user")
        headers = []
        if credential == "cookie":
            headers = [(b"cookie", b"nat-session=bob")]
        elif credential == "unsigned_jwt":
            headers = request.scope["headers"]
        elif credential == "opaque_token":
            headers = [(b"authorization", b"Bearer arbitrary-token")]
        request.scope["headers"] = headers
        ctx = SimpleNamespace(request_context=SimpleNamespace(request=request))
        monkeypatch.setattr("nat.builder.context.Context.get", lambda: SimpleNamespace(user_id=None))
        async with WorkflowBuilder.from_config(config=per_user_config) as builder:
            manager = await SessionManager.create(config=per_user_config, shared_builder=builder)
            try:
                with pytest.raises(ValueError, match="Authenticated user identity is required"):
                    await _run_through_session_manager(manager, _Input(message="hello"), ctx=ctx)
                assert manager._per_user_builders == {}
            finally:
                await manager.shutdown()


async def test_authenticated_http_tool_call_and_identity_isolation(per_user_config, monkeypatch):
    """Use actual HTTP auth middleware and tool dispatch, including an introspected dotted token."""
    from nat.authentication.credential_validator.bearer_token_validator import BearerTokenValidator
    from nat.data_models.authentication import TokenValidationResult

    async def verify(_self, token):
        return TokenValidationResult(active=token == "opaque.with.dots",
                                     client_id="test-client",
                                     subject="alice",
                                     issuer="https://issuer.example",
                                     audience=["test-api"],
                                     token_type="opaque",
                                     scopes=[])

    monkeypatch.setattr(BearerTokenValidator, "verify", verify)
    worker = FastMCPFrontEndPluginWorker(per_user_config)
    async with WorkflowBuilder.from_config(config=per_user_config) as builder:
        try:
            mcp = await worker.create_mcp_server()
            await worker._default_add_routes(mcp, builder)
            app = mcp.http_app(transport="streamable-http", path="/mcp", json_response=True)
            with TestClient(app,
                            base_url=f"http://{worker.front_end_config.host}:{worker.front_end_config.port}") as client:
                client.headers.update({"accept": "application/json, text/event-stream"})
                initialize = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "regression-test", "version": "1"
                        }
                    }
                }
                # A cookie or unsigned JWT cannot replace authentication at the HTTP boundary.
                assert client.post("/mcp", json=initialize, headers={"cookie": "nat-session=bob"}).status_code == 401
                assert client.post("/mcp",
                                   json=initialize,
                                   headers={
                                       "authorization": f"Bearer {_unsigned_jwt({'sub': 'bob'})}"
                                   }).status_code == 401
                manager = worker._session_managers[0]
                assert manager._per_user_builders == {}

                client.headers.update({"authorization": "Bearer opaque.with.dots"})
                response = client.post("/mcp", json=initialize)
                assert response.status_code == 200
                client.headers.update({
                    "mcp-session-id": response.headers["mcp-session-id"], "mcp-protocol-version": "2025-03-26"
                })
                assert client.post("/mcp", json={
                    "jsonrpc": "2.0", "method": "notifications/initialized"
                }).status_code == 202
                call = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "per_user_fastmcp_test_workflow", "arguments": {
                            "message": "hello"
                        }
                    }
                }
                result = client.post("/mcp", json=call).json()["result"]
                assert not result.get("isError", False)
                assert "per-user: hello" in result["content"][0]["text"]
                alice_id = UserManager._user_info_from_jwt({"sub": "alice"}).get_user_id()
                assert set(manager._per_user_builders) == {alice_id}

                call["id"] = 3
                result = client.post("/mcp", json=call, headers={"cookie": "nat-session=bob"}).json()["result"]
                assert result["isError"]
                assert "nat-session cookie does not match" in result["content"][0]["text"]
                assert set(manager._per_user_builders) == {alice_id}
        finally:
            await worker.cleanup()
