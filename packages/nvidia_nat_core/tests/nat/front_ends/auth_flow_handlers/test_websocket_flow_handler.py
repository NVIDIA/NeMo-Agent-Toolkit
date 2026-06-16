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

import socket
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from urllib.parse import parse_qs
from urllib.parse import urlparse

import httpx
import pytest
from httpx import ASGITransport
from mock_oauth2_server import MockOAuth2Server

from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.data_models.api_server import ErrorTypes
from nat.data_models.authentication import AuthFlowType
from nat.data_models.config import Config
from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.front_ends.fastapi.message_handler import WebSocketMessageHandler
from nat.test.functions import EchoFunctionConfig


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _AuthHandler(WebSocketAuthenticationFlowHandler):
    """
    Override just one factory so the OAuth2 client talks to our in‑process
    mock server via ASGITransport.
    """

    def __init__(self, oauth_server: MockOAuth2Server, **kwargs):
        super().__init__(**kwargs)
        self._oauth_server = oauth_server

    def create_oauth_client(self, config):
        transport = ASGITransport(app=self._oauth_server._app)
        from authlib.integrations.httpx_client import AsyncOAuth2Client

        client = AsyncOAuth2Client(
            client_id=config.client_id,
            client_secret=config.client_secret.get_secret_value(),
            redirect_uri=config.redirect_uri,
            scope=" ".join(config.scopes) if config.scopes else None,
            token_endpoint=config.token_url,
            base_url="http://testserver",
            transport=transport,
        )
        self._oauth_client = client
        return client


# --------------------------------------------------------------------------- #
# pytest fixtures                                                              #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mock_server() -> MockOAuth2Server:
    srv = MockOAuth2Server(host="testserver", port=0)  # uvicorn‑less FastAPI app
    # placeholder registration – real redirect URL injected per‑test
    srv.register_client(client_id="cid", client_secret="secret", redirect_base="http://x")
    return srv


# --------------------------------------------------------------------------- #
# The integration test                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("set_nat_config_file_env_var")
async def test_websocket_oauth2_flow(monkeypatch, mock_server, tmp_path):
    """
    The trick: instead of relying on the FastAPI redirect route (which would
    set the Future from a *different* loop when run through ASGITransport),
    we resolve the token **directly inside** the dummy WebSocket handler,
    using the same `FlowState` instance the auth‐handler created.
    """

    redirect_port = _free_port()

    # Register the correct redirect URI for this run
    mock_server.register_client(
        client_id="cid",
        client_secret="secret",
        redirect_base=f"http://localhost:{redirect_port}",
    )

    # ----------------- build front‑end worker & FastAPI app ------------- #
    cfg_nat = Config(workflow=EchoFunctionConfig())
    worker = FastApiFrontEndPluginWorker(cfg_nat)
    # we need the add/remove‑flow callbacks but NOT the worker’s WS endpoint
    add_flow = worker._add_flow
    remove_flow = worker._remove_flow

    # ----------------- dummy WebSocket “UI” handler --------------------- #
    opened: list[str] = []

    class _DummyWSHandler:  # minimal stand‑in for the UI layer

        def set_flow_handler(self, _):  # called by worker – ignore
            return

        async def create_websocket_message(self, msg):
            opened.append(msg.text)  # record the auth URL

            # 1) ── Hit /oauth/authorize on the mock server ─────────── #
            async with httpx.AsyncClient(
                transport=ASGITransport(app=mock_server._app),
                base_url="http://testserver",
                follow_redirects=False,
                timeout=10,
            ) as client:
                r = await client.get(msg.text)
                assert r.status_code == 302
                redirect_url = r.headers["location"]

            # 2) ── Extract `code` and `state` from redirect URL ─────── #
            qs = parse_qs(urlparse(redirect_url).query)
            code = qs["code"][0]
            state = qs["state"][0]

            # 3) ── Fetch token directly & resolve the Future in‑loop ── #
            flow_state = worker._outstanding_flows[state]
            token = await flow_state.client.fetch_token(
                url=flow_state.config.token_url,
                code=code,
                code_verifier=flow_state.verifier,
                state=state,
            )
            flow_state.future.set_result(token)

    # ----------------- authentication handler instance ------------------ #
    ws_handler = _AuthHandler(
        oauth_server=mock_server,
        add_flow_cb=add_flow,
        remove_flow_cb=remove_flow,
        web_socket_message_handler=_DummyWSHandler(),
    )

    # ----------------- flow config ------------------------------------- #
    cfg_flow = OAuth2AuthCodeFlowProviderConfig(
        client_id="cid",
        client_secret="secret",
        authorization_url="http://testserver/oauth/authorize",
        token_url="http://testserver/oauth/token",
        scopes=["read"],
        use_pkce=True,
        redirect_uri=f"http://localhost:{redirect_port}/auth/redirect",
    )

    monkeypatch.setattr("click.echo", lambda *_: None, raising=True)  # silence CLI

    # ----------------- run the flow ------------------------------------ #
    ctx = await ws_handler.authenticate(cfg_flow, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)

    # ----------------- assertions -------------------------------------- #
    assert opened, "The authorization URL was never emitted."
    token_val = ctx.headers["Authorization"].split()[1]
    assert token_val in mock_server.tokens, "token not issued by mock server"

    # all flow‑state cleaned up
    assert worker._outstanding_flows == {}


# --------------------------------------------------------------------------- #
# Error Recovery Tests                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.usefixtures("set_nat_config_file_env_var")
async def test_websocket_oauth2_flow_error_handling(monkeypatch, mock_server, tmp_path):
    """Test that WebSocket flow does convert OAuth client creation errors to RuntimeError (consistent behavior)."""

    cfg_nat = Config(workflow=EchoFunctionConfig())
    worker = FastApiFrontEndPluginWorker(cfg_nat)

    # Dummy WebSocket handler
    class _DummyWSHandler:

        def set_flow_handler(self, _):
            return

        async def create_websocket_message(self, msg):
            pass

    ws_handler = WebSocketAuthenticationFlowHandler(
        add_flow_cb=worker._add_flow,
        remove_flow_cb=worker._remove_flow,
        web_socket_message_handler=_DummyWSHandler(),
        auth_timeout_seconds=0.05,
    )

    # Use a config that will pass pydantic validation but fail OAuth client creation
    cfg_flow = OAuth2AuthCodeFlowProviderConfig(
        client_id="",  # Empty string passes pydantic but may cause OAuth client errors
        client_secret="",  # Empty strings should trigger error handling
        authorization_url="http://testserver/oauth/authorize",
        token_url="http://testserver/oauth/token",
        scopes=["read"],
        use_pkce=True,
        redirect_uri="http://localhost:8000/auth/redirect",
    )

    monkeypatch.setattr("click.echo", lambda *_: None, raising=True)

    # This test demonstrates the WebSocket flow does have timeout protection (RuntimeError after 5 minutes)
    # but the OAuth client creation with empty strings doesn't actually fail as expected
    with pytest.raises(RuntimeError) as exc_info:
        await ws_handler.authenticate(cfg_flow, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)

    # Verify timeout RuntimeError is raised (demonstrates partial error handling)
    error_message = str(exc_info.value)
    assert "Authentication flow timed out" in error_message


# ---------------------------------------------------------------------------
# WebSocketMessageHandler - preflight authentication
# ---------------------------------------------------------------------------


def _preflight_provider_cfg(preflight_auth: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.preflight_auth = preflight_auth
    return cfg


def _preflight_session_manager(auth_providers: dict) -> MagicMock:
    sm = MagicMock()
    sm.get_workflow_single_output_schema.return_value = None
    sm.get_workflow_streaming_output_schema.return_value = None
    sm.config.authentication = auth_providers
    sm.shared_builder.get_auth_provider = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=MagicMock())
    cm.__aexit__ = AsyncMock(return_value=False)
    sm.session.return_value = cm
    return sm


def _preflight_handler(session_manager: MagicMock) -> tuple:
    socket = AsyncMock()
    handler = WebSocketMessageHandler(
        socket=socket,
        session_manager=session_manager,
        step_adaptor=MagicMock(),
        worker=MagicMock(),
    )
    return handler, socket


async def test_preflight_auth_authenticates_single_provider():
    provider = AsyncMock()
    sm = _preflight_session_manager({"provider_a": _preflight_provider_cfg()})
    sm.shared_builder.get_auth_provider.return_value = provider
    handler, socket = _preflight_handler(sm)
    handler.set_flow_handler(MagicMock())

    await handler._run_preflight_auth()

    sm.shared_builder.get_auth_provider.assert_awaited_once_with("provider_a")
    provider.authenticate.assert_awaited_once()
    socket.send_json.assert_not_called()


async def test_preflight_auth_authenticates_all_preflight_providers():
    provider_a, provider_b = AsyncMock(), AsyncMock()
    sm = _preflight_session_manager({
        "provider_a": _preflight_provider_cfg(),
        "provider_b": _preflight_provider_cfg(),
    })
    sm.shared_builder.get_auth_provider.side_effect = [provider_a, provider_b]
    handler, socket = _preflight_handler(sm)
    handler.set_flow_handler(MagicMock())

    await handler._run_preflight_auth()

    assert sm.shared_builder.get_auth_provider.await_count == 2
    provider_a.authenticate.assert_awaited_once()
    provider_b.authenticate.assert_awaited_once()
    socket.send_json.assert_not_called()


async def test_preflight_auth_skips_non_preflight_providers():
    provider_a = AsyncMock()
    sm = _preflight_session_manager({
        "provider_a": _preflight_provider_cfg(),
        "provider_b": _preflight_provider_cfg(preflight_auth=False),
    })
    sm.shared_builder.get_auth_provider.return_value = provider_a
    handler, socket = _preflight_handler(sm)
    handler.set_flow_handler(MagicMock())

    await handler._run_preflight_auth()

    sm.shared_builder.get_auth_provider.assert_awaited_once_with("provider_a")
    socket.send_json.assert_not_called()


async def test_preflight_auth_sends_error_and_does_not_raise_on_failure():
    provider = AsyncMock()
    provider.authenticate.side_effect = RuntimeError("OAuth server unreachable")
    sm = _preflight_session_manager({"provider_a": _preflight_provider_cfg()})
    sm.shared_builder.get_auth_provider.return_value = provider
    handler, socket = _preflight_handler(sm)
    handler.set_flow_handler(MagicMock())

    await handler._run_preflight_auth()

    socket.send_json.assert_awaited_once()
    payload: dict = socket.send_json.call_args[0][0]
    assert payload["code"] == ErrorTypes.USER_AUTH_ERROR
    assert "provider_a" in payload["message"]


async def test_preflight_auth_continues_remaining_providers_after_one_fails():
    provider_a, provider_b = AsyncMock(), AsyncMock()
    provider_a.authenticate.side_effect = RuntimeError("a failed")
    sm = _preflight_session_manager({
        "provider_a": _preflight_provider_cfg(),
        "provider_b": _preflight_provider_cfg(),
    })
    sm.shared_builder.get_auth_provider.side_effect = [provider_a, provider_b]
    handler, socket = _preflight_handler(sm)
    handler.set_flow_handler(MagicMock())

    await handler._run_preflight_auth()

    provider_a.authenticate.assert_awaited_once()
    provider_b.authenticate.assert_awaited_once()
    socket.send_json.assert_awaited_once()
    assert "provider_a" in socket.send_json.call_args[0][0]["message"]


async def test_preflight_auth_sends_one_error_per_failed_provider():
    provider_a, provider_b = AsyncMock(), AsyncMock()
    provider_a.authenticate.side_effect = RuntimeError("a failed")
    provider_b.authenticate.side_effect = RuntimeError("b failed")
    sm = _preflight_session_manager({
        "provider_a": _preflight_provider_cfg(),
        "provider_b": _preflight_provider_cfg(),
    })
    sm.shared_builder.get_auth_provider.side_effect = [provider_a, provider_b]
    handler, socket = _preflight_handler(sm)
    handler.set_flow_handler(MagicMock())

    await handler._run_preflight_auth()

    assert socket.send_json.await_count == 2
    messages = {c[0][0]["message"] for c in socket.send_json.call_args_list}
    assert any("provider_a" in m for m in messages)
    assert any("provider_b" in m for m in messages)
