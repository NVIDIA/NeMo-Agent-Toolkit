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
import time
from urllib.parse import parse_qs
from urllib.parse import urlparse

import httpx
import pytest
from httpx import ASGITransport
from mock_oauth2_server import MockOAuth2Server

from nat.authentication.api_key.api_key_auth_provider_config import APIKeyAuthProviderConfig
from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.config import Config
from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.test.functions import EchoFunctionConfig


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _complete_oauth_redirect(auth_url: str, mock_server: MockOAuth2Server, outstanding_flows: dict):
    """Hit the mock OAuth server, parse the redirect, fetch a token, and resolve the flow future."""
    async with httpx.AsyncClient(
            transport=ASGITransport(app=mock_server._app),
            base_url="http://testserver",
            follow_redirects=False,
            timeout=10,
    ) as client:
        r = await client.get(auth_url)
        assert r.status_code == 302
        redirect_url = r.headers["location"]

    qs = parse_qs(urlparse(redirect_url).query)
    code = qs["code"][0]
    state = qs["state"][0]

    flow_state = outstanding_flows[state]
    token = await flow_state.client.fetch_token(
        url=flow_state.config.token_url,
        code=code,
        code_verifier=flow_state.verifier,
        state=state,
    )
    flow_state.future.set_result(token)
    return flow_state


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
    received_messages: list = []

    class _DummyWSHandler:  # minimal stand‑in for the UI layer

        def set_flow_handler(self, _):  # called by worker – ignore
            return

        async def create_websocket_message(self, msg):
            opened.append(msg.text)  # record the auth URL
            received_messages.append(msg)
            await _complete_oauth_redirect(msg.text, mock_server, worker._outstanding_flows)

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
    assert received_messages[0].use_redirect is False, "Default use_redirect_auth should emit use_redirect=False"
    token_val = ctx.headers["Authorization"].split()[1]
    assert token_val in mock_server.tokens, "token not issued by mock server"

    # all flow‑state cleaned up
    assert worker._outstanding_flows == {}


# --------------------------------------------------------------------------- #
# use_redirect_auth=True test                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("set_nat_config_file_env_var")
async def test_websocket_oauth2_flow_no_popup(monkeypatch, mock_server, tmp_path):
    """Verify that use_redirect_auth=True sends use_redirect=True in the consent prompt and
    propagates return_url into FlowState."""

    redirect_port = _free_port()

    mock_server.register_client(
        client_id="cid",
        client_secret="secret",
        redirect_base=f"http://localhost:{redirect_port}",
    )

    cfg_nat = Config(workflow=EchoFunctionConfig())
    worker = FastApiFrontEndPluginWorker(cfg_nat)
    add_flow = worker._add_flow
    remove_flow = worker._remove_flow

    received_messages: list = []
    captured_flow_states: list = []

    class _DummyWSHandler:

        def set_flow_handler(self, _):
            return

        async def create_websocket_message(self, msg):
            received_messages.append(msg)
            flow_state = await _complete_oauth_redirect(msg.text, mock_server, worker._outstanding_flows)
            captured_flow_states.append(flow_state)

    ws_handler = _AuthHandler(
        oauth_server=mock_server,
        add_flow_cb=add_flow,
        remove_flow_cb=remove_flow,
        web_socket_message_handler=_DummyWSHandler(),
        return_url="http://localhost:3000",
    )

    cfg_flow = OAuth2AuthCodeFlowProviderConfig(
        client_id="cid",
        client_secret="secret",
        authorization_url="http://testserver/oauth/authorize",
        token_url="http://testserver/oauth/token",
        scopes=["read"],
        use_pkce=True,
        redirect_uri=f"http://localhost:{redirect_port}/auth/redirect",
        use_redirect_auth=True,
    )

    monkeypatch.setattr("click.echo", lambda *_: None, raising=True)

    ctx = await ws_handler.authenticate(cfg_flow, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)

    assert received_messages, "The authorization URL was never emitted."
    assert received_messages[0].use_redirect is True, "use_redirect_auth=True should emit use_redirect=True"
    assert captured_flow_states[0].return_url == "http://localhost:3000"
    token_val = ctx.headers["Authorization"].split()[1]
    assert token_val in mock_server.tokens, "token not issued by mock server"
    assert worker._outstanding_flows == {}


# --------------------------------------------------------------------------- #
# use_redirect_auth=True without return_url guard                             #
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("set_nat_config_file_env_var")
async def test_websocket_oauth2_flow_redirect_without_return_url(monkeypatch):
    """Verify that use_redirect_auth=True with no return_url raises ValueError immediately."""

    cfg_nat = Config(workflow=EchoFunctionConfig())
    worker = FastApiFrontEndPluginWorker(cfg_nat)

    class _DummyWSHandler:

        def set_flow_handler(self, _):
            return

        async def create_websocket_message(self, msg):
            pass

    ws_handler = WebSocketAuthenticationFlowHandler(
        add_flow_cb=worker._add_flow,
        remove_flow_cb=worker._remove_flow,
        web_socket_message_handler=_DummyWSHandler(),
        return_url=None,
    )

    cfg_flow = OAuth2AuthCodeFlowProviderConfig(
        client_id="cid",
        client_secret="secret",
        authorization_url="http://testserver/oauth/authorize",
        token_url="http://testserver/oauth/token",
        scopes=["read"],
        redirect_uri="http://localhost:8000/auth/redirect",
        use_redirect_auth=True,
    )

    monkeypatch.setattr("click.echo", lambda *_: None, raising=True)

    with pytest.raises(ValueError, match="return URL"):
        await ws_handler.authenticate(cfg_flow, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)


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


# --------------------------------------------------------------------------- #
# Token-cache unit tests                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture(name="noop_handler")
def noop_handler_fixture():
    """Minimal handler with no-op callbacks for cache unit tests."""

    async def _noop_add(state, flow_state):
        pass

    async def _noop_remove(state):
        pass

    class _NullWSHandler:

        async def create_websocket_message(self, msg):
            pass

    return WebSocketAuthenticationFlowHandler(
        add_flow_cb=_noop_add,
        remove_flow_cb=_noop_remove,
        web_socket_message_handler=_NullWSHandler(),
    )


@pytest.fixture(name="minimal_oauth_config")
def minimal_oauth_config_fixture():
    return OAuth2AuthCodeFlowProviderConfig(
        client_id="test-client",
        client_secret="test-secret",
        authorization_url="http://auth.example.com/authorize",
        token_url="http://auth.example.com/token",
        redirect_uri="http://localhost:8000/callback",
    )


def test_token_cache_key_returns_none_without_required_attrs(noop_handler, minimal_oauth_config):
    """_token_cache_key returns None when session_id or token_store is absent."""
    noop_handler._token_store = {}
    noop_handler._session_id = None
    assert noop_handler._token_cache_key(minimal_oauth_config) is None

    noop_handler._token_store = None
    noop_handler._session_id = "sess-1"
    assert noop_handler._token_cache_key(minimal_oauth_config) is None


def test_token_cache_key_format(noop_handler, minimal_oauth_config):
    """_token_cache_key returns '{session_id}:{client_id}:{token_url}' when both are present."""
    noop_handler._token_store = {}
    noop_handler._session_id = "sess-1"
    key = noop_handler._token_cache_key(minimal_oauth_config)
    assert key == f"sess-1:{minimal_oauth_config.client_id}:{minimal_oauth_config.token_url}"


def test_get_cached_token_miss(noop_handler, minimal_oauth_config):
    """_get_cached_token returns None when the cache has no entry for the config."""
    noop_handler._token_store = {}
    noop_handler._session_id = "sess-1"
    assert noop_handler._get_cached_token(minimal_oauth_config) is None


@pytest.mark.parametrize("expires_at,expect_hit",
                         [
                             pytest.param(None, True, id="no_expiry"),
                             pytest.param(time.time() + 3600, True, id="future"),
                             pytest.param(time.time() - 1, False, id="past"),
                             pytest.param(time.time() + 30, False, id="within_buffer"),
                         ])
def test_get_cached_token_expiry(noop_handler, minimal_oauth_config, expires_at, expect_hit):
    """_get_cached_token returns the context when valid and evicts it when expired or within the 60s buffer."""
    ctx = AuthenticatedContext(headers={"Authorization": "Bearer tok"}, metadata={})
    store: dict = {}
    noop_handler._token_store = store
    noop_handler._session_id = "sess-1"
    key = noop_handler._token_cache_key(minimal_oauth_config)
    store[key] = (ctx, expires_at)
    result = noop_handler._get_cached_token(minimal_oauth_config)
    if expect_hit:
        assert result is ctx
    else:
        assert result is None
        assert key not in store


def test_store_token_writes_correctly(noop_handler, minimal_oauth_config):
    """_store_token writes (ctx, expires_at) to the store under the expected key."""
    store: dict = {}
    noop_handler._token_store = store
    noop_handler._session_id = "sess-1"
    expires = 9999999999.0
    ctx = AuthenticatedContext(headers={"Authorization": "Bearer tok"}, metadata={"expires_at": expires})
    noop_handler._store_token(minimal_oauth_config, ctx)
    key = noop_handler._token_cache_key(minimal_oauth_config)
    assert key in store
    stored_ctx, stored_expires = store[key]
    assert stored_ctx is ctx
    assert stored_expires == expires


# --------------------------------------------------------------------------- #
# Token-cache integration: second authenticate() returns from cache          #
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("set_nat_config_file_env_var")
async def test_authenticate_second_call_uses_cache(monkeypatch, mock_server, tmp_path):
    """After a successful flow the token is cached; a second call must not trigger OAuth again."""

    redirect_port = _free_port()
    mock_server.register_client(
        client_id="cid",
        client_secret="secret",
        redirect_base=f"http://localhost:{redirect_port}",
    )

    cfg_nat = Config(workflow=EchoFunctionConfig())
    worker = FastApiFrontEndPluginWorker(cfg_nat)
    message_count = [0]

    class _DummyWSHandler:

        def set_flow_handler(self, _):
            return

        async def create_websocket_message(self, msg):
            message_count[0] += 1
            await _complete_oauth_redirect(msg.text, mock_server, worker._outstanding_flows)

    token_store: dict = {}
    ws_handler = _AuthHandler(
        oauth_server=mock_server,
        add_flow_cb=worker._add_flow,
        remove_flow_cb=worker._remove_flow,
        web_socket_message_handler=_DummyWSHandler(),
        token_store=token_store,
        session_id="test-session",
    )

    cfg_flow = OAuth2AuthCodeFlowProviderConfig(
        client_id="cid",
        client_secret="secret",
        authorization_url="http://testserver/oauth/authorize",
        token_url="http://testserver/oauth/token",
        scopes=["read"],
        use_pkce=True,
        redirect_uri=f"http://localhost:{redirect_port}/auth/redirect",
    )

    monkeypatch.setattr("click.echo", lambda *_: None, raising=True)

    ctx1 = await ws_handler.authenticate(cfg_flow, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)
    assert message_count[0] == 1, "OAuth flow should have run exactly once"
    assert token_store, "Token must be stored after first auth"

    ctx2 = await ws_handler.authenticate(cfg_flow, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)
    assert message_count[0] == 1, "Second authenticate() must return from cache without triggering OAuth"
    assert ctx2.headers["Authorization"] == ctx1.headers["Authorization"]


# --------------------------------------------------------------------------- #
# run_eager_auth tests                                                        #
# --------------------------------------------------------------------------- #
async def test_run_eager_auth_skips_non_oauth2_providers(noop_handler):
    """run_eager_auth is a no-op for non-OAuth2 providers such as APIKeyAuthProviderConfig."""
    api_key_config = APIKeyAuthProviderConfig(raw_key="my-api-key-value")
    await noop_handler.run_eager_auth({"my_api_key": api_key_config})


async def test_run_eager_auth_skips_oauth2_provider_flag_false(noop_handler, minimal_oauth_config):
    """run_eager_auth does not trigger auth for OAuth2 providers with use_eager_auth=False (the default)."""
    # minimal_oauth_config has use_eager_auth=False (the default); if the guard were absent this would hang
    await noop_handler.run_eager_auth({"my_provider": minimal_oauth_config})


async def test_run_eager_auth_uses_cached_token(minimal_oauth_config):
    """run_eager_auth returns immediately without calling create_websocket_message on a cache hit."""

    async def _noop_add(state, flow_state):
        pass

    async def _noop_remove(state):
        pass

    message_count = [0]

    class _CountingWSHandler:

        async def create_websocket_message(self, msg):
            message_count[0] += 1

    ctx = AuthenticatedContext(headers={"Authorization": "Bearer cached-tok"}, metadata={"expires_at": None})
    store: dict = {}
    # Enable use_eager_auth so the cache lookup is actually reached
    active_config = minimal_oauth_config.model_copy(update={"use_eager_auth": True})
    handler = WebSocketAuthenticationFlowHandler(
        add_flow_cb=_noop_add,
        remove_flow_cb=_noop_remove,
        web_socket_message_handler=_CountingWSHandler(),
        token_store=store,
        session_id="sess-1",
    )
    key = handler._token_cache_key(active_config)
    store[key] = (ctx, time.time() + 3600)

    await handler.run_eager_auth({"my_provider": active_config})
    assert message_count[0] == 0, "run_eager_auth must not trigger OAuth when token is cached"
