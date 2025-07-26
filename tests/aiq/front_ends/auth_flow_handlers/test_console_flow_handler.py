# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import socket

import httpx
import pytest
from httpx import ASGITransport

from aiq.authentication.oauth2.authorization_code_flow_config import (
    OAuth2AuthorizationCodeFlowConfig,
)
from aiq.data_models.authentication import AuthFlowType
from aiq.front_ends.console.authentication_flow_handler import ConsoleAuthenticationFlowHandler
from mock_oauth2_server import MockOAuth2Server


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    """Reserve an available TCP port and immediately release it."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mock_server() -> MockOAuth2Server:
    """
    FastAPI mock‑server *without* uvicorn:
    we use its app directly via ASGITransport.
    """
    srv = MockOAuth2Server(host="testserver", port=0)  # uvicorn not started
    # we still need a client registered so the /authorize handler is happy
    srv.register_client(
        client_id="cid",
        client_secret="secret",
        redirect_base="http://127.0.0.1:9999",  # dummy; overridden per test
    )
    return srv


@pytest.fixture()
def browser_driver(monkeypatch: pytest.MonkeyPatch, mock_server: MockOAuth2Server):
    """
    Patch `webbrowser.open` so the test drives the flow over httpx.

    • First request hits the mock auth‑server (in‑process via ASGITransport).
    • That endpoint responds 302 → localhost redirect‑server spun up
      by the handler; we follow it with a *real* network call.
    """
    opened: list[str] = []

    async def _drive(url: str):
        opened.append(url)

        # 1) hit mock server (testserver)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=mock_server._app),
            base_url="http://testserver",
            follow_redirects=False,
            timeout=10,
        ) as client:
            r = await client.get(url)
            assert r.status_code == 302
            redirect_url = r.headers["location"]

        # 2) follow redirect to handler’s local FastAPI server
        async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
            await client.get(redirect_url)

    def _fake_open(url: str, *_a, **_kw):
        asyncio.get_event_loop().create_task(_drive(url))
        return True

    monkeypatch.setattr("webbrowser.open", _fake_open, raising=True)
    monkeypatch.setattr("click.echo", lambda *_: None, raising=True)
    return opened


@pytest.fixture()
def handler(mock_server: MockOAuth2Server):
    """Subclass handler so its OAuth2 client talks to our in‑process mock."""
    class _TestHandler(ConsoleAuthenticationFlowHandler):
        def __init__(self, srv):
            super().__init__()
            self._srv = srv

        # override *one* method to inject ASGITransport
        def construct_oauth_client(self, cfg):
            transport = ASGITransport(app=self._srv._app)
            from authlib.integrations.httpx_client import AsyncOAuth2Client

            client = AsyncOAuth2Client(
                client_id=cfg.client_id,
                client_secret=cfg.client_secret,
                redirect_uri=cfg.redirect_uri,
                scope=" ".join(cfg.scopes) if cfg.scopes else None,
                token_endpoint=cfg.token_url,
                base_url="http://testserver",
                transport=transport,
            )
            self._oauth_client = client
            return client

    return _TestHandler(mock_server)


# --------------------------------------------------------------------------- #
# Integration test
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_oauth2_authorization_code_flow(handler, mock_server, browser_driver):
    """Run the full flow (incl. redirect server) with minimal stubbing."""
    redirect_port = _free_port()
    # Register *again* with correct redirect URI for this test incarnation
    mock_server.register_client(
        client_id="cid",
        client_secret="secret",
        redirect_base=f"http://localhost:{redirect_port}",
    )

    cfg = OAuth2AuthorizationCodeFlowConfig(
        client_id="cid",
        client_secret="secret",
        authorization_url="http://testserver/oauth/authorize",
        token_url="http://testserver/oauth/token",
        scopes=["read"],
        use_pkce=True,
        client_url=f"http://localhost:{redirect_port}",
        run_redirect_local_server=True,
        local_redirect_server_port=redirect_port,
    )

    ctx = await handler.authenticate(cfg, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)

    # ---- Assertions ------------------------------------------------------ #
    assert browser_driver, "webbrowser.open was never invoked"
    assert ctx.headers["Authorization"].startswith("Bearer ")

    token = ctx.headers["Authorization"].split()[1]
    # token should be present in mock‑server's db
    assert token in mock_server.tokens

    # handler cleaned up its flow bookkeeping
    assert handler._active_flows == 0
    assert not handler._flows