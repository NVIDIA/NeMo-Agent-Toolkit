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
from mock_fastapi_worker import MockFastAPIWorker
from mock_oauth2_server import MockOAuth2Server

from aiq.authentication.oauth2.authorization_code_flow_config import OAuth2AuthorizationCodeFlowConfig
from aiq.data_models.authentication import AuthFlowType
from aiq.data_models.interactive import _HumanPromptOAuthConsent
from aiq.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _TestHandler(WebSocketAuthenticationFlowHandler):
    """
    Override *one* factory so the OAuth2 client talks to the in‑process
    FastAPI mock (no real network), everything else kept intact.
    """

    def __init__(self, oauth_server: MockOAuth2Server, **kwargs):
        super().__init__(**kwargs)
        self._oauth_server = oauth_server

    def create_oauth_client(self, cfg):
        transport = ASGITransport(app=self._oauth_server._app)
        from authlib.integrations.httpx_client import AsyncOAuth2Client

        client = AsyncOAuth2Client(
            client_id=cfg.client_id,
            client_secret=cfg.client_secret,
            redirect_uri=cfg.redirect_uri,
            scope=" ".join(cfg.scopes) if cfg.scopes else None,
            token_endpoint=cfg.token_url,
            base_url="http://testserver",  # matches host passed below
            transport=transport,
        )
        self._oauth_client = client
        return client


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mock_server() -> MockOAuth2Server:
    srv = MockOAuth2Server(host="testserver", port=0)  # no uvicorn needed
    # dummy client (redirect updated per test)
    srv.register_client(client_id="cid", client_secret="secret", redirect_base="http://x")
    return srv


# --------------------------------------------------------------------------- #
# The integration test                                                        #
# --------------------------------------------------------------------------- #
async def test_oauth2_flow_in_process(monkeypatch, mock_server):
    """
    1. Handler builds its redirect FastAPI app in‑memory (no uvicorn).
    2. webbrowser.open is patched to:
         • hit /oauth/authorize on the mock server via ASGITransport
         • follow the 302 to the handler’s *in‑process* redirect app.
    3. The whole Authorization‑Code dance finishes with a valid token.
    """
    redirect_port = _free_port()

    # Re‑register the client with the proper redirect URI for this test
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
        run_local_redirect_server=False,  # ← key line: no uvicorn
        client_url=f"http://localhost:{redirect_port}",
        local_redirect_server_port=redirect_port,  # still used in redirect_uri
    )

    # ----------------- patch browser and callbacks ---------------------------------- #
    opened: list[str] = []

    fastapi_worker = MockFastAPIWorker()
    await fastapi_worker.add_authorization_route()

    class MockWSMessageHandler:

        async def create_websocket_message(self, message: _HumanPromptOAuthConsent):
            """
            Mimics the front end which handles the consent and redirect.

            """
            # This is a mock handler, so we don't need to do anything here.
            url = message.text
            opened.append(url)  # Simulate the browser opening the URL
            async with httpx.AsyncClient(
                    transport=ASGITransport(app=mock_server._app),
                    base_url="http://testserver",
                    follow_redirects=False,
                    timeout=10,
            ) as c:
                r = await c.get(url)
                assert r.status_code == 302
                redirect_url = r.headers["location"]

            while fastapi_worker.app is None:
                await asyncio.sleep(0.01)

            async with httpx.AsyncClient(
                    transport=ASGITransport(app=fastapi_worker.app),
                    base_url="http://localhost",
                    follow_redirects=False,
                    timeout=10,
            ) as c:
                r = await c.get(redirect_url)
                assert r.status_code == 200

    handler = _TestHandler(mock_server,
                           add_flow_cb=fastapi_worker._add_flow,
                           remove_flow_cb=fastapi_worker._remove_flow,
                           web_socket_message_handler=MockWSMessageHandler())

    # ----------------- run flow ---------------------------------------- #
    ctx = await handler.authenticate(cfg, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)

    # ----------------- assertions -------------------------------------- #
    assert opened, "Browser was never opened"
    tok = ctx.headers["Authorization"].split()[1]
    assert tok in mock_server.tokens  # issued by mock server
