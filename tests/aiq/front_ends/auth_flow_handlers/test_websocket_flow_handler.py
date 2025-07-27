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

import socket
from urllib.parse import parse_qs
from urllib.parse import urlparse

import httpx
import pytest
from httpx import ASGITransport
from mock_oauth2_server import MockOAuth2Server

from aiq.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from aiq.data_models.authentication import AuthFlowType
from aiq.data_models.config import AIQConfig
from aiq.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler
from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from aiq.test.functions import EchoFunctionConfig


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

    def create_oauth_client(self, cfg):
        transport = ASGITransport(app=self._oauth_server._app)
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
async def test_websocket_oauth2_flow(monkeypatch, mock_server):
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
    cfg_aiq = AIQConfig(workflow=EchoFunctionConfig())
    worker = FastApiFrontEndPluginWorker(cfg_aiq)
    # we need the add/remove‑flow callbacks but NOT the worker’s WS endpoint
    add_flow = worker._add_flow  # pylint: disable=protected-access
    remove_flow = worker._remove_flow  # pylint: disable=protected-access

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
            flow_state = worker._outstanding_flows[state]  # pylint: disable=protected-access
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
        run_local_redirect_server=False,  # no uvicorn
        client_url=f"http://localhost:{redirect_port}",
        local_redirect_server_port=redirect_port,
    )

    monkeypatch.setattr("click.echo", lambda *_: None, raising=True)  # silence CLI

    # ----------------- run the flow ------------------------------------ #
    ctx = await ws_handler.authenticate(cfg_flow, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)

    # ----------------- assertions -------------------------------------- #
    assert opened, "The authorization URL was never emitted."
    token_val = ctx.headers["Authorization"].split()[1]
    assert token_val in mock_server.tokens, "token not issued by mock server"
    # all flow‑state cleaned up
    assert worker._outstanding_flows == {}  # pylint: disable=protected-access
