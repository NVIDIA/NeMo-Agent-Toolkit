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

import asyncio
import socket
from urllib.parse import parse_qs
from urllib.parse import urlparse

import httpx
import pytest
from httpx import ASGITransport
from mock_oauth2_server import MockOAuth2Server

from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.data_models.authentication import AuthFlowType
from nat.data_models.config import Config
from nat.data_models.interactive_http import ExecutionStatus
from nat.front_ends.fastapi.auth_flow_handlers.http_flow_handler import HTTPAuthenticationFlowHandler
from nat.front_ends.fastapi.execution_store import ExecutionStore
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.test.functions import EchoFunctionConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- #
# Tests: no execution context
# --------------------------------------------------------------------------- #
async def test_authenticate_raises_without_execution_context():
    """Without execution context, any auth method raises NotImplementedError."""
    handler = HTTPAuthenticationFlowHandler()
    cfg = OAuth2AuthCodeFlowProviderConfig(
        client_id="cid",
        client_secret="secret",
        authorization_url="http://example.com/auth",
        token_url="http://example.com/token",
        scopes=["read"],
        redirect_uri="http://localhost:8000/auth/redirect",
    )
    with pytest.raises(NotImplementedError, match="not supported"):
        await handler.authenticate(cfg, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)


async def test_authenticate_raises_for_unsupported_method():
    """Even with execution context, unsupported auth methods raise NotImplementedError."""
    handler = HTTPAuthenticationFlowHandler(
        add_flow_cb=None,
        remove_flow_cb=None,
    )
    store = ExecutionStore()
    record = await store.create_execution()
    handler.set_execution_context(
        execution_id=record.execution_id,
        store=store,
    )
    cfg = OAuth2AuthCodeFlowProviderConfig(
        client_id="cid",
        client_secret="secret",
        authorization_url="http://example.com/auth",
        token_url="http://example.com/token",
        scopes=["read"],
        redirect_uri="http://localhost:8000/auth/redirect",
    )
    # Use a method that we don't handle
    with pytest.raises(NotImplementedError, match="not supported"):
        await handler.authenticate(cfg, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)


# --------------------------------------------------------------------------- #
# Tests: with execution context (OAuth2 flow)
# --------------------------------------------------------------------------- #
class _HTTPFlowHandler(HTTPAuthenticationFlowHandler):
    """Override OAuth client creation to use mock transport."""

    def __init__(self, oauth_server: MockOAuth2Server, **kwargs):
        super().__init__(**kwargs)
        self._oauth_server = oauth_server

    def _create_oauth_client(self, config):
        from authlib.integrations.httpx_client import AsyncOAuth2Client

        transport = ASGITransport(app=self._oauth_server._app)
        return AsyncOAuth2Client(
            client_id=config.client_id,
            client_secret=config.client_secret.get_secret_value(),
            redirect_uri=config.redirect_uri,
            scope=" ".join(config.scopes) if config.scopes else None,
            token_endpoint=config.token_url,
            base_url="http://testserver",
            transport=transport,
        )


@pytest.fixture(scope="module")
def mock_server() -> MockOAuth2Server:
    srv = MockOAuth2Server(host="testserver", port=0)
    srv.register_client(client_id="cid", client_secret="secret", redirect_base="http://x")
    return srv


@pytest.mark.usefixtures("set_nat_config_file_env_var")
async def test_http_oauth2_flow_with_execution_context(monkeypatch, mock_server):
    """
    Full OAuth2 flow through HTTPAuthenticationFlowHandler:
    - sets execution store to oauth_required
    - signals first_outcome
    - token returned after flow_state.future is resolved
    """

    redirect_port = _free_port()
    mock_server.register_client(
        client_id="cid",
        client_secret="secret",
        redirect_base=f"http://localhost:{redirect_port}",
    )

    cfg_nat = Config(workflow=EchoFunctionConfig())
    worker = FastApiFrontEndPluginWorker(cfg_nat)

    store = ExecutionStore()
    record = await store.create_execution()

    handler = _HTTPFlowHandler(
        oauth_server=mock_server,
        add_flow_cb=worker._add_flow,
        remove_flow_cb=worker._remove_flow,
    )
    handler.set_execution_context(
        execution_id=record.execution_id,
        store=store,
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

    async def _simulate_redirect():
        """Wait for oauth_required, then simulate the redirect callback."""
        await record.first_outcome.wait()

        assert record.status == ExecutionStatus.OAUTH_REQUIRED
        assert record.pending_oauth is not None

        # Find the flow state in the worker
        state = record.pending_oauth.oauth_state
        flow_state = worker._outstanding_flows[state]

        # Simulate hitting the authorization URL and getting a code
        async with httpx.AsyncClient(
            transport=ASGITransport(app=mock_server._app),
            base_url="http://testserver",
            follow_redirects=False,
            timeout=10,
        ) as client:
            auth_url = record.pending_oauth.auth_url
            r = await client.get(auth_url)
            assert r.status_code == 302
            redirect_url = r.headers["location"]

        qs = parse_qs(urlparse(redirect_url).query)
        code = qs["code"][0]

        # Fetch token and resolve the future
        token = await flow_state.client.fetch_token(
            url=flow_state.config.token_url,
            code=code,
            code_verifier=flow_state.verifier,
            state=state,
        )
        flow_state.future.set_result(token)

    # Run authenticate and redirect simulation concurrently
    redirect_task = asyncio.create_task(_simulate_redirect())
    ctx = await handler.authenticate(cfg_flow, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)
    await redirect_task

    # Assertions
    assert "Authorization" in ctx.headers
    token_val = ctx.headers["Authorization"].split()[1]
    assert token_val in mock_server.tokens

    # After completion, execution should be back to running
    assert record.status == ExecutionStatus.RUNNING
    # All flow state cleaned up
    assert worker._outstanding_flows == {}


@pytest.mark.usefixtures("set_nat_config_file_env_var")
async def test_http_oauth2_flow_publishes_stream_event(monkeypatch, mock_server):
    """When a stream_queue is provided, a StreamOAuthEvent is pushed."""

    redirect_port = _free_port()
    mock_server.register_client(
        client_id="cid",
        client_secret="secret",
        redirect_base=f"http://localhost:{redirect_port}",
    )

    cfg_nat = Config(workflow=EchoFunctionConfig())
    worker = FastApiFrontEndPluginWorker(cfg_nat)

    store = ExecutionStore()
    record = await store.create_execution()
    stream_queue: asyncio.Queue = asyncio.Queue()

    handler = _HTTPFlowHandler(
        oauth_server=mock_server,
        add_flow_cb=worker._add_flow,
        remove_flow_cb=worker._remove_flow,
    )
    handler.set_execution_context(
        execution_id=record.execution_id,
        store=store,
        stream_queue=stream_queue,
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

    async def _simulate_redirect():
        await record.first_outcome.wait()
        state = record.pending_oauth.oauth_state
        flow_state = worker._outstanding_flows[state]

        async with httpx.AsyncClient(
                transport=ASGITransport(app=mock_server._app),
                base_url="http://testserver",
                follow_redirects=False,
                timeout=10,
        ) as client:
            r = await client.get(record.pending_oauth.auth_url)
            redirect_url = r.headers["location"]

        qs = parse_qs(urlparse(redirect_url).query)
        code = qs["code"][0]
        token = await flow_state.client.fetch_token(
            url=flow_state.config.token_url,
            code=code,
            code_verifier=flow_state.verifier,
            state=state,
        )
        flow_state.future.set_result(token)

    redirect_task = asyncio.create_task(_simulate_redirect())
    await handler.authenticate(cfg_flow, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)
    await redirect_task

    # The stream queue should have a StreamOAuthEvent
    from nat.data_models.interactive_http import StreamOAuthEvent
    event = stream_queue.get_nowait()
    assert isinstance(event, StreamOAuthEvent)
    assert event.execution_id == record.execution_id
    assert "oauth" in event.auth_url
