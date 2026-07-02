# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
import socket
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import httpx
import pytest
from httpx import ASGITransport
from mock_oauth2_server import MockOAuth2Server

from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.data_models.authentication import AuthFlowType
from nat.front_ends.console.authentication_flow_handler import ConsoleAuthenticationFlowHandler
from nat.front_ends.console.console_front_end_plugin import ConsoleFrontEndPlugin


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _TestHandler(ConsoleAuthenticationFlowHandler):
    """
    Override *one* factory so the OAuth2 client talks to the in‑process
    FastAPI mock (no real network), everything else kept intact.
    """

    def __init__(self, oauth_server: MockOAuth2Server):
        super().__init__()
        self._oauth_server = oauth_server

    def construct_oauth_client(self, cfg):
        transport = ASGITransport(app=self._oauth_server._app)
        from authlib.integrations.httpx_client import AsyncOAuth2Client

        client = AsyncOAuth2Client(
            client_id=cfg.client_id,
            client_secret=cfg.client_secret.get_secret_value(),
            redirect_uri=cfg.redirect_uri,
            scope=" ".join(cfg.scopes) if cfg.scopes else None,
            token_endpoint=cfg.token_url,
            base_url="http://testserver",  # matches host passed below
            transport=transport,
        )
        self._oauth_client = client
        return client

    async def _start_redirect_server(self) -> None:
        # Dont start the uvicorn server
        pass

    async def _stop_redirect_server(self) -> None:
        # Dont stop the uvicorn server
        pass


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

    cfg = OAuth2AuthCodeFlowProviderConfig(
        client_id="cid",
        client_secret="secret",
        authorization_url="http://testserver/oauth/authorize",
        token_url="http://testserver/oauth/token",
        scopes=["read"],
        use_pkce=True,
        redirect_uri=f"http://localhost:{redirect_port}/auth/redirect",
    )

    handler = _TestHandler(mock_server)

    # ----------------- patch browser ---------------------------------- #
    opened: list[str] = []

    async def _drive(url: str):
        opened.append(url)
        # 1) hit mock auth server (ASGI)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=mock_server._app),
            base_url="http://testserver",
            follow_redirects=False,
            timeout=10,
        ) as c:
            r = await c.get(url)
            assert r.status_code == 302
            redirect_url = r.headers["location"]

        # 2) follow redirect to handler's in‑memory FastAPI app
        #    (wait until it exists – very quick)
        while handler.redirect_app is None:
            await asyncio.sleep(0.01)

        async with httpx.AsyncClient(
                transport=ASGITransport(app=handler.redirect_app),
                base_url="http://localhost",
                follow_redirects=True,
                timeout=10,
        ) as c:
            await c.get(redirect_url)

    monkeypatch.setattr("webbrowser.open", lambda url, *_: asyncio.create_task(_drive(url)), raising=True)
    monkeypatch.setattr("click.echo", lambda *_: None, raising=True)  # silence CLI

    # ----------------- run flow ---------------------------------------- #
    ctx = await handler.authenticate(cfg, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)

    # ----------------- assertions -------------------------------------- #
    assert opened, "Browser was never opened"
    tok = ctx.headers["Authorization"].split()[1]
    assert tok in mock_server.tokens  # issued by mock server

    # internal cleanup
    assert handler._active_flows == 0
    assert not handler._flows


# --------------------------------------------------------------------------- #
# Error Recovery Tests                                                        #
# --------------------------------------------------------------------------- #
async def test_console_oauth2_flow_error_handling(monkeypatch, mock_server):
    """Test that Console flow does NOT convert OAuth client creation errors to RuntimeError (inconsistent behavior)."""

    # Create a handler that will fail during OAuth client construction
    class _FailingTestHandler(ConsoleAuthenticationFlowHandler):

        def __init__(self):
            super().__init__()

        def construct_oauth_client(self, cfg):
            # Force a failure during OAuth client creation
            raise ValueError("Invalid OAuth client configuration")

    cfg = OAuth2AuthCodeFlowProviderConfig(
        client_id="test_client",
        client_secret="test_secret",
        authorization_url="http://testserver/oauth/authorize",
        token_url="http://testserver/oauth/token",
        scopes=["read"],
        use_pkce=True,
        redirect_uri="http://localhost:8000/auth/redirect",
    )

    handler = _FailingTestHandler()

    monkeypatch.setattr("webbrowser.open", lambda *_: None, raising=True)  # Don't actually open browser
    monkeypatch.setattr("click.echo", lambda *_: None, raising=True)  # silence CLI

    # Assert that ValueError is raised (NOT converted to RuntimeError - demonstrates inconsistent error handling)
    with pytest.raises(ValueError) as exc_info:
        await handler.authenticate(cfg, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)

    # Verify the error message contains the original exception information
    error_message = str(exc_info.value)
    assert "Invalid OAuth client configuration" in error_message


# ---------------------------------------------------------------------------
# ConsoleFrontEndPlugin - preflight authentication
# ---------------------------------------------------------------------------


def _sm(auth_providers: dict) -> MagicMock:
    sm = MagicMock()
    sm.config.authentication = auth_providers
    sm.shared_builder.get_auth_provider = AsyncMock()
    return sm


def _cfg(preflight_auth: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.preflight_auth = preflight_auth
    return cfg


async def _run(sm: MagicMock) -> None:
    """Call _run_preflight_auth without instantiating the full plugin."""
    await ConsoleFrontEndPlugin._run_preflight_auth(MagicMock(), sm)


async def test_console_preflight_auth_authenticates_single_provider():
    provider = AsyncMock()
    sm = _sm({"provider_a": _cfg()})
    sm.shared_builder.get_auth_provider.return_value = provider

    await _run(sm)

    sm.shared_builder.get_auth_provider.assert_awaited_once_with("provider_a")
    provider.authenticate.assert_awaited_once()


async def test_console_preflight_auth_authenticates_all_preflight_providers():
    provider_a, provider_b = AsyncMock(), AsyncMock()
    sm = _sm({"provider_a": _cfg(), "provider_b": _cfg()})
    sm.shared_builder.get_auth_provider.side_effect = [provider_a, provider_b]

    await _run(sm)

    assert sm.shared_builder.get_auth_provider.await_count == 2
    provider_a.authenticate.assert_awaited_once()
    provider_b.authenticate.assert_awaited_once()


async def test_console_preflight_auth_skips_non_preflight_providers():
    provider_a = AsyncMock()
    sm = _sm({"provider_a": _cfg(), "provider_b": _cfg(preflight_auth=False)})
    sm.shared_builder.get_auth_provider.return_value = provider_a

    await _run(sm)

    sm.shared_builder.get_auth_provider.assert_awaited_once_with("provider_a")


async def test_console_preflight_auth_logs_warning_and_does_not_raise_on_failure(caplog):
    provider = AsyncMock()
    provider.authenticate.side_effect = RuntimeError("OAuth server unreachable")
    sm = _sm({"provider_a": _cfg()})
    sm.shared_builder.get_auth_provider.return_value = provider

    with caplog.at_level(logging.WARNING):
        await _run(sm)

    assert any("provider_a" in r.message for r in caplog.records)


async def test_console_preflight_auth_continues_remaining_providers_after_one_fails():
    provider_a, provider_b = AsyncMock(), AsyncMock()
    provider_a.authenticate.side_effect = RuntimeError("a failed")
    sm = _sm({"provider_a": _cfg(), "provider_b": _cfg()})
    sm.shared_builder.get_auth_provider.side_effect = [provider_a, provider_b]

    await _run(sm)

    provider_a.authenticate.assert_awaited_once()
    provider_b.authenticate.assert_awaited_once()
