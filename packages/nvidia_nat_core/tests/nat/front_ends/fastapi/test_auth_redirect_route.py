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

from fastapi import FastAPI
from httpx import ASGITransport
from httpx import AsyncClient

from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import FlowState
from nat.front_ends.fastapi.routes.auth import add_authorization_route

_CALLBACK_PATH = "/auth/redirect"
_RETURN_URL = "http://localhost:3000"

_POPUP_CONFIG = OAuth2AuthCodeFlowProviderConfig(
    client_id="cid",
    client_secret="secret",
    authorization_url="http://testserver/oauth/authorize",
    token_url="http://testserver/oauth/token",
    redirect_uri="http://localhost:8000/auth/redirect",
    use_redirect_auth=False,
)

_REDIRECT_CONFIG = OAuth2AuthCodeFlowProviderConfig(
    client_id="cid",
    client_secret="secret",
    authorization_url="http://testserver/oauth/authorize",
    token_url="http://testserver/oauth/token",
    redirect_uri="http://localhost:8000/auth/redirect",
    use_redirect_auth=True,
)


def _make_worker(flow_state: FlowState):
    flows = {"teststate": flow_state}

    class _Worker:
        front_end_config = type("cfg", (), {"oauth2_callback_path": _CALLBACK_PATH})()
        _outstanding_flows_lock = asyncio.Lock()
        _outstanding_flows = flows

        async def _remove_flow(self, state: str) -> None:
            flows.pop(state, None)

    return _Worker()


async def _get(worker, params: dict):
    app = FastAPI()
    await add_authorization_route(worker, app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        return await client.get(_CALLBACK_PATH, params=params)


async def test_invalid_state_returns_400():
    """An unknown state is rejected with 400 before any flow processing."""
    flow_state = FlowState(config=_POPUP_CONFIG)
    worker = _make_worker(flow_state)
    response = await _get(worker, {"state": "no-such-state", "code": "abc"})
    assert response.status_code == 400


async def test_access_denied_popup_returns_cancelled_html():
    """error=access_denied in popup mode returns the cancelled popup HTML."""
    flow_state = FlowState(config=_POPUP_CONFIG, return_url=None)
    worker = _make_worker(flow_state)
    response = await _get(worker, {"state": "teststate", "error": "access_denied"})
    assert response.status_code == 200
    assert "AUTH_CANCELLED" in response.text
    assert "AUTH_ERROR" not in response.text


async def test_access_denied_popup_with_return_url_still_returns_cancelled_html():
    """error=access_denied in popup mode uses popup HTML even when return_url is set."""
    flow_state = FlowState(config=_POPUP_CONFIG, return_url=_RETURN_URL)
    worker = _make_worker(flow_state)
    response = await _get(worker, {"state": "teststate", "error": "access_denied"})
    assert response.status_code == 200
    assert "AUTH_CANCELLED" in response.text
    assert "AUTH_ERROR" not in response.text
    assert _RETURN_URL.replace("/", "\\u002f") not in response.text
    assert "oauth_auth_completed" not in response.text


async def test_access_denied_redirect_returns_cancelled_html():
    """error=access_denied in redirect mode returns the redirect-back cancelled page."""
    flow_state = FlowState(config=_REDIRECT_CONFIG, return_url=_RETURN_URL)
    worker = _make_worker(flow_state)
    response = await _get(worker, {"state": "teststate", "error": "access_denied"})
    assert response.status_code == 200
    assert _RETURN_URL.replace("/", "\\u002f") in response.text
    assert "AUTH_CANCELLED" not in response.text
    assert "oauth_auth_completed" not in response.text


async def test_access_denied_sets_cancellation_exception_on_future():
    """error=access_denied resolves the future with an 'Authorisation denied' exception."""
    flow_state = FlowState(config=_POPUP_CONFIG, return_url=None)
    worker = _make_worker(flow_state)
    await _get(worker, {"state": "teststate", "error": "access_denied"})
    assert flow_state.future.done()
    assert "Authorisation denied" in str(flow_state.future.exception())
    assert "access_denied" in str(flow_state.future.exception())


async def test_provider_error_popup_returns_error_html():
    """Non-access_denied errors in popup mode return error HTML, not cancelled HTML."""
    flow_state = FlowState(config=_POPUP_CONFIG, return_url=None)
    worker = _make_worker(flow_state)
    response = await _get(worker, {"state": "teststate", "error": "server_error"})
    assert response.status_code == 200
    assert "AUTH_ERROR" in response.text
    assert "AUTH_CANCELLED" not in response.text


async def test_provider_error_redirect_returns_error_html():
    """Non-access_denied errors in redirect mode redirect back without oauth_auth_completed."""
    flow_state = FlowState(config=_REDIRECT_CONFIG, return_url=_RETURN_URL)
    worker = _make_worker(flow_state)
    response = await _get(worker, {"state": "teststate", "error": "server_error"})
    assert response.status_code == 200
    assert _RETURN_URL.replace("/", "\\u002f") in response.text
    assert "AUTH_CANCELLED" not in response.text
    assert "oauth_auth_completed" not in response.text


async def test_provider_error_sets_oauth_error_exception_on_future():
    """Non-access_denied errors resolve the future with an 'OAuth error' exception including the code."""
    flow_state = FlowState(config=_POPUP_CONFIG, return_url=None)
    worker = _make_worker(flow_state)
    await _get(worker, {"state": "teststate", "error": "server_error", "error_description": "internal"})
    assert flow_state.future.done()
    assert "OAuth error" in str(flow_state.future.exception())
    assert "server_error" in str(flow_state.future.exception())
    assert "internal" in str(flow_state.future.exception())
