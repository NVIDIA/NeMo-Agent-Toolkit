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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import httpx2
import pytest

from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.data_models.authentication import AuthFlowType
from nat.plugins.mcp.auth.auth_flow_handler import MCPAuthenticationFlowHandler


@pytest.mark.parametrize("resource", [None, "https://resource.example/mcp"])
async def test_mcp_successful_callback(monkeypatch: pytest.MonkeyPatch, resource: str | None):
    handler = MCPAuthenticationFlowHandler()
    client = MagicMock()
    client.fetch_token = AsyncMock(return_value={"access_token": "fixture-token"})
    client.create_authorization_url.side_effect = lambda url, **kw: (url, kw["state"])
    monkeypatch.setattr(handler, "construct_oauth_client", lambda _: client)
    monkeypatch.setattr(handler, "_start_redirect_server", AsyncMock())
    monkeypatch.setattr(handler, "_stop_redirect_server", AsyncMock())
    opened = asyncio.Event()
    monkeypatch.setattr("webbrowser.open", lambda _: opened.set())

    config = OAuth2AuthCodeFlowProviderConfig(
        client_id="test-client",
        client_secret="test-secret",
        authorization_url="https://issuer.example/authorize",
        token_url="https://issuer.example/token",
        redirect_uri="http://localhost:8000/auth/redirect",
        authorization_kwargs={"resource": resource} if resource is not None else None,
    )

    task = asyncio.create_task(handler.authenticate(config, AuthFlowType.OAUTH2_AUTHORIZATION_CODE))
    try:
        await asyncio.wait_for(opened.wait(), timeout=5)
        state = client.create_authorization_url.call_args.kwargs["state"]
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=handler.redirect_app),
                                      base_url="http://localhost:8000") as browser:
            await browser.get("/auth/redirect", params={"code": "fixture-code", "state": state})
        result = await asyncio.wait_for(task, timeout=5)
        assert result.headers == {"Authorization": "Bearer fixture-token"}
        client.fetch_token.assert_awaited_once()
        params = client.fetch_token.await_args.kwargs
        assert params["state"] == state
        assert params["url"] == config.token_url
        if resource is None:
            assert "resource" not in params
        else:
            assert params["resource"] == resource
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
