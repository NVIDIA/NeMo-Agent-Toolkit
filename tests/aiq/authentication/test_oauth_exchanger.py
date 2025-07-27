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

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Awaitable
from typing import Callable

import pytest

from aiq.authentication.oauth2.oauth2_auth_code_flow_provider import OAuth2AuthCodeFlowProvider
from aiq.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from aiq.builder.context import AIQContext
from aiq.data_models.authentication import AuthenticatedContext
from aiq.data_models.authentication import AuthFlowType
from aiq.data_models.authentication import AuthResult
from aiq.data_models.authentication import BearerTokenCred


# --------------------------------------------------------------------------- #
# Helpers / Fixtures
# --------------------------------------------------------------------------- #
def _patch_context(
    monkeypatch: pytest.MonkeyPatch,
    callback: Callable[[OAuth2AuthCodeFlowProviderConfig, AuthFlowType], Awaitable[AuthenticatedContext]],
) -> None:

    class _DummyCtx:

        def __init__(self, cb):
            self.user_auth_callback = cb

    monkeypatch.setattr(AIQContext, "get", staticmethod(lambda: _DummyCtx(callback)), raising=True)


@pytest.fixture()
def cfg() -> OAuth2AuthCodeFlowProviderConfig:
    return OAuth2AuthCodeFlowProviderConfig(client_id="cid",
                                            client_secret="secret",
                                            authorization_url="https://example.com/auth",
                                            token_url="https://example.com/token",
                                            scopes=["openid", "profile"],
                                            use_pkce=True,
                                            client_url="http://localhost:9000")


def _bearer_ctx(token: str, expires_at: datetime) -> AuthenticatedContext:
    return AuthenticatedContext(
        headers={"Authorization": f"Bearer {token}"},
        metadata={
            "expires_at": expires_at,
            "raw_token": {
                "access_token": token, "refresh_token": "refTok"
            },
        },
    )


# --------------------------------------------------------------------------- #
# 1. Config model tests
# --------------------------------------------------------------------------- #
def test_config_redirect_uri_defaults():
    cfg = OAuth2AuthCodeFlowProviderConfig(
        client_id="id",
        client_secret="sec",
        authorization_url="a",
        token_url="t",
    )
    assert cfg.redirect_uri == "http://localhost:8000/auth/redirect"


def test_config_redirect_uri_custom(cfg):
    assert cfg.redirect_uri == "http://localhost:9000/auth/redirect"
    assert cfg.use_pkce is True


# --------------------------------------------------------------------------- #
# 2. Happy-path authentication
# --------------------------------------------------------------------------- #
async def test_authenticate_success(monkeypatch, cfg):
    calls = {"n": 0}

    async def cb(conf, flow):
        calls["n"] += 1
        assert conf is cfg
        assert flow is AuthFlowType.OAUTH2_AUTHORIZATION_CODE
        return _bearer_ctx(
            token="tok",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )

    _patch_context(monkeypatch, cb)

    client = OAuth2AuthCodeFlowProvider(cfg)
    res = await client.authenticate(user_id="u1")

    assert calls["n"] == 1
    assert isinstance(res, AuthResult)
    cred = res.credentials[0]
    assert isinstance(cred, BearerTokenCred)
    assert cred.token.get_secret_value() == "tok"


# --------------------------------------------------------------------------- #
# 3. Caching
# --------------------------------------------------------------------------- #
async def test_authenticate_caches(monkeypatch, cfg):
    calls = {"n": 0}

    async def cb(conf, flow):
        calls["n"] += 1
        return _bearer_ctx(
            token="tok",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )

    _patch_context(monkeypatch, cb)
    client = OAuth2AuthCodeFlowProvider(cfg)

    await client.authenticate("dup")
    await client.authenticate("dup")  # cached

    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# 4. Token refresh succeeds
# --------------------------------------------------------------------------- #
async def test_refresh_expired_token(monkeypatch, cfg):
    future_ts = int((datetime.now(timezone.utc) + timedelta(minutes=20)).timestamp())

    class _DummyAuthlibClient:

        def __init__(self, *args, **kwargs):  # ← NEW
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def refresh_token(self, token_url, refresh_token):
            assert token_url == cfg.token_url
            assert refresh_token == "refTok"
            return {"access_token": "newTok", "expires_at": future_ts}

    # **fixed patch line**
    monkeypatch.setattr(
        "aiq.authentication.oauth2.oauth2_auth_code_flow_provider.AuthlibOAuth2Client",
        _DummyAuthlibClient,
        raising=True,
    )

    async def fail_cb(*_a, **_kw):
        raise RuntimeError("should not hit callback")

    _patch_context(monkeypatch, fail_cb)

    client = OAuth2AuthCodeFlowProvider(cfg)
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    client._authenticated_tokens["bob"] = AuthResult(
        credentials=[BearerTokenCred(token="stale")],  # type: ignore[arg-type]
        token_expires_at=past,
        raw={"refresh_token": "refTok"},
    )

    res = await client.authenticate("bob")
    assert res.credentials[0].token.get_secret_value() == "newTok"


# --------------------------------------------------------------------------- #
# 5. Refresh fails → fallback to callback
# --------------------------------------------------------------------------- #
async def test_refresh_fallback_to_callback(monkeypatch, cfg):

    class _RaisingClient:

        def __init__(self, *args, **kwargs):  # ← NEW
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def refresh_token(self, *_a, **_kw):
            raise RuntimeError("network down")

    # **fixed patch line**
    monkeypatch.setattr(
        "aiq.authentication.oauth2.oauth2_auth_code_flow_provider.AuthlibOAuth2Client",
        _RaisingClient,
        raising=True,
    )

    hits = {"n": 0}

    async def cb(conf, flow):
        hits["n"] += 1
        return _bearer_ctx(
            token="fallbackTok",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )

    _patch_context(monkeypatch, cb)

    client = OAuth2AuthCodeFlowProvider(cfg)
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    client._authenticated_tokens["eve"] = AuthResult(
        credentials=[BearerTokenCred(token="old")],  # type: ignore[arg-type]
        token_expires_at=past,
        raw={"refresh_token": "badTok"},
    )

    res = await client.authenticate("eve")
    assert hits["n"] == 1
    assert res.credentials[0].token.get_secret_value() == "fallbackTok"


# --------------------------------------------------------------------------- #
# 6. Invalid header & callback error paths
# --------------------------------------------------------------------------- #
async def test_invalid_authorization_header(monkeypatch, cfg):

    async def cb(*_a, **_kw):
        return AuthenticatedContext(headers={"Authorization": "Token abc"}, metadata={})

    _patch_context(monkeypatch, cb)
    client = OAuth2AuthCodeFlowProvider(cfg)

    with pytest.raises(RuntimeError, match="Invalid Authorization header"):
        await client.authenticate("bad")


async def test_callback_error(monkeypatch, cfg):

    async def cb(*_a, **_kw):
        raise RuntimeError("frontend crash")

    _patch_context(monkeypatch, cb)

    client = OAuth2AuthCodeFlowProvider(cfg)
    with pytest.raises(RuntimeError):
        await client.authenticate(None)
