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

import pytest

from nat.authentication.http_basic_auth.http_basic_auth_provider import HTTPBasicAuthProvider
from nat.authentication.http_basic_auth.register import HTTPBasicAuthProviderConfig
from nat.builder.context import Context
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import BasicAuthCred
from nat.data_models.authentication import BearerTokenCred

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _patch_context(monkeypatch: pytest.MonkeyPatch, callback):
    """Replace AIQContext.get() so the exchanger sees *our* callback."""

    class _DummyCtx:

        def __init__(self, cb):
            self.user_auth_callback = cb

    monkeypatch.setattr(Context, "get", staticmethod(lambda: _DummyCtx(callback)), raising=True)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #


async def test_success(monkeypatch):
    """Happy-path: callback supplies username/password and Authorization header."""

    async def cb(cfg, flow):  # noqa: D401
        assert flow is AuthFlowType.HTTP_BASIC
        return AuthenticatedContext(
            headers={"Authorization": "Basic dXNlcjpwYXNz"},  # base64("user:pass")
            metadata={
                "username": "user", "password": "pass"
            },
        )

    _patch_context(monkeypatch, cb)

    exchanger = HTTPBasicAuthProvider(HTTPBasicAuthProviderConfig())
    res = await exchanger.authenticate(user_id="42")

    # two credentials: BasicAuthCred + BearerTokenCred
    assert len(res.credentials) == 2
    basic, bearer = res.credentials
    assert isinstance(basic, BasicAuthCred)
    assert isinstance(bearer, BearerTokenCred)
    assert basic.username.get_secret_value() == "user"
    assert basic.password.get_secret_value() == "pass"
    assert bearer.scheme == "Basic"
    assert bearer.token.get_secret_value() == "dXNlcjpwYXNz"


async def test_caching(monkeypatch):
    """Second call with same user_id should NOT re-invoke the callback."""
    hits = {"n": 0}

    async def cb(cfg, flow):  # noqa: D401
        hits["n"] += 1
        return AuthenticatedContext(
            headers={"Authorization": "Basic YQ=="},
            metadata={
                "username": "a", "password": "b"
            },
        )

    _patch_context(monkeypatch, cb)

    exchanger = HTTPBasicAuthProvider(HTTPBasicAuthProviderConfig())
    await exchanger.authenticate("dup")
    await exchanger.authenticate("dup")  # should use cached result

    assert hits["n"] == 1


async def test_missing_authorization_header(monkeypatch):
    """Callback returns no `Authorization` header → RuntimeError."""

    async def cb(cfg, flow):  # noqa: D401
        return AuthenticatedContext(headers={}, metadata={})

    _patch_context(monkeypatch, cb)

    exchanger = HTTPBasicAuthProvider(HTTPBasicAuthProviderConfig())

    with pytest.raises(RuntimeError, match="No Authorization header"):
        await exchanger.authenticate("u123")


async def test_callback_exception_bubbles(monkeypatch):
    """Errors in the callback are wrapped in a helpful RuntimeError."""

    async def cb(cfg, flow):  # noqa: D401
        raise RuntimeError("frontend blew up")

    _patch_context(monkeypatch, cb)

    exchanger = HTTPBasicAuthProvider(HTTPBasicAuthProviderConfig())

    with pytest.raises(RuntimeError, match="Authentication callback failed"):
        await exchanger.authenticate("u456")
