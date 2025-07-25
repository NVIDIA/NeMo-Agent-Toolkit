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

import importlib
import sys
from types import ModuleType
from typing import Any

import pytest

import aiq.authentication.api_key.api_key_client as api_key_client
# --------------------------------------------------------------------------- #
# Import the modules we are testing
# --------------------------------------------------------------------------- #
import aiq.authentication.api_key.api_key_config as api_key_config

# Handy names
APIKeyConfig = api_key_config.APIKeyConfig
HeaderAuthScheme = api_key_config.HeaderAuthScheme
APIKeyFieldError = api_key_config.APIKeyFieldError
HeaderNameFieldError = api_key_config.HeaderNameFieldError
HeaderPrefixFieldError = api_key_config.HeaderPrefixFieldError
APIKeyClient = api_key_client.APIKeyClient
BearerTokenCred = api_key_client.BearerTokenCred
AuthResult = api_key_client.AuthResult


# --------------------------------------------------------------------------- #
# Patching helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _patch_interfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Several symbols live in `aiq.authentication.interfaces`.  Create a dummy
    module with the bare minimum so we can import/patch it without the full
    runtime.
    """
    dummy = ModuleType("aiq.authentication.interfaces")
    dummy.AUTHORIZATION_HEADER = "Authorization"

    # A no-op base class so APIKeyClient super().__init__() succeeds
    class _DummyBase:  # noqa: D401  (simple / imperative name)

        def __init__(self, config: Any) -> None:
            self.config = config

    dummy.AuthenticationClientBase = _DummyBase

    # Expose the dummy in sys.modules **before** the client under test is used.
    sys.modules["aiq.authentication.interfaces"] = dummy

    # Re-import the client module so it picks up the patched base/constant.
    importlib.reload(api_key_client)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_config(
    *,
    raw_key: str = "superSecretAPIKey",
    scheme: HeaderAuthScheme = HeaderAuthScheme.BEARER,
    header_name: str | None = "Authorization",
    header_prefix: str | None = "Bearer",
) -> APIKeyConfig:
    """Factory producing a valid APIKeyConfig for the given scheme."""
    return APIKeyConfig(
        raw_key=raw_key,
        auth_scheme=scheme,
        header_name=header_name,
        header_prefix=header_prefix,
    )


# --------------------------------------------------------------------------- #
# APIKeyConfig – validation tests
# --------------------------------------------------------------------------- #
def test_config_valid_bearer():
    cfg = make_config()
    assert cfg.raw_key == "superSecretAPIKey"
    assert cfg.auth_scheme is HeaderAuthScheme.BEARER


def test_config_valid_x_api_key():
    cfg = make_config(
        scheme=HeaderAuthScheme.X_API_KEY,
        header_name="X-API-KEY",
        header_prefix="X-API-KEY",
    )
    assert cfg.auth_scheme is HeaderAuthScheme.X_API_KEY


def test_config_valid_custom():
    cfg = make_config(
        scheme=HeaderAuthScheme.CUSTOM,
        header_name="X-Custom-Auth",
        header_prefix="Token",
    )
    assert cfg.header_name == "X-Custom-Auth"
    assert cfg.header_prefix == "Token"


@pytest.mark.parametrize("bad_key", ["short", " white space ", "bad key\n"])
def test_config_invalid_raw_key(bad_key):
    with pytest.raises(APIKeyFieldError):
        make_config(raw_key=bad_key)


def test_config_invalid_header_name_format():
    with pytest.raises(HeaderNameFieldError):
        make_config(header_name="Bad Header")  # contains space


def test_config_invalid_header_prefix_nonascii():
    with pytest.raises(HeaderPrefixFieldError):
        make_config(header_prefix="préfix")  # non-ASCII


# --------------------------------------------------------------------------- #
# APIKeyClient – _construct_authentication_header
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_construct_header_bearer(monkeypatch: pytest.MonkeyPatch):
    cfg = make_config()
    client = APIKeyClient(cfg)

    hdr: BearerTokenCred = await client._construct_authentication_header()  # type: ignore[attr-defined]

    assert hdr.header_name == "Authorization"
    assert hdr.scheme == "Bearer"
    assert hdr.token.get_secret_value() == cfg.raw_key


@pytest.mark.asyncio
async def test_construct_header_x_api_key():
    cfg = make_config(
        scheme=HeaderAuthScheme.X_API_KEY,
        header_name="X-API-KEY",
        header_prefix="X-API-KEY",
    )
    client = APIKeyClient(cfg)
    hdr: BearerTokenCred = await client._construct_authentication_header()  # type: ignore[attr-defined]

    assert hdr.scheme == "X-API-Key"
    assert hdr.header_name == ""  # per implementation
    assert hdr.token.get_secret_value() == cfg.raw_key


@pytest.mark.asyncio
async def test_construct_header_custom():
    cfg = make_config(
        scheme=HeaderAuthScheme.CUSTOM,
        header_name="X-Custom",
        header_prefix="Token",
    )
    client = APIKeyClient(cfg)
    hdr: BearerTokenCred = await client._construct_authentication_header()  # type: ignore[attr-defined]

    assert hdr.header_name == "X-Custom"
    assert hdr.scheme == "Token"
    assert hdr.token.get_secret_value() == cfg.raw_key


# --------------------------------------------------------------------------- #
# APIKeyClient – authenticate high-level method
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_authenticate_returns_authresult():
    cfg = make_config()
    client = APIKeyClient(cfg)
    res: AuthResult = await client.authenticate(user_id="user-123")  # type: ignore[attr-defined]

    assert isinstance(res, AuthResult)
    assert len(res.credentials) == 1
    cred: BearerTokenCred = res.credentials[0]  # type: ignore[assignment]
    assert cred.token.get_secret_value() == cfg.raw_key
