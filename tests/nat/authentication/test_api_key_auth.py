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

import pytest

# --------------------------------------------------------------------------- #
# Import the modules we are testing
# --------------------------------------------------------------------------- #
from nat.authentication.api_key import api_key_auth_provider
from nat.authentication.api_key import api_key_auth_provider_config
from nat.builder.workflow_builder import WorkflowBuilder

# Handy names
APIKeyAuthProviderConfig = api_key_auth_provider_config.APIKeyAuthProviderConfig
HeaderAuthScheme = api_key_auth_provider_config.HeaderAuthScheme
APIKeyFieldError = api_key_auth_provider_config.APIKeyFieldError
HeaderNameFieldError = api_key_auth_provider_config.HeaderNameFieldError
HeaderPrefixFieldError = api_key_auth_provider_config.HeaderPrefixFieldError
APIKeyAuthProvider = api_key_auth_provider.APIKeyAuthProvider
BearerTokenCred = api_key_auth_provider.BearerTokenCred
AuthResult = api_key_auth_provider.AuthResult


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_config(
    *,
    raw_key: str = "superSecretAPIKey",
    scheme: HeaderAuthScheme = HeaderAuthScheme.BEARER,
    header_name: str | None = "Authorization",
    header_prefix: str | None = "Bearer",
) -> APIKeyAuthProviderConfig:
    """Factory producing a valid APIKeyAuthProviderConfig for the given scheme."""
    return APIKeyAuthProviderConfig(
        raw_key=raw_key,
        auth_scheme=scheme,
        custom_header_name=header_name,
        custom_header_prefix=header_prefix,
    )


# --------------------------------------------------------------------------- #
# APIKeyAuthProviderConfig – validation tests
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
    assert cfg.custom_header_name == "X-Custom-Auth"
    assert cfg.custom_header_prefix == "Token"


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
# APIKeyAuthProvider – _construct_authentication_header
# --------------------------------------------------------------------------- #
async def test_construct_header_bearer(monkeypatch: pytest.MonkeyPatch):  # pylint:disable=unused-argument
    cfg = make_config()

    async with WorkflowBuilder() as builder:

        provider = await builder.add_auth_provider(name="test", config=cfg)

        result = await provider.authenticate(user_id="1")

        assert isinstance(result.credentials[0], BearerTokenCred)

        cred: BearerTokenCred = result.credentials[0]

        assert cred.header_name == "Authorization"
        assert cred.scheme == "Bearer"
        assert cred.token.get_secret_value() == cfg.raw_key


async def test_construct_header_x_api_key():
    cfg = make_config(
        scheme=HeaderAuthScheme.X_API_KEY,
        header_name="X-API-KEY",
        header_prefix="X-API-KEY",
    )

    async with WorkflowBuilder() as builder:

        provider = await builder.add_auth_provider(name="test", config=cfg)

        result = await provider.authenticate(user_id="1")

        assert isinstance(result.credentials[0], BearerTokenCred)

        cred: BearerTokenCred = result.credentials[0]

        assert cred.scheme == "X-API-Key"
        assert cred.header_name == ""  # per implementation
        assert cred.token.get_secret_value() == cfg.raw_key


async def test_construct_header_custom():
    cfg = make_config(
        scheme=HeaderAuthScheme.CUSTOM,
        header_name="X-Custom",
        header_prefix="Token",
    )

    async with WorkflowBuilder() as builder:

        provider = await builder.add_auth_provider(name="test", config=cfg)

        result = await provider.authenticate(user_id="1")

        assert isinstance(result.credentials[0], BearerTokenCred)

        cred: BearerTokenCred = result.credentials[0]

        assert cred.header_name == "X-Custom"
        assert cred.scheme == "Token"
        assert cred.token.get_secret_value() == cfg.raw_key
