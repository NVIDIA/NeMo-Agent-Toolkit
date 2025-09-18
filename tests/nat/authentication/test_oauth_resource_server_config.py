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
from pydantic import ValidationError

from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig


# ---------- Base fixture ----------
@pytest.fixture
def base_config() -> OAuth2ResourceServerConfig:
    """
    Minimal valid baseline:
      - issuer_url is HTTPS
      - all other fields None/empty
    """
    return OAuth2ResourceServerConfig(
        issuer_url="https://issuer.example.com",
        scopes=[],
        audience=None,
        jwks_uri=None,
        discovery_url=None,
        introspection_endpoint=None,
        client_id=None,
        client_secret=None,
    )


def _build_from(base: OAuth2ResourceServerConfig, **updates) -> OAuth2ResourceServerConfig:
    data = base.model_dump()
    data.update(updates)
    return OAuth2ResourceServerConfig(**data)


# ===============================
# issuer_url
# ===============================
@pytest.mark.parametrize(
    "value",
    [
        "https://issuer.example.com",
        "http://localhost:8080",  # localhost can be http
        "https://issuer.example.com/",  # trailing slash OK
    ],
)
def test_issuer_url_valid(base_config: OAuth2ResourceServerConfig, value: str):
    cfg = _build_from(base_config, issuer_url=value)
    assert cfg.issuer_url == value


@pytest.mark.parametrize(
    "value",
    [
        "http://example.com",  # remote + http (not localhost)
        "",
        "ftp://issuer.example.com",
    ],
)
def test_issuer_url_invalid(base_config: OAuth2ResourceServerConfig, value: str):
    with pytest.raises(ValidationError):
        _build_from(base_config, issuer_url=value)


# ===============================
# scopes
# ===============================
@pytest.mark.parametrize(
    "value",
    [
        [],
        ["read"],
        ["read", "write"],
    ],
)
def test_scopes_valid(base_config: OAuth2ResourceServerConfig, value):
    cfg = _build_from(base_config, scopes=value)
    assert cfg.scopes == value


@pytest.mark.parametrize(
    "value",
    [
        "read write",  # must be list[str]
        [1, 2],  # must be list[str]
        None,  # pydantic coerces? enforce list or default—treat None as invalid here
    ],
)
def test_scopes_invalid(base_config: OAuth2ResourceServerConfig, value):
    with pytest.raises(ValidationError):
        _build_from(base_config, scopes=value)


# ===============================
# audience
# ===============================
@pytest.mark.parametrize(
    "value",
    [
        None,
        "api://resource",
        "https://example.com/my-api",
    ],
)
def test_audience_valid(base_config: OAuth2ResourceServerConfig, value):
    cfg = _build_from(base_config, audience=value)
    assert cfg.audience == value


@pytest.mark.parametrize(
    "value",
    [
        123,
        ["not-a-string"],
        {
            "aud": "x"
        },
    ],
)
def test_audience_invalid(base_config: OAuth2ResourceServerConfig, value):
    with pytest.raises(ValidationError):
        _build_from(base_config, audience=value)


# ===============================
# jwks_uri
# ===============================
@pytest.mark.parametrize(
    "value",
    [
        None,
        "https://issuer.example.com/.well-known/jwks.json",
        "http://localhost/.well-known/jwks.json",  # localhost can be http
    ],
)
def test_jwks_uri_valid(base_config: OAuth2ResourceServerConfig, value):
    cfg = _build_from(base_config, jwks_uri=value)
    assert cfg.jwks_uri == value


@pytest.mark.parametrize(
    "value",
    [
        "http://example.com/.well-known/jwks.json",  # remote + http
        "gopher://issuer/.well-known/jwks.json",
        "ftp://issuer/.well-known/jwks.json",
    ],
)
def test_jwks_uri_invalid(base_config: OAuth2ResourceServerConfig, value):
    with pytest.raises(ValidationError):
        _build_from(base_config, jwks_uri=value)


# ===============================
# discovery_url
# ===============================
@pytest.mark.parametrize(
    "value",
    [
        None,
        "https://issuer.example.com/.well-known/openid-configuration",
        "http://localhost/.well-known/openid-configuration",  # localhost can be http
    ],
)
def test_discovery_url_valid(base_config: OAuth2ResourceServerConfig, value):
    cfg = _build_from(base_config, discovery_url=value)
    assert cfg.discovery_url == value


@pytest.mark.parametrize(
    "value",
    [
        "http://example.com/.well-known/openid-configuration",  # remote + http
        "ftp://issuer/.well-known/openid-configuration",
    ],
)
def test_discovery_url_invalid(base_config: OAuth2ResourceServerConfig, value):
    with pytest.raises(ValidationError):
        _build_from(base_config, discovery_url=value)


# ===============================
# introspection_endpoint
# ===============================
@pytest.mark.parametrize(
    "value",
    [
        None,  # not enabling opaque path is fine
        # Valid remote HTTPS with required creds supplied inside the test body
        "https://issuer.example.com/oauth2/introspect",  # Localhost can be http
        "http://localhost/oauth2/introspect",
    ],
)
def test_introspection_endpoint_valid(base_config: OAuth2ResourceServerConfig, value: str):
    if value is None:
        cfg = _build_from(base_config, introspection_endpoint=None)
        assert cfg.introspection_endpoint is None
        return

    # Supply required deps here so parametrization still only passes the field values
    cfg = _build_from(
        base_config,
        introspection_endpoint=value,
        client_id="client-abc",
        client_secret="secret-xyz",
    )
    assert cfg.introspection_endpoint == value
    assert cfg.client_id == "client-abc"
    assert cfg.client_secret == "secret-xyz"


@pytest.mark.parametrize(
    "value",
    [
        # Remote non-https (not localhost) should be rejected even with creds
        "http://example.com/oauth2/introspect",  # Also treat weird schemes as invalid
        "ftp://issuer.example.com/oauth2/introspect",
    ],
)
def test_introspection_endpoint_invalid_url(base_config: OAuth2ResourceServerConfig, value: str):
    with pytest.raises(ValidationError):
        _build_from(
            base_config,
            introspection_endpoint=value,
            client_id="client-abc",
            client_secret="secret-xyz",
        )


@pytest.mark.parametrize(
    "client_id,client_secret",
    [
        (None, None),
        ("client-abc", None),
        (None, "secret-xyz"),
    ],
)
def test_introspection_endpoint_missing_credentials_invalid(base_config: OAuth2ResourceServerConfig,
                                                            client_id,
                                                            client_secret):
    with pytest.raises(ValidationError):
        _build_from(
            base_config,
            introspection_endpoint="https://issuer.example.com/oauth2/introspect",
            client_id=client_id,
            client_secret=client_secret,
        )


# ===============================
# client_id
# ===============================
@pytest.mark.parametrize(
    "value",
    [
        None,  # fine when introspection_endpoint not set
        "client-abc",  # fine when introspection not set
    ],
)
def test_client_id_valid_without_introspection(base_config: OAuth2ResourceServerConfig, value):
    cfg = _build_from(base_config, client_id=value, introspection_endpoint=None, client_secret=None)
    assert cfg.client_id == value


@pytest.mark.parametrize(
    "value",
    [
        None,  # invalid if introspection_endpoint set (and secret provided)
        "client-abc",  # we’ll set endpoint but **omit** secret to trigger invalid (missing secret)
    ],
)
def test_client_id_invalid_with_introspection_when_counterpart_missing(base_config: OAuth2ResourceServerConfig, value):
    # If value is None -> missing id; if value is str -> we will omit secret
    with pytest.raises(ValidationError):
        _build_from(
            base_config,
            introspection_endpoint="https://issuer.example.com/oauth2/introspect",
            client_id=value,
            client_secret=None,  # intentionally missing
        )


# ===============================
# client_secret
# ===============================
@pytest.mark.parametrize(
    "value",
    [
        None,  # fine when introspection_endpoint not set
        "secret-xyz",  # fine when introspection not set
    ],
)
def test_client_secret_valid_without_introspection(base_config: OAuth2ResourceServerConfig, value):
    cfg = _build_from(base_config, client_secret=value, introspection_endpoint=None, client_id=None)
    assert cfg.client_secret == value


@pytest.mark.parametrize(
    "value",
    [
        None,  # invalid if introspection_endpoint set (and id provided)
        "secret-xyz",  # we’ll set endpoint but **omit** id to trigger invalid (missing id)
    ],
)
def test_client_secret_invalid_with_introspection_when_counterpart_missing(base_config: OAuth2ResourceServerConfig,
                                                                           value):
    with pytest.raises(ValidationError):
        _build_from(
            base_config,
            introspection_endpoint="https://issuer.example.com/oauth2/introspect",
            client_id=None,  # intentionally missing
            client_secret=value,
        )
