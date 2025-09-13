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

import base64
import json
import time
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.authentication.credential_validator.bearer_token_validator import BearerTokenValidator
from nat.data_models.authentication import TokenValidationResult

# -------------------------------
# Test helpers
# -------------------------------


def b64url(obj: Any) -> str:
    raw = obj if isinstance(obj, (bytes, bytearray)) else json.dumps(obj).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


class MockClaims(dict):
    """Mimics Claims enough for our code path."""

    def validate(self, *, leeway: int = 0, **_):
        return None


def make_jwt(alg="RS256", payload: dict | None = None, kid: str | None = "kid1", typ: str = "at+jwt"):
    header = {"alg": alg, "typ": typ}
    if kid:
        header["kid"] = kid
    payload = payload or {
        "iss": "https://issuer.test", "sub": "user123", "aud": "api://default", "exp": int(time.time()) + 3600
    }
    return f"{b64url(header)}.{b64url(payload)}.{b64url(b'sig')}"


# -------------------------------
# _is_jwt_token (structure only)
# -------------------------------


def test_is_jwt_token_shape_only():
    bearer_token_validator = BearerTokenValidator()
    assert bearer_token_validator._is_jwt_token(make_jwt()) is True
    assert bearer_token_validator._is_jwt_token("opaque-token") is False
    assert bearer_token_validator._is_jwt_token("one.two") is False
    assert bearer_token_validator._is_jwt_token("one.two.three.four") is False


# -------------------------------
# HTTPS enforcement
# -------------------------------


def test_require_https_allows_https_and_localhost():
    bearer_token_validator = BearerTokenValidator()
    bearer_token_validator._require_https("https://ok.example", "url")
    bearer_token_validator._require_https("http://localhost:8080", "url")
    bearer_token_validator._require_https("http://127.0.0.1:3000", "url")
    bearer_token_validator._require_https("http://[::1]:3000", "url")


def test_require_https_rejects_plain_http():
    bearer_token_validator = BearerTokenValidator()
    with pytest.raises(ValueError, match="must use HTTPS"):
        bearer_token_validator._require_https("http://example.com", "url")


# -------------------------------
# JWT verification path
# -------------------------------


async def test_verify_jwt_success_returns_result():
    bearer_token_validator = BearerTokenValidator(leeway=45)

    token = make_jwt(
        alg="RS256",
        payload={
            "iss": "https://issuer.test",
            "sub": "user123",
            "aud": "api://default",
            "exp": int(time.time()) + 3600,
            "nbf": int(time.time()),
            "iat": int(time.time()),
        },
        typ="at+jwt",
    )

    # Route: JWT
    # Patch unverified header read to avoid base64 bugs from hand-crafted token if needed
    with patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header") as gethdr, \
         patch.object(bearer_token_validator, "_fetch_jwks", return_value=MagicMock()) as _jwks_mock, \
         patch("nat.authentication.credential_validator.bearer_token_validator.jwt.decode") as mock_decode:  # noqa: E501

        gethdr.return_value = {"alg": "RS256", "typ": "at+jwt"}
        # Set up algorithm cache to allow RS256
        bearer_token_validator._jwks_alg_cache["mock_jwks_uri"] = {"RS256"}

        claims = MockClaims({
            "iss": "https://issuer.test",
            "sub": "user123",
            "aud": "api://default",
            "exp": int(time.time()) + 3600,
            "nbf": int(time.time()),
            "iat": int(time.time()),
            "jti": "abc123",
        })
        mock_decode.return_value = claims

        result = await bearer_token_validator.verify(token)
        assert isinstance(result, TokenValidationResult)
        assert result.active is True
        assert result.issuer == "https://issuer.test"
        assert result.subject == "user123"
        assert result.audience == ["api://default"]
        assert result.token_type == "at+jwt"
        assert result.jti == "abc123"
        assert result.expires_at and isinstance(result.expires_at, int)


async def test_verify_jwt_rejects_alg_none():
    bearer_token_validator = BearerTokenValidator()

    token = make_jwt(alg="none")
    with patch(
            "nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header") as get_header:
        get_header.return_value = {"alg": "none", "typ": "JWT"}
        assert await bearer_token_validator.verify(token) is None


async def test_verify_jwt_rejects_header_alg_not_permitted_by_jwks():
    bearer_token_validator = BearerTokenValidator()

    token = make_jwt(alg="HS256")
    # Use the actual JWKS URI that _resolve_jwks_uri will produce
    jwks_uri = "https://issuer.test/.well-known/jwks.json"
    with patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header") as get_header, \
         patch.object(bearer_token_validator, "_fetch_jwks", return_value=MagicMock()) as _fetch_jwks_mock, \
         patch("nat.authentication.credential_validator.bearer_token_validator.jwt.decode", return_value=MockClaims({  # noqa: E501
             "iss": "https://issuer.test", "sub": "u", "aud": "x", "exp": int(time.time()) + 60
         })):
        # Simulate JWKS-derived allowed algs - only RSA family, reject HS256
        get_header.return_value = {"alg": "HS256", "typ": "JWT"}
        bearer_token_validator._jwks_alg_cache[jwks_uri] = {"RS256"}  # Only RS256 allowed, not HS256
        # This should be rejected by header-alg vs JWKS-family check, not by decode exception
        result = await bearer_token_validator.verify(token)
        assert result is None


async def test_verify_jwt_returns_none_when_no_jwks_available():
    bearer_token_validator = BearerTokenValidator()
    token = make_jwt()

    with patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header") as get_header, \
         patch.object(bearer_token_validator, "_fetch_jwks", return_value=None):  # noqa: E501
        get_header.return_value = {"alg": "RS256", "typ": "JWT"}
        # Don't set up algorithm cache to simulate no key set available
        assert await bearer_token_validator.verify(token) is None


# -------------------------------
# Opaque/introspection path
# -------------------------------


async def test_verify_opaque_success_active_true():
    bearer_token_validator = BearerTokenValidator(
        introspection_endpoint="https://as.test/introspect",
        client_id="cid",
        client_secret="csec",
    )

    with patch("nat.authentication.credential_validator.bearer_token_validator.AsyncOAuth2Client") as mock_client:
        oauth_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = oauth_client
        oauth_client.introspect_token.return_value = {
            "active": True,
            "client_id": "cid",
            "username": "alice",
            "token_type": "access_token",
            "exp": int(time.time()) + 120,
            "nbf": int(time.time()) - 10,
            "iat": int(time.time()) - 10,
            "aud": ["api://default", "other"],
            "iss": "https://issuer.maybe",
            "sub": "user123",
            "jti": "id-7",
            "scope": "read write",
        }
        result = await bearer_token_validator.verify("opaque-token")
        assert isinstance(result, TokenValidationResult)
        assert result.active is True
        assert result.client_id == "cid"
        assert result.username == "alice"
        assert result.audience == ["api://default", "other"]
        assert result.subject == "user123"
        assert result.jti == "id-7"
        assert result.scopes == ["read", "write"]


async def test_verify_opaque_inactive_returns_inactive_result():
    bearer_token_validator = BearerTokenValidator(
        introspection_endpoint="https://as.test/introspect",
        client_id="cid",
        client_secret="csec",
    )
    with patch("nat.authentication.credential_validator.bearer_token_validator.AsyncOAuth2Client") as mock_client:
        oauth_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = oauth_client
        oauth_client.introspect_token.return_value = {"active": False, "client_id": "cid", "token_type": "access_token"}
        result = await bearer_token_validator.verify("t")
        assert isinstance(result, TokenValidationResult)
        assert result.active is False
        assert result.client_id == "cid"


async def test_verify_opaque_missing_credentials_returns_none():
    # Introspection endpoint configured without creds -> verify should return None (no attempt)
    bearer_token_validator = BearerTokenValidator(introspection_endpoint="https://as.test/introspect")
    assert await bearer_token_validator.verify("opaque-token") is None


async def test_verify_opaque_http_error_returns_none():
    bearer_token_validator = BearerTokenValidator(
        introspection_endpoint="https://as.test/introspect",
        client_id="cid",
        client_secret="csec",
    )
    with patch("nat.authentication.credential_validator.bearer_token_validator.AsyncOAuth2Client") as mock_client:
        oauth_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = oauth_client
        oauth_client.introspect_token.side_effect = Exception("network error")
        assert await bearer_token_validator.verify("opaque-token") is None


# -------------------------------
# OIDC Discovery + JWKS fetch (single path + caching)
# -------------------------------


async def test_get_oidc_configuration_success_and_cache():
    bearer_token_validator = BearerTokenValidator()
    with patch("httpx.AsyncClient") as mock_client:
        ac = AsyncMock()
        mock_client.return_value.__aenter__.return_value = ac
        ac.get.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "issuer": "https://issuer.test", "jwks_uri": "https://issuer.test/jwks"
            }),
        )
        config = await bearer_token_validator._get_oidc_configuration(
            "https://issuer.test/.well-known/openid-configuration")
        assert config["issuer"] == "https://issuer.test"
        # subsequent call should be cached
        config2 = await bearer_token_validator._get_oidc_configuration(
            "https://issuer.test/.well-known/openid-configuration")
        assert config2 is config


async def test_fetch_jwks_fetch_cache_and_algorithm_derivation():
    """Test JWKS fetch, caching, and algorithm cache population."""
    bearer_token_validator = BearerTokenValidator()
    mock_jwks_response = {
        "keys": [
            {
                "kty": "RSA",
                "kid": "rsa-key-1",
                "use": "sig",
                "n":
                    "0vx7agoebGcQSuuPiLJXZptN9nndrQmbPFRP_gdHPfNjxbMoVhvvJf7N2Ls_mw-wRsqiLcUyWZBaLjuDN8-mA",  # noqa: E501
                "e": "AQAB"  # Standard RSA exponent
            },
            {
                "kty": "EC",
                "crv": "P-256",
                "kid": "ec-key-1",
                "use": "sig",
                "x": "WKn-ZIGevcwGIyyrzFoZNBdaq9_TsqzGHwHitJBcBmXU",  # Mock x coordinate
                "y": "y77As5vbZdIh6AzjQxDwsLWzzx_LR6qNUfOo-hLSSWU"  # Mock y coordinate
            }
        ]
    }

    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

        jwks_uri = "https://issuer.test/jwks"

        # First call should fetch and cache
        keyset = await bearer_token_validator._fetch_jwks(jwks_uri)
        assert keyset is not None

        # Verify algorithm cache was populated based on key types
        derived_algs = bearer_token_validator._jwks_alg_cache[jwks_uri]
        assert "RS256" in derived_algs  # RSA key should enable RS* algorithms
        assert "PS256" in derived_algs  # RSA key should enable PS* algorithms
        assert "ES256" in derived_algs  # EC P-256 key should enable ES256

        # Second call should return cached keyset
        keyset2 = await bearer_token_validator._fetch_jwks(jwks_uri)
        assert keyset2 is keyset  # Same object reference (cached)


async def test_jwt_header_parse_failure_returns_none():
    bearer_token_validator = BearerTokenValidator()
    invalid_token = "a.b.c"  # 3 parts but not valid base64/json header
    with patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header"
               ) as header_mock:
        header_mock.side_effect = Exception("bad header")
        assert await bearer_token_validator.verify(invalid_token) is None


async def test_jwt_missing_exp_returns_none():
    bearer_token_validator = BearerTokenValidator()
    token = make_jwt(payload={"iss": "https://issuer.test", "sub": "u1", "aud": "api://d"})  # omit exp
    with patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header", return_value={"alg": "RS256"}), \
         patch.object(bearer_token_validator, "_fetch_jwks", return_value=MagicMock()), \
         patch("nat.authentication.credential_validator.bearer_token_validator.jwt.decode") as jwt_decode:  # noqa: E501
        bearer_token_validator._jwks_alg_cache["mock_jwks_uri"] = {"RS256"}
        from authlib.jose.errors import MissingClaimError
        jwt_decode.side_effect = MissingClaimError("exp")
        assert await bearer_token_validator.verify(token) is None


async def test_jwt_expired_returns_none():
    bearer_token_validator = BearerTokenValidator(leeway=0)
    token = make_jwt(payload={"iss": "https://issuer.test", "sub": "u", "aud": "api://d", "exp": int(time.time()) - 5})
    with patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header", return_value={"alg": "RS256"}), \
         patch.object(bearer_token_validator, "_fetch_jwks", return_value=MagicMock()), \
         patch("nat.authentication.credential_validator.bearer_token_validator.jwt.decode") as jwt_decode:  # noqa: E501
        bearer_token_validator._jwks_alg_cache["mock_jwks_uri"] = {"RS256"}
        from authlib.jose.errors import ExpiredTokenError
        jwt_decode.side_effect = ExpiredTokenError("expired")
        assert await bearer_token_validator.verify(token) is None


async def test_jwt_nbf_in_future_returns_none():
    bearer_token_validator = BearerTokenValidator()
    future = int(time.time()) + 3600
    token = make_jwt(payload={
        "iss": "https://issuer.test", "sub": "u", "aud": "api://d", "exp": future + 3600, "nbf": future
    })
    with patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header", return_value={"alg": "RS256"}), \
         patch.object(bearer_token_validator, "_fetch_jwks", return_value=MagicMock()), \
         patch("nat.authentication.credential_validator.bearer_token_validator.jwt.decode") as jwt_decode:  # noqa: E501
        bearer_token_validator._jwks_alg_cache["mock_jwks_uri"] = {"RS256"}
        from authlib.jose.errors import InvalidClaimError
        jwt_decode.side_effect = InvalidClaimError("nbf")
        assert await bearer_token_validator.verify(token) is None


async def test_jwt_audience_normalization_string_and_list():
    bearer_token_validator = BearerTokenValidator()
    with patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header", return_value={"alg": "RS256","typ": "at+jwt"}), \
         patch.object(bearer_token_validator, "_fetch_jwks", return_value=MagicMock()):  # noqa: E501
        bearer_token_validator._jwks_alg_cache["mock_jwks_uri"] = {"RS256"}
        # aud as string
        with patch("nat.authentication.credential_validator.bearer_token_validator.jwt.decode") as jwt_decode:
            claims = MockClaims({
                "iss": "https://issuer.test", "sub": "u", "aud": "api://one", "exp": int(time.time()) + 60
            })
            jwt_decode.return_value = claims
            result = await bearer_token_validator.verify(make_jwt())
            assert result.audience == ["api://one"]
        # aud as list
        with patch("nat.authentication.credential_validator.bearer_token_validator.jwt.decode") as jwt_decode:
            claims = MockClaims({
                "iss": "https://issuer.test", "sub": "u", "aud": ["a", "b"], "exp": int(time.time()) + 60
            })
            jwt_decode.return_value = claims
            result = await bearer_token_validator.verify(make_jwt())
            assert result.audience == ["a", "b"]


async def test_jwt_typ_defaults_when_missing():
    bearer_token_validator = BearerTokenValidator()
    with patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header", return_value={"alg": "RS256"}), \
         patch.object(bearer_token_validator, "_fetch_jwks", return_value=MagicMock()), \
         patch("nat.authentication.credential_validator.bearer_token_validator.jwt.decode") as jwt_decode:  # noqa: E501
        bearer_token_validator._jwks_alg_cache["mock_jwks_uri"] = {"RS256"}
        claims = MockClaims({"iss": "https://issuer.test", "sub": "u", "aud": "x", "exp": int(time.time()) + 60})
        jwt_decode.return_value = claims
        result = await bearer_token_validator.verify(make_jwt())
        assert result.token_type == "at+jwt"


# ---- Bearer prefix dispatch test
async def test_verify_strips_bearer_prefix_for_jwt():
    bearer_token_validator = BearerTokenValidator()
    token = make_jwt()
    jwks_uri = "https://issuer.test/.well-known/jwks.json"

    with patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header",
               return_value={"alg": "RS256", "typ": "at+jwt"}), \
         patch.object(bearer_token_validator, "_fetch_jwks", return_value=MagicMock()), \
         patch("nat.authentication.credential_validator.bearer_token_validator.jwt.decode",
               return_value=MockClaims({"iss": "https://issuer.test","sub": "u","aud": "x","exp": int(time.time())+60})):  # noqa: E501
        bearer_token_validator._jwks_alg_cache[jwks_uri] = {"RS256"}
        result = await bearer_token_validator.verify("Bearer " + token)
        assert isinstance(result, TokenValidationResult)


def test_init_rejects_non_https_urls():
    with pytest.raises(ValueError):
        BearerTokenValidator(jwks_uri="http://evil.example/jwks")
    with pytest.raises(ValueError):
        BearerTokenValidator(introspection_endpoint="http://evil.example/introspect")
    with pytest.raises(ValueError):
        BearerTokenValidator(discovery_url="http://evil.example/.well-known/openid-configuration")


async def test_jwt_rejects_alg_not_permitted_by_jwks_family():
    bearer_token_validator = BearerTokenValidator()

    # Use the actual JWKS URI that _resolve_jwks_uri will produce
    jwks_uri = "https://issuer.test/.well-known/jwks.json"
    with patch.object(bearer_token_validator, "_fetch_jwks", return_value=MagicMock()) as _jwks_mock, \
         patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header") as header_mock, \
         patch("nat.authentication.credential_validator.bearer_token_validator.jwt.decode", return_value=MockClaims({  # noqa: E501
             "iss": "https://issuer.test", "sub": "u", "aud": "x", "exp": int(time.time()) + 60
         })):
        header_mock.return_value = {"alg": "HS256"}
        # Set up algorithm cache to only allow RSA family algorithms, reject symmetric HS256
        bearer_token_validator._jwks_alg_cache[jwks_uri] = {"RS256", "PS256"}
        # This should be rejected by header-alg vs JWKS-family check, not by decode exception
        assert await bearer_token_validator.verify(make_jwt(alg="HS256")) is None


async def test_verify_opaque_missing_active_treated_inactive():
    bearer_token_validator = BearerTokenValidator(
        introspection_endpoint="https://as.test/introspect",
        client_id="cid",
        client_secret="csec",
    )
    with patch("nat.authentication.credential_validator.bearer_token_validator.AsyncOAuth2Client") as mock_client:
        oauth_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = oauth_client
        oauth_client.introspect_token.return_value = {"client_id": "cid", "token_type": "access_token"}  # no 'active'
        result = await bearer_token_validator.verify("opaque-token")
        assert isinstance(result, TokenValidationResult)
        assert result.active is False


async def test_verify_opaque_audience_normalization_string():
    bearer_token_validator = BearerTokenValidator(
        introspection_endpoint="https://as.test/introspect",
        client_id="cid",
        client_secret="csec",
    )
    with patch("nat.authentication.credential_validator.bearer_token_validator.AsyncOAuth2Client") as mock_client:
        oauth_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = oauth_client
        oauth_client.introspect_token.return_value = {
            "active": True,
            "client_id": "cid",
            "aud": "api://single-audience"  # String audience
        }
        result = await bearer_token_validator.verify("opaque-token")
        assert result is not None
        assert result.audience == ["api://single-audience"]  # Should normalize to list


async def test_verify_opaque_token_type_fallback():
    bearer_token_validator = BearerTokenValidator(
        introspection_endpoint="https://as.test/introspect",
        client_id="cid",
        client_secret="csec",
    )
    with patch("nat.authentication.credential_validator.bearer_token_validator.AsyncOAuth2Client") as mock_client:
        oauth_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = oauth_client
        oauth_client.introspect_token.return_value = {
            "active": True, "client_id": "cid"
            # No token_type field
        }
        result = await bearer_token_validator.verify("opaque-token")
        assert result is not None
        assert result.token_type == "opaque"  # Should default to "opaque"


async def test_verify_opaque_scope_parsing_empty_string():
    bearer_token_validator = BearerTokenValidator(
        introspection_endpoint="https://as.test/introspect",
        client_id="cid",
        client_secret="csec",
    )
    with patch("nat.authentication.credential_validator.bearer_token_validator.AsyncOAuth2Client") as mock_client:
        oauth_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = oauth_client
        oauth_client.introspect_token.return_value = {
            "active": True,
            "client_id": "cid",
            "scope": ""  # Empty string
        }
        result = await bearer_token_validator.verify("opaque-token")
        assert result is not None
        assert result.scopes is None  # Empty string should result in None


async def test_verify_opaque_scope_parsing_list():
    bearer_token_validator = BearerTokenValidator(
        introspection_endpoint="https://as.test/introspect",
        client_id="cid",
        client_secret="csec",
    )
    with patch("nat.authentication.credential_validator.bearer_token_validator.AsyncOAuth2Client") as mock_client:
        oauth_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = oauth_client
        oauth_client.introspect_token.return_value = {
            "active": True,
            "client_id": "cid",
            "scope": ["read", "write"]  # List input
        }
        result = await bearer_token_validator.verify("opaque-token")
        assert result is not None
        assert result.scopes == ["read", "write"]  # Should pass through as-is


# ---- 12) Minimal introspection response test
async def test_verify_opaque_active_true_minimal_body():
    bearer_token_validator = BearerTokenValidator(introspection_endpoint="https://as.test/introspect",
                                                  client_id="cid",
                                                  client_secret="sec")
    with patch("nat.authentication.credential_validator.bearer_token_validator.AsyncOAuth2Client") as mock_client:
        oauth_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = oauth_client
        oauth_client.introspect_token.return_value = {"active": True, "client_id": "cid"}
        result = await bearer_token_validator.verify("opaque")
        assert isinstance(result, TokenValidationResult)
        assert result.active is True and result.client_id == "cid"


async def test_jwks_uri_resolution_cache_key_consistency():
    """Test that algorithm cache keys match the actual JWKS URI resolution."""
    bearer_token_validator = BearerTokenValidator(issuer="https://issuer.test")

    # Mock JWKS response
    mock_jwks_response = {
        "keys": [{
            "kty": "RSA",
            "kid": "test-key",
            "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbPFRP_gdHPfNjxbMoVhvvJf7N2Ls_mw-wRsqiLcUyWZBaLjuDN8-mA",
            "e": "AQAB"
        }]
    }

    with patch("httpx.AsyncClient") as mock_client, \
         patch("nat.authentication.credential_validator.bearer_token_validator.pyjwt.get_unverified_header") as gethdr, \
         patch("nat.authentication.credential_validator.bearer_token_validator.jwt.decode") as mock_decode:  # noqa: E501

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

        # Mock JWT header and decode
        gethdr.return_value = {"alg": "RS256", "typ": "at+jwt"}
        claims = MockClaims({"iss": "https://issuer.test", "sub": "u", "exp": int(time.time()) + 60})
        mock_decode.return_value = claims

        # Verify a JWT - this should populate cache with the resolved JWKS URI
        _result = await bearer_token_validator.verify(make_jwt())

        # The algorithm cache should have an entry for the resolved JWKS URI
        expected_jwks_uri = "https://issuer.test/.well-known/jwks.json"
        assert expected_jwks_uri in bearer_token_validator._jwks_alg_cache
        assert "RS256" in bearer_token_validator._jwks_alg_cache[expected_jwks_uri]
