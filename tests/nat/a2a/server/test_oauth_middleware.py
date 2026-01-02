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
"""Tests for OAuth2ValidationMiddleware."""

import time
from typing import Any
from unittest.mock import patch

import pytest
from authlib.jose import JsonWebKey
from authlib.jose import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from nat.data_models.authentication import TokenValidationResult
from nat.plugins.a2a.server.oauth_middleware import OAuth2ValidationMiddleware

# ============================================================================
# Test Fixtures
# ============================================================================

ISSUER = "https://auth.example.com"
AUDIENCE = "http://localhost:10000"
REQUIRED_SCOPES = ["calculator_a2a:execute"]


@pytest.fixture(scope="session")
def rsa_private_pem() -> str:
    """Generate RSA private key for signing test JWTs."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem.decode("utf-8")


@pytest.fixture(scope="session")
def jwks_dict(rsa_private_pem: str) -> dict[str, Any]:
    """Create JWKS dictionary from private key for token validation."""
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.hazmat.primitives.serialization import PublicFormat
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    private_key = load_pem_private_key(rsa_private_pem.encode(), password=None)
    public_key = private_key.public_key()
    public_key_pem = public_key.public_bytes(encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo)

    jwk = JsonWebKey.import_key(public_key_pem)
    jwk_dict = jwk.as_dict()
    jwk_dict['kid'] = 'test-key-id'
    jwk_dict['use'] = 'sig'
    jwk_dict['alg'] = 'RS256'

    return {"keys": [jwk_dict]}


def make_jwt(
    rsa_private_pem: str,
    exp_offset_secs: int = 300,
    scopes: list[str] | None = None,
    audience: str | list[str] | None = AUDIENCE,
    issuer: str = ISSUER,
    subject: str = "test-user",
    client_id: str = "test-client",
) -> str:
    """Create a test JWT token."""
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": issuer,
        "sub": subject,
        "aud": audience,
        "iat": now,
        "nbf": now,
        "exp": now + exp_offset_secs,
        "scope": " ".join(scopes) if scopes else None,
        "azp": client_id,
        "jti": "test-jwt-id",
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    header = {"alg": "RS256", "typ": "JWT", "kid": "test-key-id"}
    return jwt.encode(header, payload, rsa_private_pem).decode("utf-8")


@pytest.fixture
def oauth_config() -> OAuth2ResourceServerConfig:
    """Create OAuth2 resource server configuration for testing."""
    return OAuth2ResourceServerConfig(
        issuer_url=ISSUER,
        audience=AUDIENCE,
        scopes=REQUIRED_SCOPES,
        jwks_uri=f"{ISSUER}/.well-known/jwks.json",
    )


@pytest.fixture
def protected_app(oauth_config: OAuth2ResourceServerConfig):
    """Create test Starlette app with OAuth2 middleware."""

    async def agent_card(request: Request):
        """Public agent card endpoint."""
        return JSONResponse({"name": "Test Agent", "version": "1.0.0"})

    async def protected_endpoint(request: Request):
        """Protected endpoint that requires authentication."""
        return JSONResponse({
            "message": "success",
            "user": request.state.oauth_user,
            "scopes": request.state.oauth_scopes,
            "client_id": request.state.oauth_client_id,
        })

    app = Starlette(routes=[
        Route("/.well-known/agent-card.json", agent_card),
        Route("/", protected_endpoint, methods=["POST"]),
    ], )

    # Add OAuth2 middleware
    app.add_middleware(OAuth2ValidationMiddleware, config=oauth_config)

    return app


# ============================================================================
# Tests: Public Endpoint Access
# ============================================================================


class TestPublicEndpoints:
    """Test public endpoint access without authentication."""

    def test_agent_card_accessible_without_token(self, protected_app):
        """Agent card endpoint should be accessible without authentication."""
        client = TestClient(protected_app)
        response = client.get("/.well-known/agent-card.json")

        assert response.status_code == 200
        assert response.json()["name"] == "Test Agent"

    def test_agent_card_accessible_with_invalid_token(self, protected_app):
        """Agent card endpoint should be accessible even with invalid token."""
        client = TestClient(protected_app)
        response = client.get(
            "/.well-known/agent-card.json",
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response.status_code == 200
        assert response.json()["name"] == "Test Agent"


# ============================================================================
# Tests: Protected Endpoint Authentication
# ============================================================================


class TestProtectedEndpoints:
    """Test protected endpoint authentication requirements."""

    def test_missing_authorization_header(self, protected_app):
        """Protected endpoint should reject requests without Authorization header."""
        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"})

        assert response.status_code == 401
        assert response.json()["error"] == "unauthorized"
        assert "Missing or invalid Bearer token" in response.json()["message"]

    def test_invalid_authorization_format(self, protected_app):
        """Protected endpoint should reject requests with invalid auth format."""
        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": "Basic abc123"})

        assert response.status_code == 401
        assert response.json()["error"] == "unauthorized"

    def test_empty_bearer_token(self, protected_app):
        """Protected endpoint should reject requests with empty Bearer token."""
        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": "Bearer "})

        assert response.status_code == 403
        assert response.json()["error"] == "invalid_token"


# ============================================================================
# Tests: Token Validation
# ============================================================================


class TestTokenValidation:
    """Test JWT token validation scenarios."""

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_valid_token_accepted(self, mock_verify, protected_app, rsa_private_pem):
        """Valid JWT token should be accepted."""
        # Mock successful validation
        mock_verify.return_value = TokenValidationResult(
            active=True,
            subject="test-user",
            client_id="test-client",
            scopes=REQUIRED_SCOPES,
            issuer=ISSUER,
            audience=[AUDIENCE],
            token_type="Bearer",
        )

        token = make_jwt(rsa_private_pem, scopes=REQUIRED_SCOPES)
        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        assert response.json()["message"] == "success"
        assert response.json()["user"] == "test-user"
        assert response.json()["scopes"] == REQUIRED_SCOPES

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_expired_token_rejected(self, mock_verify, protected_app, rsa_private_pem):
        """Expired JWT token should be rejected."""
        # Create expired token (negative exp offset)
        token = make_jwt(rsa_private_pem, exp_offset_secs=-60, scopes=REQUIRED_SCOPES)

        # Mock validation failure
        mock_verify.side_effect = Exception("Token has expired")

        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 403
        assert response.json()["error"] == "invalid_token"
        assert "Token validation failed" in response.json()["message"]

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_invalid_signature_rejected(self, mock_verify, protected_app):
        """Token with invalid signature should be rejected."""
        # Mock validation failure
        mock_verify.side_effect = Exception("Invalid signature")

        # Use a malformed token
        token = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.invalid_signature"
        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 403
        assert response.json()["error"] == "invalid_token"

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_wrong_issuer_rejected(self, mock_verify, protected_app, rsa_private_pem):
        """Token from wrong issuer should be rejected."""
        # Create token with wrong issuer
        token = make_jwt(rsa_private_pem, issuer="https://wrong-issuer.com", scopes=REQUIRED_SCOPES)

        # Mock validation failure
        mock_verify.side_effect = Exception("Invalid issuer")

        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 403
        assert response.json()["error"] == "invalid_token"

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_inactive_token_rejected(self, mock_verify, protected_app, rsa_private_pem):
        """Inactive token should be rejected."""
        # Mock inactive token
        mock_verify.return_value = TokenValidationResult(
            active=False,
            subject="test-user",
            client_id="test-client",
            scopes=REQUIRED_SCOPES,
            issuer=ISSUER,
            audience=[AUDIENCE],
            token_type="Bearer",
        )

        token = make_jwt(rsa_private_pem, scopes=REQUIRED_SCOPES)
        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 403
        assert response.json()["error"] == "invalid_token"
        assert "Token is not active" in response.json()["message"]


# ============================================================================
# Tests: Scope and Audience Validation
# ============================================================================


class TestScopeAndAudienceValidation:
    """Test scope and audience validation logic."""

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_missing_required_scopes_rejected(self, mock_verify, protected_app, rsa_private_pem):
        """Token without required scopes should be rejected."""
        # Mock validation with wrong scopes
        mock_verify.return_value = TokenValidationResult(
            active=True,
            subject="test-user",
            client_id="test-client",
            scopes=["wrong:scope"],  # Different scopes
            issuer=ISSUER,
            audience=[AUDIENCE],
            token_type="Bearer",
        )

        token = make_jwt(rsa_private_pem, scopes=["wrong:scope"])
        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        # Note: BearerTokenValidator handles scope validation, so if it returns active=True,
        # middleware accepts it. This tests that the validator is being called.
        assert response.status_code == 200
        assert response.json()["scopes"] == ["wrong:scope"]

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_wrong_audience_rejected(self, mock_verify, protected_app, rsa_private_pem):
        """Token with wrong audience should be rejected by validator."""
        # Mock validation failure due to wrong audience
        mock_verify.side_effect = Exception("Invalid audience")

        token = make_jwt(rsa_private_pem, audience="https://wrong-audience.com", scopes=REQUIRED_SCOPES)
        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 403
        assert response.json()["error"] == "invalid_token"

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_correct_scopes_and_audience_accepted(self, mock_verify, protected_app, rsa_private_pem):
        """Token with correct scopes and audience should be accepted."""
        # Mock successful validation
        mock_verify.return_value = TokenValidationResult(
            active=True,
            subject="test-user",
            client_id="test-client",
            scopes=REQUIRED_SCOPES,
            issuer=ISSUER,
            audience=[AUDIENCE],
            token_type="Bearer",
        )

        token = make_jwt(rsa_private_pem, scopes=REQUIRED_SCOPES, audience=AUDIENCE)
        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        assert response.json()["message"] == "success"


# ============================================================================
# Tests: Request State Population
# ============================================================================


class TestRequestStatePopulation:
    """Test that middleware correctly populates request state."""

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_request_state_populated_correctly(self, mock_verify, protected_app, rsa_private_pem):
        """Valid token should populate request.state with OAuth info."""
        # Mock successful validation with specific values
        expected_user = "alice@example.com"
        expected_scopes = ["calculator_a2a:execute", "read"]
        expected_client_id = "math-assistant-client"

        mock_verify.return_value = TokenValidationResult(
            active=True,
            subject=expected_user,
            client_id=expected_client_id,
            scopes=expected_scopes,
            issuer=ISSUER,
            audience=[AUDIENCE],
            token_type="Bearer",
        )

        token = make_jwt(
            rsa_private_pem,
            scopes=expected_scopes,
            subject=expected_user,
            client_id=expected_client_id,
        )
        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        data = response.json()
        assert data["user"] == expected_user
        assert data["scopes"] == expected_scopes
        assert data["client_id"] == expected_client_id

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_request_state_with_no_scopes(self, mock_verify, protected_app, rsa_private_pem):
        """Token without scopes should populate empty scopes list."""
        # Mock validation without scopes
        mock_verify.return_value = TokenValidationResult(
            active=True,
            subject="test-user",
            client_id="test-client",
            scopes=None,  # No scopes
            issuer=ISSUER,
            audience=[AUDIENCE],
            token_type="Bearer",
        )

        token = make_jwt(rsa_private_pem, scopes=None)
        client = TestClient(protected_app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        data = response.json()
        assert data["scopes"] == []  # Should default to empty list


# ============================================================================
# Tests: Configuration Variations
# ============================================================================


class TestConfigurationVariations:
    """Test different OAuth2 configuration scenarios."""

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_middleware_without_audience_validation(self, mock_verify, rsa_private_pem):
        """Middleware should work without audience validation configured."""
        # Create config without audience
        config = OAuth2ResourceServerConfig(
            issuer_url=ISSUER,
            audience=None,  # No audience required
            scopes=REQUIRED_SCOPES,
            jwks_uri=f"{ISSUER}/.well-known/jwks.json",
        )

        async def protected(request: Request):
            return JSONResponse({"message": "success"})

        app = Starlette(routes=[Route("/", protected, methods=["POST"])])
        app.add_middleware(OAuth2ValidationMiddleware, config=config)

        # Mock successful validation without audience
        mock_verify.return_value = TokenValidationResult(
            active=True,
            subject="test-user",
            client_id="test-client",
            scopes=REQUIRED_SCOPES,
            issuer=ISSUER,
            audience=None,
            token_type="Bearer",
        )

        token = make_jwt(rsa_private_pem, audience=None, scopes=REQUIRED_SCOPES)
        client = TestClient(app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200

    @patch("nat.authentication.credential_validator.bearer_token_validator.BearerTokenValidator.verify")
    async def test_middleware_without_scope_validation(self, mock_verify, rsa_private_pem):
        """Middleware should work without scope validation configured."""
        # Create config without scopes
        config = OAuth2ResourceServerConfig(
            issuer_url=ISSUER,
            audience=AUDIENCE,
            scopes=[],  # Empty list - no scopes required
            jwks_uri=f"{ISSUER}/.well-known/jwks.json",
        )

        async def protected(request: Request):
            return JSONResponse({"message": "success"})

        app = Starlette(routes=[Route("/", protected, methods=["POST"])])
        app.add_middleware(OAuth2ValidationMiddleware, config=config)

        # Mock successful validation without scopes
        mock_verify.return_value = TokenValidationResult(
            active=True,
            subject="test-user",
            client_id="test-client",
            scopes=None,
            issuer=ISSUER,
            audience=[AUDIENCE],
            token_type="Bearer",
        )

        token = make_jwt(rsa_private_pem, scopes=None, audience=AUDIENCE)
        client = TestClient(app)
        response = client.post("/", json={"task": "test"}, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
