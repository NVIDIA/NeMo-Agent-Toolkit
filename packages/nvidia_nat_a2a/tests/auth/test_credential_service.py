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
"""Tests for A2ACredentialService."""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from a2a.client import ClientCallContext
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import APIKeySecurityScheme
from a2a.types import AuthorizationCodeOAuthFlow
from a2a.types import HTTPAuthSecurityScheme
from a2a.types import In
from a2a.types import OAuth2SecurityScheme
from a2a.types import OAuthFlows
from a2a.types import OpenIdConnectSecurityScheme
from a2a.types import SecurityScheme
from pydantic import SecretStr

from nat.authentication.interfaces import AuthProviderBase
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BasicAuthCred
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.authentication import CredentialKind
from nat.data_models.authentication import HeaderCred
from nat.plugins.a2a.auth.credential_service import A2ACredentialService

# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


class MockAuthProvider(AuthProviderBase):
    """Generic mock auth provider for testing."""

    def __init__(self, auth_result: AuthResult | None = None):
        super().__init__(Mock())  # type: ignore[arg-type]
        self.auth_result = auth_result
        self.authenticate_called_with = []

    async def authenticate(self, user_id: str | None = None, **kwargs) -> AuthResult:
        self.authenticate_called_with.append(user_id)
        if self.auth_result is None:
            raise ValueError("Authentication failed")
        return self.auth_result


@pytest.fixture
def mock_auth_provider():
    """Fixture factory to create a mock auth provider with specific name for validation."""

    def _create(provider_name: str, auth_result: AuthResult | None = None):

        def mock_init(self, result):
            AuthProviderBase.__init__(self, Mock())  # type: ignore[arg-type]
            self.auth_result = result
            self.authenticate_called_with = []

        async def mock_authenticate(self, user_id=None, **kwargs):
            self.authenticate_called_with.append(user_id)
            if self.auth_result is None:
                raise ValueError("Authentication failed")
            return self.auth_result

        # Dynamically create a class with the desired name
        cls = type(provider_name, (AuthProviderBase, ), {
            '__init__': mock_init,
            'authenticate': mock_authenticate,
        })
        return cls(auth_result)  # type: ignore[call-arg]

    return _create


@pytest.fixture
def oauth2_scheme():
    """OAuth2 security scheme fixture."""
    return SecurityScheme(root=OAuth2SecurityScheme(
        type="oauth2",
        flows=OAuthFlows(authorization_code=AuthorizationCodeOAuthFlow(
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            scopes={"read": "Read access"},
        )),
    ))


@pytest.fixture
def oidc_scheme():
    """OpenID Connect security scheme fixture."""
    return SecurityScheme(root=OpenIdConnectSecurityScheme(
        type="openIdConnect",
        open_id_connect_url="https://auth.example.com/.well-known/openid-configuration",
    ))


@pytest.fixture
def http_bearer_scheme():
    """HTTP Bearer security scheme fixture."""
    return SecurityScheme(root=HTTPAuthSecurityScheme(type="http", scheme="bearer"))


@pytest.fixture
def api_key_scheme():
    """API Key security scheme fixture."""
    return SecurityScheme(root=APIKeySecurityScheme(
        type="apiKey",
        name="X-API-Key",
        in_=In.header,
    ))


@pytest.fixture
def http_basic_scheme():
    """HTTP Basic security scheme fixture."""
    return SecurityScheme(root=HTTPAuthSecurityScheme(type="http", scheme="basic"))


@pytest.fixture
def sample_agent_card():
    """Fixture factory to create a sample AgentCard with security schemes."""

    def _create(security_schemes: dict[str, SecurityScheme]) -> AgentCard:
        return AgentCard(
            name="Test Agent",
            description="Test agent for authentication",
            version="1.0.0",
            url="https://test-agent.example.com",
            default_input_modes=["text"],
            default_output_modes=["text"],
            skills=[],
            capabilities=AgentCapabilities(),
            security_schemes=security_schemes,
            security=[{
                scheme_name: []
            } for scheme_name in security_schemes.keys()],
        )

    return _create


# ============================================================================
# Credential Mapping Tests
# ============================================================================


@pytest.mark.parametrize("scheme_name,scheme_fixture,token_value",
                         [
                             ("oauth2", "oauth2_scheme", "test-access-token"),
                             ("oidc", "oidc_scheme", "test-id-token"),
                             ("http_bearer", "http_bearer_scheme", "test-bearer-token"),
                         ])
async def test_bearer_token_mapping(
    scheme_name,
    scheme_fixture,
    token_value,
    request,
    mock_auth_provider,
    sample_agent_card,
    mock_user_context,
):
    """Test BearerTokenCred maps to various bearer-compatible schemes."""
    scheme = request.getfixturevalue(scheme_fixture)
    auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr(token_value))])
    provider = mock_auth_provider("MockOAuth2Provider", auth_result)

    card = sample_agent_card({scheme_name: scheme})
    service = A2ACredentialService(
        auth_provider=provider,
        agent_card=card,
    )

    # Mock the Context to return a user_id
    with patch('nat.plugins.a2a.auth.credential_service.Context') as mock_context:
        mock_context.get.return_value = mock_user_context
        credential = await service.get_credentials(scheme_name, None)

    assert credential == token_value
    assert provider.authenticate_called_with == ["test-user"]


async def test_header_credential_with_api_key_scheme(api_key_scheme, mock_auth_provider, sample_agent_card):
    """Test HeaderCred maps to APIKeySecurityScheme in header."""
    auth_result = AuthResult(
        credentials=[HeaderCred(kind=CredentialKind.HEADER, name="X-API-Key", value=SecretStr("test-api-key"))])
    provider = mock_auth_provider("MockAPIKeyProvider", auth_result)

    card = sample_agent_card({"api_key": api_key_scheme})
    service = A2ACredentialService(auth_provider=provider, agent_card=card)

    # Mock the Context to return a user_id
    with patch('nat.plugins.a2a.auth.credential_service.Context') as mock_context:
        mock_context.get.return_value.user_id = "test-user"
        credential = await service.get_credentials("api_key", None)

    assert credential == "test-api-key"


# ============================================================================
# Token Lifecycle Tests
# ============================================================================


async def test_token_expiration_triggers_reauthentication(oauth2_scheme, mock_auth_provider, sample_agent_card):
    """Test expired cached token triggers re-authentication on next call."""
    expired_result = AuthResult(
        credentials=[BearerTokenCred(token=SecretStr("expired-token"))],
        token_expires_at=datetime.now(UTC) - timedelta(hours=1),
    )
    fresh_result = AuthResult(
        credentials=[BearerTokenCred(token=SecretStr("fresh-token"))],
        token_expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    call_count = [0]

    async def authenticate_with_states(user_id=None, **kwargs):
        call_count[0] += 1
        return expired_result if call_count[0] == 1 else fresh_result

    provider = mock_auth_provider("MockOAuth2Provider", expired_result)
    provider.authenticate = authenticate_with_states

    card = sample_agent_card({"oauth": oauth2_scheme})
    service = A2ACredentialService(auth_provider=provider, agent_card=card)

    # Mock the Context to return a user_id
    with patch('nat.plugins.a2a.auth.credential_service.Context') as mock_context:
        mock_context.get.return_value.user_id = "test-user"
        # First call: gets and caches expired token (provider's responsibility to return valid tokens)
        credential1 = await service.get_credentials("oauth", None)
        assert credential1 == "expired-token"
        assert call_count[0] == 1

        # Second call: detects cache is expired, re-authenticates and gets fresh token
        credential2 = await service.get_credentials("oauth", None)
        assert credential2 == "fresh-token"
        assert call_count[0] == 2


async def test_credential_caching(oauth2_scheme, mock_auth_provider, sample_agent_card):
    """Test credentials are cached between calls."""
    auth_result = AuthResult(
        credentials=[BearerTokenCred(token=SecretStr("cached-token"))],
        token_expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    provider = mock_auth_provider("MockOAuth2Provider", auth_result)

    card = sample_agent_card({"oauth": oauth2_scheme})
    service = A2ACredentialService(auth_provider=provider, agent_card=card)

    # Mock the Context to return a user_id
    with patch('nat.plugins.a2a.auth.credential_service.Context') as mock_context:
        mock_context.get.return_value.user_id = "test-user"
        credential1 = await service.get_credentials("oauth", None)
        credential2 = await service.get_credentials("oauth", None)

    assert credential1 == credential2 == "cached-token"
    assert len(provider.authenticate_called_with) == 1  # Only called once


# ============================================================================
# Context and User ID Tests
# ============================================================================


async def test_user_id_from_context(oauth2_scheme, mock_auth_provider, sample_agent_card, mock_user_context):
    """Test user_id is extracted from NAT Context."""
    auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("test-token"))])
    provider = mock_auth_provider("MockOAuth2Provider", auth_result)

    card = sample_agent_card({"oauth": oauth2_scheme})
    service = A2ACredentialService(
        auth_provider=provider,
        agent_card=card,
    )

    # Mock the Context to return a user_id
    # Override the user_id for this specific test
    mock_user_context.user_id = "context-user"

    with patch('nat.plugins.a2a.auth.credential_service.Context') as mock_context:
        mock_context.get.return_value = mock_user_context
        # Note: user_id is sourced from mocked Context.get().user_id, not from the context parameter
        context = ClientCallContext(state={"sessionId": "context-user"})
        credential = await service.get_credentials("oauth", context)

    assert credential == "test-token"
    assert provider.authenticate_called_with == ["context-user"]


# ============================================================================
# Error Handling Tests
# ============================================================================


async def test_missing_security_scheme_returns_none(mock_auth_provider, sample_agent_card):
    """Test returns None when security scheme is not defined."""
    auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("test-token"))])
    provider = mock_auth_provider("MockOAuth2Provider", auth_result)

    card = sample_agent_card({})
    service = A2ACredentialService(auth_provider=provider, agent_card=card)

    # Mock the Context to return a user_id
    with patch('nat.plugins.a2a.auth.credential_service.Context') as mock_context:
        mock_context.get.return_value.user_id = "test-user"
        credential = await service.get_credentials("nonexistent", None)

    assert credential is None


async def test_authentication_failure_returns_none(oauth2_scheme, mock_auth_provider, sample_agent_card):
    """Test returns None when authentication fails."""
    provider = mock_auth_provider("MockOAuth2Provider", None)

    card = sample_agent_card({"oauth": oauth2_scheme})
    service = A2ACredentialService(auth_provider=provider, agent_card=card)

    # Mock the Context to return a user_id
    with patch('nat.plugins.a2a.auth.credential_service.Context') as mock_context:
        mock_context.get.return_value.user_id = "test-user"
        credential = await service.get_credentials("oauth", None)

    assert credential is None


# ============================================================================
# Provider Compatibility Validation Tests
# ============================================================================


@pytest.mark.parametrize(
    "provider_name,scheme_fixture,scheme_name,should_pass",
    [
        ("MockOAuth2Provider", "oauth2_scheme", "oauth", True),
        ("MockOAuth2Provider", "oidc_scheme", "oidc", True),
        ("MockOAuth2Provider", "http_bearer_scheme", "bearer", True),
        ("MockAPIKeyProvider", "api_key_scheme", "apiKey", True),
        ("MockHTTPBasicProvider", "http_basic_scheme", "basic", True),
        ("MockHTTPBasicProvider", "oauth2_scheme", "oauth", False),  # Incompatible
    ])
def test_provider_validation(provider_name,
                             scheme_fixture,
                             scheme_name,
                             should_pass,
                             request,
                             mock_auth_provider,
                             sample_agent_card):
    """Test provider-scheme compatibility validation."""
    scheme = request.getfixturevalue(scheme_fixture)
    auth_result = AuthResult(credentials=[
        BearerTokenCred(token=SecretStr("token")) if "OAuth2" in provider_name or "Bearer" in scheme_name else
        BasicAuthCred(username=SecretStr("user"), password=SecretStr("pass")) if "Basic" in
        provider_name else HeaderCred(name="X-API-Key", value=SecretStr("key"))
    ])

    provider = mock_auth_provider(provider_name, auth_result)
    card = sample_agent_card({scheme_name: scheme})

    if should_pass:
        service = A2ACredentialService(
            auth_provider=provider,
            agent_card=card,
        )
        assert service is not None
    else:
        with pytest.raises(ValueError, match="not compatible with agent's security requirements"):
            A2ACredentialService(
                auth_provider=provider,
                agent_card=card,
            )


@pytest.mark.parametrize(
    "agent_card_config",
    [
        {},  # No security schemes
        None,  # None agent card
    ])
def test_validation_skipped_when_no_schemes(agent_card_config, mock_auth_provider, sample_agent_card):
    """Test validation is skipped when agent has no security schemes."""
    auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("token"))])
    provider = mock_auth_provider("MockOAuth2Provider", auth_result)

    card = sample_agent_card(agent_card_config) if agent_card_config is not None else None
    service = A2ACredentialService(
        auth_provider=provider,
        agent_card=card,
    )

    assert service is not None
