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

from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig

# ========== CLIENT_SERVER_URL VALIDATION ==========


@pytest.mark.parametrize("client_server_url",
                         [
                             "https://api.example.com",
                             "http://localhost:8000",
                             "http://127.0.0.1:8080",
                             "https://localhost:8443",
                             "https://service.internal.domain",
                             "http://localhost",
                         ])
async def test_client_server_url_valid(client_server_url):
    """Test valid client_server_url configurations."""
    # Should not raise AuthCodeGrantConfigClientServerUrlFieldError
    AuthCodeGrantConfig(client_server_url=client_server_url,
                        authorization_url="https://auth.example.com/oauth/authorize",
                        authorization_token_url="https://auth.example.com/oauth/token",
                        consent_prompt_key="oauth_consent_key",
                        client_secret="super_secure_client_secret_123456",
                        client_id="my_client_app_123",
                        audience="api.example.com",
                        scope=["read", "write"])


@pytest.mark.parametrize(
    "invalid_client_server_url",
    [
        "",  # Empty URL (value_missing)
        "ftp://example.com",  # Invalid scheme (invalid_scheme)
        "//example.com",  # No scheme (invalid_scheme)
        "https://",  # No hostname (invalid_url)
        "mailto:test@example.com",  # Invalid scheme (invalid_scheme)
    ])
async def test_client_server_url_invalid(invalid_client_server_url):
    """Test invalid client_server_url configurations."""
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientServerUrlFieldError

    # Should raise AuthCodeGrantConfigClientServerUrlFieldError
    with pytest.raises(AuthCodeGrantConfigClientServerUrlFieldError):
        AuthCodeGrantConfig(client_server_url=invalid_client_server_url,
                            authorization_url="https://auth.example.com/oauth/authorize",
                            authorization_token_url="https://auth.example.com/oauth/token",
                            consent_prompt_key="oauth_consent_key",
                            client_secret="super_secure_client_secret_123456",
                            client_id="my_client_app_123",
                            audience="api.example.com",
                            scope=["read", "write"])


# ========== OAUTH URLS HTTPS VALIDATION ==========
@pytest.mark.parametrize(
    "valid_authorization_url",
    [
        "https://auth.example.com/oauth/authorize",  # Standard OAuth endpoint
        "https://login.microsoftonline.com/tenant/oauth2/v2.0/authorize",  # Microsoft Azure AD
        "https://accounts.google.com/o/oauth2/auth",  # Google OAuth
        "https://github.com/login/oauth/authorize",  # GitHub OAuth
        "https://api.internal.company.com/auth/oauth/authorize",  # Internal company API
    ])
async def test_authorization_url_valid(valid_authorization_url):
    """Test valid authorization_url configurations following RFC 6819 Section 4.2.1."""

    # Should not raise AuthCodeGrantConfigAuthorizationUrlFieldError
    AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                        authorization_url=valid_authorization_url,
                        authorization_token_url="https://auth.example.com/oauth/token",
                        consent_prompt_key="oauth_consent_key",
                        client_secret="super_secure_client_secret_123456",
                        client_id="my_client_app_123",
                        audience="api.example.com",
                        scope=["read", "write"])


@pytest.mark.parametrize(
    "invalid_authorization_url",
    [
        "",  # Empty URL (value_missing)
        "https://",  # No hostname (invalid_url)
        "https://auth.example.com",  # No path (path_missing)
        "https://auth.example.com/",  # Root path only (path_missing)
        "ftp://auth.example.com/oauth/authorize",  # Invalid scheme (https_required)
    ])
async def test_authorization_url_invalid(invalid_authorization_url):
    """Test invalid authorization_url configurations."""
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigAuthorizationUrlFieldError

    # Should raise AuthCodeGrantConfigAuthorizationUrlFieldError
    with pytest.raises(AuthCodeGrantConfigAuthorizationUrlFieldError):
        AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                            authorization_url=invalid_authorization_url,
                            authorization_token_url="https://auth.example.com/oauth/token",
                            consent_prompt_key="oauth_consent_key",
                            client_secret="super_secure_client_secret_123456",
                            client_id="my_client_app_123",
                            audience="api.example.com",
                            scope=["read", "write"])


@pytest.mark.parametrize(
    "valid_authorization_token_url",
    [
        "https://auth.example.com/oauth/token",  # Standard OAuth token endpoint
        "https://login.microsoftonline.com/tenant/oauth2/v2.0/token",  # Microsoft Azure AD
        "https://oauth2.googleapis.com/token",  # Google OAuth
        "https://github.com/login/oauth/access_token",  # GitHub OAuth
        "https://api.internal.company.com/auth/oauth/token",  # Internal company API
    ])
async def test_authorization_token_url_valid(valid_authorization_token_url):
    """Test valid authorization_token_url configurations following RFC 6819 Section 4.2.1."""
    # Should not raise AuthCodeGrantConfigAuthorizationUrlFieldError
    AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                        authorization_url="https://auth.example.com/oauth/authorize",
                        authorization_token_url=valid_authorization_token_url,
                        consent_prompt_key="oauth_consent_key",
                        client_secret="super_secure_client_secret_123456",
                        client_id="my_client_app_123",
                        audience="api.example.com",
                        scope=["read", "write"])


@pytest.mark.parametrize(
    "invalid_authorization_token_url",
    [
        "",  # Empty URL (value_missing)
        "http://auth.example.com/oauth/token",  # HTTP instead of HTTPS (https_required)
        "https://",  # No hostname (invalid_url)
        "https://auth.example.com",  # No path (path_missing)
        "https://auth.example.com/",  # Root path only (path_missing)
        "ftp://auth.example.com/oauth/token",  # Invalid scheme (https_required)
    ])
async def test_authorization_token_url_invalid(invalid_authorization_token_url):
    """Test invalid authorization_token_url configurations."""
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigAuthorizationUrlFieldError

    # Should raise AuthCodeGrantConfigAuthorizationUrlFieldError
    with pytest.raises(AuthCodeGrantConfigAuthorizationUrlFieldError):
        AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                            authorization_url="https://auth.example.com/oauth/authorize",
                            authorization_token_url=invalid_authorization_token_url,
                            consent_prompt_key="oauth_consent_key",
                            client_secret="super_secure_client_secret_123456",
                            client_id="my_client_app_123",
                            audience="api.example.com",
                            scope=["read", "write"])


# ========== CONSENT_PROMPT_KEY VALIDATION ==========
@pytest.mark.parametrize(
    "valid_consent_prompt_key",
    [
        "oauth_consent_key",  # Standard key (18 chars)
        "12345678",  # Minimum length (8 chars)
        "my_secure_prompt_key_123",  # Longer key with numbers
        "ConsentKey123!@#",  # Mixed case with special chars
        "a" * 32,  # Very long key (32 chars)
        "user-auth-prompt-key",  # Dashes allowed
    ])
async def test_consent_prompt_key_valid(valid_consent_prompt_key):
    """Test valid consent_prompt_key configurations."""

    AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                        authorization_url="https://auth.example.com/oauth/authorize",
                        authorization_token_url="https://auth.example.com/oauth/token",
                        consent_prompt_key=valid_consent_prompt_key,
                        client_secret="super_secure_client_secret_123456",
                        client_id="my_client_app_123",
                        audience="api.example.com",
                        scope=["read", "write"])


@pytest.mark.parametrize(
    "invalid_consent_prompt_key",
    [
        "",  # Empty key (value_missing)
        "1234567",  # Too short - 7 chars (value_too_short)
        "  oauth_key  ",  # Leading/trailing whitespace (whitespace_found)
        " key",  # Leading whitespace (whitespace_found)
        "key ",  # Trailing whitespace (whitespace_found)
        "short",  # Too short - 5 chars (value_too_short)
    ])
async def test_consent_prompt_key_invalid(invalid_consent_prompt_key):
    """Test invalid consent_prompt_key configurations."""
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigConsentPromptKeyFieldError

    # Should raise AuthCodeGrantConfigConsentPromptKeyFieldError
    with pytest.raises(AuthCodeGrantConfigConsentPromptKeyFieldError):
        AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                            authorization_url="https://auth.example.com/oauth/authorize",
                            authorization_token_url="https://auth.example.com/oauth/token",
                            consent_prompt_key=invalid_consent_prompt_key,
                            client_secret="super_secure_client_secret_123456",
                            client_id="my_client_app_123",
                            audience="api.example.com",
                            scope=["read", "write"])


# ========== CLIENT_SECRET VALIDATION ==========


@pytest.mark.parametrize(
    "valid_secret",
    [
        "super_secure_client_secret_123456",  # Default valid secret (34 chars)
        "1234567890123456",  # Exactly 16 characters
        "a" * 64  # Very long secret (64 chars)
    ])
async def test_client_secret_valid(valid_secret):
    """
    Test valid client_secret configurations.
    """

    AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                        authorization_url="https://auth.example.com/oauth/authorize",
                        authorization_token_url="https://auth.example.com/oauth/token",
                        consent_prompt_key="oauth_consent_key",
                        client_secret=valid_secret,
                        client_id="my_client_app_123",
                        audience="api.example.com",
                        scope=["read", "write"])


@pytest.mark.parametrize(
    "invalid_secret",
    [
        "",  # Empty secret
        "short"  # Too short (less than 16 chars)
    ])
async def test_client_secret_invalid(invalid_secret):
    """
    Test valid Auth Code Grant Config client secret.
    """

    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientSecretFieldError

    # Should raise AuthCodeGrantConfigClientSecretFieldError
    with pytest.raises(AuthCodeGrantConfigClientSecretFieldError):
        AuthCodeGrantConfig(client_server_url="https://test.com",
                            authorization_url="https://test.com/auth",
                            authorization_token_url="https://test.com/token",
                            consent_prompt_key="test_key_secure",
                            client_secret=invalid_secret,
                            client_id="test_client",
                            audience="test_audience",
                            scope=["test_scope"])


# ========== CLIENT_ID VALIDATION ==========
@pytest.mark.parametrize("valid_client_id", ["my_client_app_123", "abc", "my_application_client_id_123"])
async def test_client_id_valid(valid_client_id):
    """Test valid client_id configurations."""

    # Should not raise AuthCodeGrantConfigClientIDFieldError
    AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                        authorization_url="https://auth.example.com/oauth/authorize",
                        authorization_token_url="https://auth.example.com/oauth/token",
                        consent_prompt_key="oauth_consent_key",
                        client_secret="super_secure_client_secret_123456",
                        client_id=valid_client_id,
                        audience="api.example.com",
                        scope=["read", "write"])


@pytest.mark.parametrize(
    "invalid_client_id",
    [
        "",  # Empty client ID
        "  client_with_spaces  "  # Client ID with whitespace
    ])
async def test_client_id_invalid(invalid_client_id):
    """Test invalid client_id configurations."""
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientIDFieldError

    # Should raise AuthCodeGrantConfigClientIDFieldError
    with pytest.raises(AuthCodeGrantConfigClientIDFieldError):
        AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                            authorization_url="https://auth.example.com/oauth/authorize",
                            authorization_token_url="https://auth.example.com/oauth/token",
                            consent_prompt_key="oauth_consent_key",
                            client_secret="super_secure_client_secret_123456",
                            client_id=invalid_client_id,
                            audience="api.example.com",
                            scope=["read", "write"])


# ========== SCOPE VALIDATION ==========
@pytest.mark.parametrize(
    "valid_scope",
    [
        ["read", "write"],  # Default valid scopes
        ["read"],  # Single scope
        ["read", "write", "update", "delete"],  # Multiple limited scopes
        [f"scope_{i}" for i in range(10)]  # Maximum allowed scopes (10)
    ])
async def test_scope_valid(valid_scope):
    """
    Test valid scope configurations.

    RFC 6819 Section 4.3.1 - Access Token Disclosure to Unintended Recipients
    Scopes should follow principle of least privilege.
    """

    # Should not raise AuthCodeGrantConfigScopeFieldError
    AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                        authorization_url="https://auth.example.com/oauth/authorize",
                        authorization_token_url="https://auth.example.com/oauth/token",
                        consent_prompt_key="oauth_consent_key",
                        client_secret="super_secure_client_secret_123456",
                        client_id="my_client_app_123",
                        audience="api.example.com",
                        scope=valid_scope)


@pytest.mark.parametrize(
    "invalid_scope",
    [
        [],  # Empty scope list
        ["read", "write", "read"],  # Duplicate scopes
        ["read", "", "write"],  # Empty/whitespace scopes
        ["*", "all", "root", "admin", "superuser"]  # Dangerous scopes
    ])
async def test_scope_invalid(invalid_scope):
    """
    Test invalid scope configurations.

    RFC 6819 Section 4.3.1 - Access Token Disclosure to Unintended Recipients
    Scopes should follow principle of least privilege.
    """
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigScopeFieldError

    # Should raise AuthCodeGrantConfigScopeFieldError
    with pytest.raises(AuthCodeGrantConfigScopeFieldError):
        AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                            authorization_url="https://auth.example.com/oauth/authorize",
                            authorization_token_url="https://auth.example.com/oauth/token",
                            consent_prompt_key="oauth_consent_key",
                            client_secret="super_secure_client_secret_123456",
                            client_id="my_client_app_123",
                            audience="api.example.com",
                            scope=invalid_scope)


# ========== AUDIENCE VALIDATION ==========
@pytest.mark.parametrize(
    "valid_audience",
    [
        "https://api.example.com",  # Fully qualified HTTPS URI
        "https://service.internal.local",  # Internal HTTPS URI
        "my-api-service",  # Registered logical identifier
        "urn:my-service:api",  # URN format identifier
    ])
async def test_audience_field_valid(valid_audience):
    """
    Test valid audience configurations.
    """

    # Should not raise AuthCodeGrantConfigAudienceFieldError
    AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                        authorization_url="https://auth.example.com/oauth/authorize",
                        authorization_token_url="https://auth.example.com/oauth/token",
                        consent_prompt_key="oauth_consent_key",
                        client_secret="super_secure_client_secret_123456",
                        client_id="my_client_app_123",
                        audience=valid_audience,
                        scope=["read", "write"])


@pytest.mark.parametrize(
    "invalid_audience",
    [
        "",  # Empty audience
        "  api.example.com  ",  # Leading/trailing whitespace
        "*.example.com"  # Wildcard audience
    ])
async def test_audience_field_invalid(invalid_audience):
    """
    Test invalid audience configurations against RFC-compliant expectations.
    """
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigAudienceFieldError

    # Should raise AuthCodeGrantConfigAudienceFieldError
    with pytest.raises(AuthCodeGrantConfigAudienceFieldError):
        AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                            authorization_url="https://auth.example.com/oauth/authorize",
                            authorization_token_url="https://auth.example.com/oauth/token",
                            consent_prompt_key="oauth_consent_key",
                            client_secret="super_secure_client_secret_123456",
                            client_id="my_client_app_123",
                            audience=invalid_audience,
                            scope=["read", "write"])


async def test_state_parameter_validation():
    """Test Auth Code Grant Flow state parameter validation."""

    auth_code_grant_config = AuthCodeGrantConfig(client_server_url="https://test.com",
                                                 authorization_url="https://test.com/auth",
                                                 authorization_token_url="https://test.com/token",
                                                 consent_prompt_key="test_key_secure",
                                                 client_secret="test_secret_secure_16_chars_minimum",
                                                 client_id="test_client",
                                                 audience="test_audience",
                                                 scope=["test_scope"])

    # Verify state parameter is present and random
    assert auth_code_grant_config.state is not None

    # Very state parameter satifies minimum entropy
    assert len(auth_code_grant_config.state) >= 16

    # Test state parameter validation would prevent CSRF attacks
    different_state = "different_state_value"
    assert auth_code_grant_config.state != different_state
