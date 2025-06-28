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

import webbrowser
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantFlowError
from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig
from aiq.authentication.oauth2.auth_code_grant_config import ConsentPromptMode
from aiq.authentication.oauth2.auth_code_grant_manager import AuthCodeGrantManager
from aiq.authentication.request_manager import RequestManager
from aiq.authentication.response_manager import ResponseManager
from aiq.data_models.authentication import ExecutionMode


@pytest.fixture
def mock_request_manager():
    return AsyncMock(spec=RequestManager)


@pytest.fixture
def mock_response_manager():
    return AsyncMock(spec=ResponseManager)


@pytest.fixture
def auth_code_grant_manager(mock_request_manager: AsyncMock, mock_response_manager: AsyncMock):
    return AuthCodeGrantManager(mock_request_manager, mock_response_manager, ExecutionMode.SERVER)


# OAuth Auth Code Grant Validation Tests


async def test_auth_code_grant_valid_credentials(auth_code_grant_manager: AuthCodeGrantManager):
    """Test Auth Code Grant Flow credentials."""

    auth_code_grant_manager._encrypted_config = AuthCodeGrantConfig(access_token="valid_token",
                                                                    access_token_expires_in=datetime.now(timezone.utc) +
                                                                    timedelta(hours=1),
                                                                    client_server_url="https://test.com",
                                                                    authorization_url="https://test.com/auth",
                                                                    authorization_token_url="https://test.com/token",
                                                                    consent_prompt_key="test_key_secure",
                                                                    client_secret="test_secret_secure_16_chars_minimum",
                                                                    client_id="test_client",
                                                                    audience="test_audience",
                                                                    scope=["test_scope"])

    # Return True if access token is present and not expired.
    result = await auth_code_grant_manager.validate_authentication_credentials()
    assert result is True


async def test_auth_code_grant_credentials_expired(auth_code_grant_manager: AuthCodeGrantManager):
    """Test Auth Code Grant Flow credentials with expired access token."""

    auth_code_grant_manager._encrypted_config = AuthCodeGrantConfig(access_token="expired_token",
                                                                    access_token_expires_in=datetime.now(timezone.utc) -
                                                                    timedelta(hours=1),
                                                                    client_server_url="https://test.com",
                                                                    authorization_url="https://test.com/auth",
                                                                    authorization_token_url="https://test.com/token",
                                                                    consent_prompt_key="test_key_secure",
                                                                    client_secret="test_secret_secure_16_chars_minimum",
                                                                    client_id="test_client",
                                                                    audience="test_audience",
                                                                    scope=["test_scope"])

    # Return False if access token is expired.
    result = await auth_code_grant_manager.validate_authentication_credentials()
    assert result is False


async def test_auth_code_grant_no_access_token(auth_code_grant_manager: AuthCodeGrantManager):
    """Test Auth Code Grant Flow credentials without access token."""

    auth_code_grant_manager._encrypted_config = AuthCodeGrantConfig(client_server_url="https://test.com",
                                                                    authorization_url="https://test.com/auth",
                                                                    authorization_token_url="https://test.com/token",
                                                                    consent_prompt_key="test_key_secure",
                                                                    client_secret="test_secret_secure_16_chars_minimum",
                                                                    client_id="test_client",
                                                                    audience="test_audience",
                                                                    scope=["test_scope"])

    # Return False if access token is missing.
    result = await auth_code_grant_manager.validate_authentication_credentials()
    assert result is False


async def test_get_access_token_with_refresh_token_success(auth_code_grant_manager: AuthCodeGrantManager,
                                                           mock_request_manager: AsyncMock):
    """Test successful refresh token flow."""
    from aiq.authentication.credentials_manager import _CredentialsManager

    auth_code_grant_config = AuthCodeGrantConfig(
        access_token="mock_access_token",
        access_token_expires_in=datetime.now(timezone.utc) - timedelta(hours=1),  # expired
        client_id="test_client",
        client_secret="test_secret_secure_16_chars_minimum",
        refresh_token="valid_refresh_token",
        authorization_token_url="https://example.com/token",  # use dummy safe URL
        client_server_url="https://example.com",
        authorization_url="https://example.com/auth",
        consent_prompt_key="test_key_secure",
        audience="test_audience",
        scope=["test_scope"])

    # Assign mock request manager to the auth code grant manager
    auth_code_grant_manager._request_manager = mock_request_manager

    # Mock credentials manager encryption key
    _CredentialsManager().generate_credentials_encryption_key()

    _CredentialsManager()._authentication_configs["test_config"] = auth_code_grant_config

    _CredentialsManager().encrypt_authentication_configs()

    # Set expired access token to force refresh
    auth_code_grant_manager._encrypted_config = _CredentialsManager().get_authentication_config("test_config")

    # Setup mocked HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "new_access_token", "expires_in": 3600, "refresh_token": "new_refresh_token"
    }
    mock_request_manager.send_request.return_value = mock_response

    # Attempt to get refresh token.
    await auth_code_grant_manager._get_access_token_with_refresh_token()

    # Mock refresh token flow
    assert _CredentialsManager().decrypt_value(
        auth_code_grant_manager._encrypted_config.access_token) == "new_access_token"
    assert _CredentialsManager().decrypt_value(
        auth_code_grant_manager._encrypted_config.refresh_token) == "new_refresh_token"
    assert await auth_code_grant_manager.validate_authentication_credentials() is True


@pytest.fixture
def response_manager():
    return ResponseManager()


async def test_auth_code_grant_redirect_302_without_location(response_manager: ResponseManager):
    """Test handling of 302 redirect response without Location header."""

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 302
    mock_response.headers = {}

    auth_provider = AuthCodeGrantConfig(client_server_url="https://test.com",
                                        authorization_url="https://test.com/auth",
                                        authorization_token_url="https://test.com/token",
                                        consent_prompt_key="test_key_secure",
                                        client_secret="test_secret_secure_16_chars_minimum",
                                        client_id="test_client",
                                        audience="test_audience",
                                        scope=["test_scope"])

    # Raises AuthCodeGrantFlowError if Location header is missing.
    with pytest.raises(AuthCodeGrantFlowError):
        await response_manager.handle_auth_code_grant_response_codes(mock_response, auth_provider)


async def test_auth_code_grant_handles_400_response(response_manager: ResponseManager):
    """Test handling of 400 error response."""
    error_codes = [400, 401, 403, 404, 405, 422, 429]

    mock_response = MagicMock(spec=httpx.Response)

    auth_provider = AuthCodeGrantConfig(client_server_url="https://test.com",
                                        authorization_url="https://test.com/auth",
                                        authorization_token_url="https://test.com/token",
                                        consent_prompt_key="test_key_secure",
                                        client_secret="test_secret_secure_16_chars_minimum",
                                        client_id="test_client",
                                        audience="test_audience",
                                        scope=["test_scope"])

    # Raises AuthCodeGrantFlowError if response status code is in the 400 range
    for error_code in error_codes:
        mock_response.status_code = error_code
        with pytest.raises(AuthCodeGrantFlowError):
            await response_manager.handle_auth_code_grant_response_codes(mock_response, auth_provider)


async def test_auth_code_grant_handles_unknown_response_codes(response_manager: ResponseManager):
    """Test handling of unknown error code in HTTPresponse."""
    error_codes = [500, 502, 503, 504, 900, 899]

    mock_response = MagicMock(spec=httpx.Response)

    auth_provider = AuthCodeGrantConfig(client_server_url="https://test.com",
                                        authorization_url="https://test.com/auth",
                                        authorization_token_url="https://test.com/token",
                                        consent_prompt_key="test_key_secure",
                                        client_secret="test_secret_secure_16_chars_minimum",
                                        client_id="test_client",
                                        audience="test_audience",
                                        scope=["test_scope"])

    # Raises AuthCodeGrantFlowError if response status code is unknown.
    for error_code in error_codes:
        mock_response.status_code = error_code
        with pytest.raises(AuthCodeGrantFlowError):
            await response_manager.handle_auth_code_grant_response_codes(mock_response, auth_provider)


async def test_auth_code_grant_consent_browser_redirect_error_302(response_manager: ResponseManager):
    """Test handling of browser error in 302 consent browser."""

    location_header = "https://test.com/consent"
    auth_provider = AuthCodeGrantConfig(client_server_url="https://test.com",
                                        authorization_url="https://test.com/auth",
                                        authorization_token_url="https://test.com/token",
                                        consent_prompt_key="test_key_secure",
                                        client_secret="test_secret_secure_16_chars_minimum",
                                        client_id="test_client",
                                        audience="test_audience",
                                        scope=["test_scope"],
                                        consent_prompt_mode=ConsentPromptMode.BROWSER)

    # Raise AuthCodeGrantFlowError if browser error occurs.
    with patch('webbrowser.get', side_effect=webbrowser.Error("Browser error")):

        with pytest.raises(AuthCodeGrantFlowError):
            await response_manager._handle_auth_code_grant_302_consent_browser(location_header, auth_provider)


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


# ========== CLIENT_SERVER_URL VALIDATION ==========


def test_client_server_url_valid():
    """Test valid client_server_url configurations."""

    valid_client_server_urls = [
        "https://api.example.com",  # HTTPS URL
        "http://localhost:8000",  # HTTP localhost
        "http://127.0.0.1:8080",  # HTTP localhost IP
        "https://localhost:8443",  # HTTPS localhost
        "https://service.internal.domain",  # Internal domain
        "http://localhost",  # HTTP localhost without port
    ]

    for client_server_url in valid_client_server_urls:
        config = AuthCodeGrantConfig(client_server_url=client_server_url,
                                     authorization_url="https://auth.example.com/oauth/authorize",
                                     authorization_token_url="https://auth.example.com/oauth/token",
                                     consent_prompt_key="oauth_consent_key",
                                     client_secret="super_secure_client_secret_123456",
                                     client_id="my_client_app_123",
                                     audience="api.example.com",
                                     scope=["read", "write"])
        assert config.client_server_url == client_server_url


def test_client_server_url_invalid():
    """Test invalid client_server_url configurations."""
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientServerUrlFieldError

    invalid_client_server_urls = [
        "",  # Empty URL (value_missing)
        "ftp://example.com",  # Invalid scheme (invalid_scheme)
        "//example.com",  # No scheme (invalid_scheme)
        "https://",  # No hostname (invalid_url)
        "mailto:test@example.com",  # Invalid scheme (invalid_scheme)
    ]

    for client_server_url in invalid_client_server_urls:
        with pytest.raises(AuthCodeGrantConfigClientServerUrlFieldError):
            AuthCodeGrantConfig(client_server_url=client_server_url,
                                authorization_url="https://auth.example.com/oauth/authorize",
                                authorization_token_url="https://auth.example.com/oauth/token",
                                consent_prompt_key="oauth_consent_key",
                                client_secret="super_secure_client_secret_123456",
                                client_id="my_client_app_123",
                                audience="api.example.com",
                                scope=["read", "write"])


# ========== OAUTH URLS HTTPS VALIDATION ==========


def test_authorization_url_valid():
    """Test valid authorization_url configurations following RFC 6819 Section 4.2.1."""

    valid_authorization_urls = [
        "https://auth.example.com/oauth/authorize",  # Standard OAuth endpoint
        "https://login.microsoftonline.com/tenant/oauth2/v2.0/authorize",  # Microsoft Azure AD
        "https://accounts.google.com/o/oauth2/auth",  # Google OAuth
        "https://github.com/login/oauth/authorize",  # GitHub OAuth
        "https://api.internal.company.com/auth/oauth/authorize",  # Internal company API
    ]

    for authorization_url in valid_authorization_urls:
        config = AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                     authorization_url=authorization_url,
                                     authorization_token_url="https://auth.example.com/oauth/token",
                                     consent_prompt_key="oauth_consent_key",
                                     client_secret="super_secure_client_secret_123456",
                                     client_id="my_client_app_123",
                                     audience="api.example.com",
                                     scope=["read", "write"])
        assert config.authorization_url == authorization_url


def test_authorization_url_invalid():
    """Test invalid authorization_url configurations."""
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigAuthorizationUrlFieldError

    invalid_authorization_urls = [
        "",  # Empty URL (value_missing)
        "https://",  # No hostname (invalid_url)
        "https://auth.example.com",  # No path (path_missing)
        "https://auth.example.com/",  # Root path only (path_missing)
        "ftp://auth.example.com/oauth/authorize",  # Invalid scheme (https_required)
    ]

    for authorization_url in invalid_authorization_urls:
        with pytest.raises(AuthCodeGrantConfigAuthorizationUrlFieldError):
            AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                authorization_url=authorization_url,
                                authorization_token_url="https://auth.example.com/oauth/token",
                                consent_prompt_key="oauth_consent_key",
                                client_secret="super_secure_client_secret_123456",
                                client_id="my_client_app_123",
                                audience="api.example.com",
                                scope=["read", "write"])


def test_authorization_token_url_valid():
    """Test valid authorization_token_url configurations following RFC 6819 Section 4.2.1."""

    valid_authorization_token_urls = [
        "https://auth.example.com/oauth/token",  # Standard OAuth token endpoint
        "https://login.microsoftonline.com/tenant/oauth2/v2.0/token",  # Microsoft Azure AD
        "https://oauth2.googleapis.com/token",  # Google OAuth
        "https://github.com/login/oauth/access_token",  # GitHub OAuth
        "https://api.internal.company.com/auth/oauth/token",  # Internal company API
    ]

    for authorization_token_url in valid_authorization_token_urls:
        config = AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                     authorization_url="https://auth.example.com/oauth/authorize",
                                     authorization_token_url=authorization_token_url,
                                     consent_prompt_key="oauth_consent_key",
                                     client_secret="super_secure_client_secret_123456",
                                     client_id="my_client_app_123",
                                     audience="api.example.com",
                                     scope=["read", "write"])
        assert config.authorization_token_url == authorization_token_url


def test_authorization_token_url_invalid():
    """Test invalid authorization_token_url configurations."""
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigAuthorizationUrlFieldError

    invalid_authorization_token_urls = [
        "",  # Empty URL (value_missing)
        "http://auth.example.com/oauth/token",  # HTTP instead of HTTPS (https_required)
        "https://",  # No hostname (invalid_url)
        "https://auth.example.com",  # No path (path_missing)
        "https://auth.example.com/",  # Root path only (path_missing)
        "ftp://auth.example.com/oauth/token",  # Invalid scheme (https_required)
    ]

    for authorization_token_url in invalid_authorization_token_urls:
        with pytest.raises(AuthCodeGrantConfigAuthorizationUrlFieldError):
            AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                authorization_url="https://auth.example.com/oauth/authorize",
                                authorization_token_url=authorization_token_url,
                                consent_prompt_key="oauth_consent_key",
                                client_secret="super_secure_client_secret_123456",
                                client_id="my_client_app_123",
                                audience="api.example.com",
                                scope=["read", "write"])


# ========== CONSENT_PROMPT_KEY VALIDATION ==========


def test_consent_prompt_key_valid():
    """Test valid consent_prompt_key configurations."""

    valid_consent_prompt_keys = [
        "oauth_consent_key",  # Standard key (18 chars)
        "12345678",  # Minimum length (8 chars)
        "my_secure_prompt_key_123",  # Longer key with numbers
        "ConsentKey123!@#",  # Mixed case with special chars
        "a" * 32,  # Very long key (32 chars)
        "user-auth-prompt-key",  # Dashes allowed
    ]

    for consent_prompt_key in valid_consent_prompt_keys:
        config = AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                     authorization_url="https://auth.example.com/oauth/authorize",
                                     authorization_token_url="https://auth.example.com/oauth/token",
                                     consent_prompt_key=consent_prompt_key,
                                     client_secret="super_secure_client_secret_123456",
                                     client_id="my_client_app_123",
                                     audience="api.example.com",
                                     scope=["read", "write"])
        assert config.consent_prompt_key == consent_prompt_key


def test_consent_prompt_key_invalid():
    """Test invalid consent_prompt_key configurations."""
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigConsentPromptKeyFieldError

    invalid_consent_prompt_keys = [
        "",  # Empty key (value_missing)
        "1234567",  # Too short - 7 chars (value_too_short)
        "  oauth_key  ",  # Leading/trailing whitespace (whitespace_found)
        " key",  # Leading whitespace (whitespace_found)
        "key ",  # Trailing whitespace (whitespace_found)
        "short",  # Too short - 5 chars (value_too_short)
    ]

    for consent_prompt_key in invalid_consent_prompt_keys:
        with pytest.raises(AuthCodeGrantConfigConsentPromptKeyFieldError):
            AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                authorization_url="https://auth.example.com/oauth/authorize",
                                authorization_token_url="https://auth.example.com/oauth/token",
                                consent_prompt_key=consent_prompt_key,
                                client_secret="super_secure_client_secret_123456",
                                client_id="my_client_app_123",
                                audience="api.example.com",
                                scope=["read", "write"])


# ========== CLIENT_SECRET VALIDATION ==========


def test_client_secret_valid():
    """
    Test valid client_secret configurations.

    RFC 6819 Section 5.1.4.2.2 - Use High Entropy for Secrets
    When creating secrets not intended for usage by human users (e.g.,
    client secrets or token handles), the authorization server should
    include a reasonable level of entropy in order to mitigate the risk
    of guessing attacks. The token value should be >=128 bits long and
    constructed from a cryptographically strong random or pseudo-random
    number sequence.
    """

    valid_secrets = [
        "super_secure_client_secret_123456",  # Default valid secret (34 chars)
        "1234567890123456",  # Exactly 16 characters
        "a" * 64  # Very long secret (64 chars)
    ]

    for secret in valid_secrets:
        config = AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                     authorization_url="https://auth.example.com/oauth/authorize",
                                     authorization_token_url="https://auth.example.com/oauth/token",
                                     consent_prompt_key="oauth_consent_key",
                                     client_secret=secret,
                                     client_id="my_client_app_123",
                                     audience="api.example.com",
                                     scope=["read", "write"])
        assert config.client_secret == secret


def test_client_secret_invalid():
    """
    Test valid Auth Code Grant Config client secret.
    """

    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientSecretFieldError

    invalid_secrets = [
        "",  # Empty secret
        "short"  # Too short (less than 16 chars)
    ]

    for secret in invalid_secrets:
        with pytest.raises(AuthCodeGrantConfigClientSecretFieldError):
            AuthCodeGrantConfig(client_server_url="https://test.com",
                                authorization_url="https://test.com/auth",
                                authorization_token_url="https://test.com/token",
                                consent_prompt_key="test_key_secure",
                                client_secret=secret,
                                client_id="test_client",
                                audience="test_audience",
                                scope=["test_scope"])


# ========== CLIENT_ID VALIDATION ==========


def test_client_id_valid():
    """Test valid client_id configurations."""

    valid_client_ids = ["my_client_app_123", "abc", "my_application_client_id_123"]

    for client_id in valid_client_ids:
        config = AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                     authorization_url="https://auth.example.com/oauth/authorize",
                                     authorization_token_url="https://auth.example.com/oauth/token",
                                     consent_prompt_key="oauth_consent_key",
                                     client_secret="super_secure_client_secret_123456",
                                     client_id=client_id,
                                     audience="api.example.com",
                                     scope=["read", "write"])
        assert config.client_id == client_id


def test_client_id_invalid():
    """Test invalid client_id configurations."""
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientIDFieldError

    invalid_client_ids = [
        "",  # Empty client ID
        "  client_with_spaces  "  # Client ID with whitespace
    ]

    for client_id in invalid_client_ids:
        with pytest.raises(AuthCodeGrantConfigClientIDFieldError):
            AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                authorization_url="https://auth.example.com/oauth/authorize",
                                authorization_token_url="https://auth.example.com/oauth/token",
                                consent_prompt_key="oauth_consent_key",
                                client_secret="super_secure_client_secret_123456",
                                client_id=client_id,
                                audience="api.example.com",
                                scope=["read", "write"])


# ========== SCOPE VALIDATION ==========


def test_scope_valid():
    """
    Test valid scope configurations.

    RFC 6819 Section 4.3.1 - Access Token Disclosure to Unintended Recipients
    Scopes should follow principle of least privilege.
    """

    valid_scopes = [
        ["read", "write"],  # Default valid scopes
        ["read"],  # Single scope
        ["read", "write", "update", "delete"],  # Multiple limited scopes
        [f"scope_{i}" for i in range(10)]  # Maximum allowed scopes (10)
    ]

    for scope in valid_scopes:
        config = AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                     authorization_url="https://auth.example.com/oauth/authorize",
                                     authorization_token_url="https://auth.example.com/oauth/token",
                                     consent_prompt_key="oauth_consent_key",
                                     client_secret="super_secure_client_secret_123456",
                                     client_id="my_client_app_123",
                                     audience="api.example.com",
                                     scope=scope)
        assert config.scope == scope


def test_scope_invalid():
    """
    Test invalid scope configurations.

    RFC 6819 Section 4.3.1 - Access Token Disclosure to Unintended Recipients
    Scopes should follow principle of least privilege.
    """
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigScopeFieldError

    invalid_scopes = [
        [],  # Empty scope list
        ["read", "write", "read"],  # Duplicate scopes
        ["read", "", "write"],  # Empty/whitespace scopes
        ["*", "all", "root", "admin", "superuser"]  # Dangerous scopes
    ]

    for scope in invalid_scopes:
        with pytest.raises(AuthCodeGrantConfigScopeFieldError):
            AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                authorization_url="https://auth.example.com/oauth/authorize",
                                authorization_token_url="https://auth.example.com/oauth/token",
                                consent_prompt_key="oauth_consent_key",
                                client_secret="super_secure_client_secret_123456",
                                client_id="my_client_app_123",
                                audience="api.example.com",
                                scope=scope)


# ========== AUDIENCE VALIDATION ==========


def test_audience_field_valid():
    """
    Test valid audience configurations.
    """

    valid_audiences = [
        "https://api.example.com",  # Fully qualified HTTPS URI
        "https://service.internal.local",  # Internal HTTPS URI
        "my-api-service",  # Registered logical identifier
        "urn:my-service:api",  # URN format identifier
    ]

    for audience in valid_audiences:
        config = AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                     authorization_url="https://auth.example.com/oauth/authorize",
                                     authorization_token_url="https://auth.example.com/oauth/token",
                                     consent_prompt_key="oauth_consent_key",
                                     client_secret="super_secure_client_secret_123456",
                                     client_id="my_client_app_123",
                                     audience=audience,
                                     scope=["read", "write"])
        assert config.audience == audience


def test_audience_field_invalid():
    """
    Test invalid audience configurations against RFC-compliant expectations.
    """
    from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigAudienceFieldError

    invalid_audiences = [
        "",  # Empty audience
        "  api.example.com  ",  # Leading/trailing whitespace
        "*.example.com"  # Wildcard audience
    ]

    for audience in invalid_audiences:
        with pytest.raises(AuthCodeGrantConfigAudienceFieldError):
            AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                authorization_url="https://auth.example.com/oauth/authorize",
                                authorization_token_url="https://auth.example.com/oauth/token",
                                consent_prompt_key="oauth_consent_key",
                                client_secret="super_secure_client_secret_123456",
                                client_id="my_client_app_123",
                                audience=audience,
                                scope=["read", "write"])
