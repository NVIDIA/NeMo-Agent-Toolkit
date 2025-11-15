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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import HeaderCred
from nat.plugins.mcp.auth.service_account.provider import MCPServiceAccountProvider
from nat.plugins.mcp.auth.service_account.provider_config import MCPServiceAccountProviderConfig
from nat.plugins.mcp.auth.service_account.provider_config import ServiceTokenConfig
from nat.plugins.mcp.auth.service_account.token_client import ServiceAccountTokenClient

# --------------------------------------------------------------------------- #
# Test Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def service_account_config() -> MCPServiceAccountProviderConfig:
    """Create a sample service account config for testing."""
    return MCPServiceAccountProviderConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",  # type: ignore
        token_url="https://auth.example.com/token",
        scopes="read write",
        service_token=ServiceTokenConfig(
            token="test_service_token",  # type: ignore
            header="X-Service-Account-Token",
        ),
    )


@pytest.fixture
def minimal_config() -> MCPServiceAccountProviderConfig:
    """Create minimal config without optional fields."""
    return MCPServiceAccountProviderConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",  # type: ignore
        token_url="https://auth.example.com/token",
        scopes="read write",
    )


@pytest.fixture
def mock_token_response():
    """Mock successful OAuth2 token response."""
    return {
        "access_token": "mock_access_token_12345", "token_type": "Bearer", "expires_in": 3600, "scope": "read write"
    }


# --------------------------------------------------------------------------- #
# Configuration Tests
# --------------------------------------------------------------------------- #


class TestServiceTokenConfig:
    """Test ServiceTokenConfig validation."""

    def test_valid_static_token_config(self):
        """Test valid configuration with static token."""
        config = ServiceTokenConfig(
            token="test_token",  # type: ignore
            header="X-Custom-Header",
        )
        assert config.token.get_secret_value() == "test_token"  # type: ignore
        assert config.header == "X-Custom-Header"
        assert config.function is None

    def test_valid_dynamic_function_config(self):
        """Test valid configuration with dynamic function."""
        config = ServiceTokenConfig(
            function="module.path.function_name",
            kwargs={"vault_path": "secrets/test"},
        )
        assert config.function == "module.path.function_name"
        assert config.kwargs == {"vault_path": "secrets/test"}
        assert config.token is None

    def test_validation_requires_token_or_function(self):
        """Test that either token or function must be provided."""
        with pytest.raises(ValueError, match="Either 'token' or 'function' must be provided"):
            ServiceTokenConfig()

    def test_validation_rejects_both_token_and_function(self):
        """Test that both token and function cannot be provided together."""
        with pytest.raises(ValueError, match="Cannot specify both 'token' and 'function'"):
            ServiceTokenConfig(
                token="test_token",  # type: ignore
                function="module.function",
            )

    def test_default_header_name(self):
        """Test default header name is X-Service-Account-Token."""
        config = ServiceTokenConfig(token="test")  # type: ignore
        assert config.header == "X-Service-Account-Token"


# --------------------------------------------------------------------------- #
# Token Client Tests
# --------------------------------------------------------------------------- #


class TestServiceAccountTokenClient:
    """Test OAuth2 token client functionality."""

    async def test_fetch_token_success(self, minimal_config, mock_token_response):
        """Test successful token fetching from OAuth2 server."""
        client = ServiceAccountTokenClient(
            client_id=minimal_config.client_id,
            client_secret=minimal_config.client_secret,
            token_url=minimal_config.token_url,
            scopes=minimal_config.scopes,
        )

        # Mock the OAuth2 token endpoint
        with patch("httpx.AsyncClient") as mock_http:
            mock_resp = MagicMock()  # Response object is sync, not async
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_token_response
            mock_http.return_value.__aenter__.return_value.post.return_value = mock_resp

            # Fetch token
            token = await client.get_access_token()

            # Verify token is returned as SecretStr
            assert isinstance(token, SecretStr)
            assert token.get_secret_value() == "mock_access_token_12345"

            # Verify OAuth2 request was made correctly
            mock_http.return_value.__aenter__.return_value.post.assert_called_once()
            call_args = mock_http.return_value.__aenter__.return_value.post.call_args

            # Verify the request URL
            assert call_args[0][0] == "https://auth.example.com/token"

            # Verify client credentials were sent in Basic Auth (not masked!)
            headers = call_args[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"].startswith("Basic ")
            # The Base64 encoded value should contain the actual secret, not **********

    async def test_fetch_token_caching(self, minimal_config, mock_token_response):
        """Test that tokens are cached and reused."""
        client = ServiceAccountTokenClient(
            client_id=minimal_config.client_id,
            client_secret=minimal_config.client_secret,
            token_url=minimal_config.token_url,
            scopes=minimal_config.scopes,
        )

        with patch("httpx.AsyncClient") as mock_http:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_token_response
            mock_http.return_value.__aenter__.return_value.post.return_value = mock_resp

            # First call fetches token
            token1 = await client.get_access_token()

            # Second call should use cache
            token2 = await client.get_access_token()

            # Tokens should be identical
            assert token1.get_secret_value() == token2.get_secret_value()

            # Only one HTTP request should have been made
            assert mock_http.return_value.__aenter__.return_value.post.call_count == 1

    async def test_fetch_token_401_unauthorized(self, minimal_config):
        """Test handling of invalid credentials (401 Unauthorized)."""
        client = ServiceAccountTokenClient(
            client_id=minimal_config.client_id,
            client_secret=minimal_config.client_secret,
            token_url=minimal_config.token_url,
            scopes=minimal_config.scopes,
        )

        with patch("httpx.AsyncClient") as mock_http:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_resp.text = "Invalid credentials"
            mock_http.return_value.__aenter__.return_value.post.return_value = mock_resp

            # Should raise RuntimeError with clear message
            with pytest.raises(RuntimeError, match="Invalid service account credentials"):
                await client.get_access_token()

    async def test_fetch_token_network_error(self, minimal_config):
        """Test handling of network errors."""
        import httpx

        client = ServiceAccountTokenClient(
            client_id=minimal_config.client_id,
            client_secret=minimal_config.client_secret,
            token_url=minimal_config.token_url,
            scopes=minimal_config.scopes,
        )

        with patch("httpx.AsyncClient") as mock_http:
            # Use httpx.RequestError which is what the code catches
            mock_http.return_value.__aenter__.return_value.post.side_effect = httpx.RequestError("Network error")

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Service account token request failed"):
                await client.get_access_token()


# --------------------------------------------------------------------------- #
# Provider Tests
# --------------------------------------------------------------------------- #


class TestMCPServiceAccountProvider:
    """Test service account authentication provider."""

    async def test_authenticate_success_with_service_token(self, service_account_config):
        """Test successful authentication with service token (dual authentication pattern)."""
        provider = MCPServiceAccountProvider(service_account_config)

        # Mock the token client to return OAuth2 access token
        with patch.object(provider._token_client, "get_access_token") as mock_get_token:
            mock_get_token.return_value = SecretStr("oauth2_access_token")

            # Authenticate
            result = await provider.authenticate(user_id="test_user")

            # Verify AuthResult structure
            assert isinstance(result, AuthResult)
            assert len(result.credentials) == 2  # Authorization + X-Service-Account-Token

            # Verify Authorization header (provider adds "Bearer " prefix)
            auth_cred = result.credentials[0]
            assert isinstance(auth_cred, HeaderCred)
            assert auth_cred.name == "Authorization"
            assert auth_cred.value.get_secret_value() == "Bearer oauth2_access_token"

            # Verify X-Service-Account-Token header
            service_cred = result.credentials[1]
            assert isinstance(service_cred, HeaderCred)
            assert service_cred.name == "X-Service-Account-Token"
            assert service_cred.value.get_secret_value() == "test_service_token"

    async def test_authenticate_success_without_service_token(self, minimal_config):
        """Test authentication without optional service token."""
        provider = MCPServiceAccountProvider(minimal_config)

        with patch.object(provider._token_client, "get_access_token") as mock_get_token:
            mock_get_token.return_value = SecretStr("oauth2_access_token")

            result = await provider.authenticate(user_id="test_user")

            # Should only have Authorization header (no service token)
            assert len(result.credentials) == 1
            assert result.credentials[0].name == "Authorization"
            assert result.credentials[0].value.get_secret_value() == "Bearer oauth2_access_token"

    async def test_authenticate_single_auth_pattern(self):
        """Test single authentication pattern (OAuth2 token only, no service token)."""
        config = MCPServiceAccountProviderConfig(
            client_id="test",
            client_secret="secret",  # type: ignore
            token_url="https://token.url",
            scopes="read",
        )
        provider = MCPServiceAccountProvider(config)

        with patch.object(provider._token_client, "get_access_token") as mock_get_token:
            mock_get_token.return_value = SecretStr("oauth2_token")

            result = await provider.authenticate()

            # Should have token with Bearer prefix from provider
            assert result.credentials[0].value.get_secret_value() == "Bearer oauth2_token"

    async def test_authenticate_propagates_token_client_errors(self, minimal_config):
        """Test that token client errors are propagated correctly."""
        provider = MCPServiceAccountProvider(minimal_config)

        with patch.object(provider._token_client, "get_access_token") as mock_get_token:
            mock_get_token.side_effect = RuntimeError("Invalid service account credentials")

            # Error should propagate
            with pytest.raises(RuntimeError, match="Invalid service account credentials"):
                await provider.authenticate(user_id="test_user")

    async def test_authenticate_with_dynamic_function_returning_tuple(self):
        """Test service token from dynamic function that returns (header, token) tuple."""

        # Create a mock async function that returns both header and token
        async def mock_get_service_token(vault_path="test", **kwargs):
            return ("X-Custom-Header", "dynamic_token_value")

        config = MCPServiceAccountProviderConfig(
            client_id="test",
            client_secret="secret",  # type: ignore
            token_url="https://token.url",
            scopes="read",
            service_token=ServiceTokenConfig(
                function="dummy.function.path",  # Will be mocked, not actually loaded
                kwargs={"vault_path": "secrets/test"},
            ),
        )

        # Mock the function loading to prevent import error
        with patch.object(MCPServiceAccountProvider, "_load_function", return_value=mock_get_service_token):
            provider = MCPServiceAccountProvider(config)

        with patch.object(provider._token_client, "get_access_token") as mock_get_token:
            mock_get_token.return_value = SecretStr("oauth2_token")

            result = await provider.authenticate()

            # Verify both headers
            assert len(result.credentials) == 2

            # Verify custom header from function
            service_cred = result.credentials[1]
            assert service_cred.name == "X-Custom-Header"
            assert service_cred.value.get_secret_value() == "dynamic_token_value"

    async def test_authenticate_with_dynamic_function_error_handling(self):
        """Test error handling when dynamic function fails."""

        # Create a mock async function that raises an error
        async def mock_failing_function(**kwargs):
            raise ValueError("Vault connection failed")

        config = MCPServiceAccountProviderConfig(
            client_id="test",
            client_secret="secret",  # type: ignore
            token_url="https://token.url",
            scopes="read",
            service_token=ServiceTokenConfig(
                function="dummy.function.path",  # Will be mocked, not actually loaded
            ),
        )

        # Mock the function loading to prevent import error
        with patch.object(MCPServiceAccountProvider, "_load_function", return_value=mock_failing_function):
            provider = MCPServiceAccountProvider(config)

        with patch.object(provider._token_client, "get_access_token") as mock_get_token:
            mock_get_token.return_value = SecretStr("oauth2_token")

            # Should raise RuntimeError with clear message
            with pytest.raises(RuntimeError, match="Failed to get service token from function"):
                await provider.authenticate()


# --------------------------------------------------------------------------- #
# Integration Tests
# --------------------------------------------------------------------------- #


class TestMCPServiceAccountIntegration:
    """Integration tests for complete authentication flow."""

    async def test_full_auth_flow_with_dual_headers(self, service_account_config, mock_token_response):
        """Test complete authentication flow with dual-header pattern (Jira/Jama scenario)."""
        provider = MCPServiceAccountProvider(service_account_config)

        # Mock OAuth2 server
        with patch("httpx.AsyncClient") as mock_http:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_token_response
            mock_http.return_value.__aenter__.return_value.post.return_value = mock_resp

            # Full authentication flow
            result = await provider.authenticate(user_id="test_user")

            # Verify complete AuthResult
            assert isinstance(result, AuthResult)
            assert len(result.credentials) == 2

            # Verify both headers are present
            auth_header = next(c for c in result.credentials if c.name == "Authorization")
            service_header = next(c for c in result.credentials if c.name == "X-Service-Account-Token")

            # Verify Authorization header format
            assert "Bearer " in auth_header.value.get_secret_value()
            assert "mock_access_token_12345" in auth_header.value.get_secret_value()

            # Verify service token header
            assert service_header.value.get_secret_value() == "test_service_token"

            # Verify token expiry is set
            assert result.token_expires_at is not None

    async def test_auth_flow_with_token_refresh(self, minimal_config, mock_token_response):
        """Test that expired tokens are automatically refreshed."""
        provider = MCPServiceAccountProvider(minimal_config)

        with patch("httpx.AsyncClient") as mock_http:
            mock_resp = MagicMock()
            mock_resp.status_code = 200

            # First call returns token with short expiry
            short_expiry_response = mock_token_response.copy()
            short_expiry_response["expires_in"] = 0  # Expires immediately
            mock_resp.json.return_value = short_expiry_response
            mock_http.return_value.__aenter__.return_value.post.return_value = mock_resp

            # First authentication
            result1 = await provider.authenticate(user_id="test_user")
            assert len(result1.credentials) == 1

            # Change mock to return fresh token
            fresh_response = mock_token_response.copy()
            fresh_response["access_token"] = "new_refreshed_token"
            mock_resp.json.return_value = fresh_response

            # Second authentication should fetch new token (cache expired)
            result2 = await provider.authenticate(user_id="test_user")
            assert len(result2.credentials) == 1

            # Should have made 2 HTTP requests (no caching due to immediate expiry)
            assert mock_http.return_value.__aenter__.return_value.post.call_count == 2

    async def test_end_to_end_oauth2_flow(self, service_account_config, mock_token_response):
        """Test end-to-end OAuth2 client credentials flow."""
        provider = MCPServiceAccountProvider(service_account_config)

        with patch("httpx.AsyncClient") as mock_http:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_token_response
            mock_http.return_value.__aenter__.return_value.post.return_value = mock_resp

            # Authenticate
            result = await provider.authenticate(user_id="test_user")

            # Verify OAuth2 request was made correctly
            call_args = mock_http.return_value.__aenter__.return_value.post.call_args

            # Verify URL
            assert call_args[0][0] == "https://auth.example.com/token"

            # Verify headers contain Basic Auth
            headers = call_args[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"].startswith("Basic ")

            # Verify request body contains grant type and scopes
            data = call_args[1]["data"]
            assert data["grant_type"] == "client_credentials"
            assert data["scope"] == "read write"

            # Verify result contains properly formatted credentials
            assert isinstance(result, AuthResult)
            assert len(result.credentials) == 2  # Authorization + Service token
            assert result.token_expires_at is not None
