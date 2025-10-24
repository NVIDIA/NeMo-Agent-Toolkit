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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.plugins.mcp.auth.auth_provider import DiscoverOAuth2Endpoints
from nat.plugins.mcp.auth.auth_provider import DynamicClientRegistration
from nat.plugins.mcp.auth.auth_provider import MCPOAuth2Provider
from nat.plugins.mcp.auth.auth_provider import OAuth2Credentials
from nat.plugins.mcp.auth.auth_provider import OAuth2Endpoints
from nat.plugins.mcp.auth.auth_provider_config import MCPOAuth2ProviderConfig

# --------------------------------------------------------------------------- #
# Test Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_config() -> MCPOAuth2ProviderConfig:
    """Create a mock MCP OAuth2 provider config for testing."""
    return MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_name="Test Client",
        enable_dynamic_registration=True,
    )


@pytest.fixture
def mock_config_with_credentials() -> MCPOAuth2ProviderConfig:
    """Create a mock config with pre-registered credentials."""
    return MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client_id",
        client_secret="test_client_secret",
        client_name="Test Client",
        enable_dynamic_registration=False,
    )


@pytest.fixture
def mock_endpoints() -> OAuth2Endpoints:
    """Create mock OAuth2 endpoints for testing."""
    return OAuth2Endpoints(
        authorization_url="https://auth.example.com/authorize",  # type: ignore
        token_url="https://auth.example.com/token",  # type: ignore
        registration_url="https://auth.example.com/register",  # type: ignore
    )


@pytest.fixture
def mock_credentials() -> OAuth2Credentials:
    """Create mock OAuth2 credentials for testing."""
    return OAuth2Credentials(
        client_id="test_client_id",
        client_secret="test_client_secret",
    )


# --------------------------------------------------------------------------- #
# DiscoverOAuth2Endpoints Tests
# --------------------------------------------------------------------------- #


class TestDiscoverOAuth2Endpoints:
    """Test the DiscoverOAuth2Endpoints class."""

    async def test_discover_cached_endpoints(self, mock_config):
        """Test that cached endpoints are returned for non-401 requests."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        # Set up cached endpoints
        cached_endpoints = OAuth2Endpoints(
            authorization_url="https://auth.example.com/authorize",  # type: ignore
            token_url="https://auth.example.com/token",  # type: ignore
        )
        discoverer._cached_endpoints = cached_endpoints

        # Test normal request returns cached endpoints
        endpoints, changed = await discoverer.discover(response=None)

        assert endpoints == cached_endpoints
        assert changed is False

    async def test_discover_with_www_authenticate_hint(self, mock_config):
        """Test discovery using WWW-Authenticate header hint."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        # Mock the protected resource metadata response
        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock(return_value=None)
            mock_resp.aread.return_value = b'{"authorization_servers": ["https://auth.example.com"]}'
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp

            # Mock OAuth metadata response
            with patch.object(discoverer, '_discover_via_issuer_or_base') as mock_discover:
                mock_discover.return_value = OAuth2Endpoints(
                    authorization_url="https://auth.example.com/authorize",  # type: ignore
                    token_url="https://auth.example.com/token",  # type: ignore
                    registration_url="https://auth.example.com/register",  # type: ignore
                )

                # Create a mock 401 response with WWW-Authenticate header
                mock_response = AsyncMock()
                mock_response.status_code = 401
                mock_response.headers = {
                    'WWW-Authenticate':
                        'Bearer realm="api", resource_metadata="https://auth.example.com/.well-known/oauth-protected-resource"'
                }

                endpoints, changed = await discoverer.discover(response=mock_response)

                assert endpoints is not None
                assert changed is True

    async def test_discover_fallback_to_server_base(self, mock_config):
        """Test discovery falls back to server base URL when no hint provided."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        with patch.object(discoverer, '_discover_via_issuer_or_base') as mock_discover:
            mock_discover.return_value = OAuth2Endpoints(
                authorization_url="https://auth.example.com/authorize",  # type: ignore
                token_url="https://auth.example.com/token",  # type: ignore
            )

            endpoints, changed = await discoverer.discover(response=None)

            assert endpoints is not None
            assert changed is True
            mock_discover.assert_called_once_with("https://example.com/mcp")

    def test_extract_from_www_authenticate_header(self, mock_config):
        """Test extracting resource_metadata URL from WWW-Authenticate header."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        # Test with double quotes
        url = discoverer._extract_from_www_authenticate_header(
            'Bearer realm="api", resource_metadata="https://auth.example.com/.well-known/oauth-protected-resource"')
        assert url == "https://auth.example.com/.well-known/oauth-protected-resource"

        # Test with single quotes
        url = discoverer._extract_from_www_authenticate_header(
            "Bearer realm='api', resource_metadata='https://auth.example.com/.well-known/oauth-protected-resource'")
        assert url == "https://auth.example.com/.well-known/oauth-protected-resource"

        # Test without quotes
        url = discoverer._extract_from_www_authenticate_header(
            "Bearer realm=api, resource_metadata=https://auth.example.com/.well-known/oauth-protected-resource")
        assert url == "https://auth.example.com/.well-known/oauth-protected-resource"

        # Test case insensitive
        url = discoverer._extract_from_www_authenticate_header(
            "Bearer realm=api, RESOURCE_METADATA=https://auth.example.com/.well-known/oauth-protected-resource")
        assert url == "https://auth.example.com/.well-known/oauth-protected-resource"

        # Test no match
        url = discoverer._extract_from_www_authenticate_header("Bearer realm=api")
        assert url is None

    def test_authorization_base_url(self, mock_config):
        """Test extracting authorization base URL from server URL."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        base_url = discoverer._authorization_base_url()
        assert base_url == "https://example.com"

    def test_build_path_aware_discovery_urls(self, mock_config):
        """Test building path-aware discovery URLs."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        # Test with path
        urls = discoverer._build_path_aware_discovery_urls("https://auth.example.com/api/v1")
        expected = [
            "https://auth.example.com/.well-known/oauth-protected-resource/api/v1",
            "https://auth.example.com/api/v1/.well-known/oauth-authorization-server",
            "https://auth.example.com/.well-known/oauth-authorization-server",
            "https://auth.example.com/.well-known/openid-configuration/api/v1",
            "https://auth.example.com/api/v1/.well-known/openid-configuration",
            "https://auth.example.com/api/v1/.well-known/openid-configuration",
        ]
        assert urls == expected

        # Test without path
        urls = discoverer._build_path_aware_discovery_urls("https://auth.example.com")
        expected = [
            "https://auth.example.com/.well-known/oauth-authorization-server",
            "https://auth.example.com/.well-known/openid-configuration",
        ]
        assert urls == expected


# --------------------------------------------------------------------------- #
# DynamicClientRegistration Tests
# --------------------------------------------------------------------------- #


class TestDynamicClientRegistration:
    """Test the DynamicClientRegistration class."""

    async def test_register_success(self, mock_config, mock_endpoints):
        """Test successful client registration."""
        registrar = DynamicClientRegistration(mock_config)

        # Mock the registration response
        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock(return_value=None)
            mock_resp.aread.return_value = b'{"client_id": "registered_client_id",\
            "client_secret": "registered_client_secret", "redirect_uris": ["https://example.com/callback"]}'

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp

            credentials = await registrar.register(mock_endpoints, ["read", "write"])

            assert credentials.client_id == "registered_client_id"
            assert credentials.client_secret == "registered_client_secret"

    async def test_register_without_registration_url(self, mock_config):
        """Test registration falls back to /register when no registration URL provided."""
        registrar = DynamicClientRegistration(mock_config)

        endpoints = OAuth2Endpoints(
            authorization_url="https://auth.example.com/authorize",  # type: ignore
            token_url="https://auth.example.com/token",  # type: ignore
            registration_url=None,
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock(return_value=None)
            mock_resp.aread.return_value = b'{"client_id": "registered_client_id", "redirect_uris": ["https://example.com/callback"]}'
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp

            credentials = await registrar.register(endpoints, None)

            assert credentials.client_id == "registered_client_id"
            # Verify it used the fallback URL
            mock_client.return_value.__aenter__.return_value.post.assert_called_once()
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            assert call_args[0][0] == "https://example.com/register"

    async def test_register_invalid_response(self, mock_config, mock_endpoints):
        """Test registration fails with invalid JSON response."""
        registrar = DynamicClientRegistration(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock(return_value=None)
            mock_resp.aread.return_value = b'invalid json'
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp

            with pytest.raises(RuntimeError, match="Registration response was not valid"):
                await registrar.register(mock_endpoints, None)

    async def test_register_missing_client_id(self, mock_config, mock_endpoints):
        """Test registration fails when no client_id is returned."""
        registrar = DynamicClientRegistration(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock(return_value=None)
            mock_resp.aread.return_value = b'{"client_secret": "secret", "redirect_uris": ["https://example.com/callback"]}'
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp

            with pytest.raises(RuntimeError):
                await registrar.register(mock_endpoints, None)


# --------------------------------------------------------------------------- #
# MCPOAuth2Provider Tests
# --------------------------------------------------------------------------- #


class TestMCPOAuth2Provider:
    """Test the MCPOAuth2Provider class."""

    async def test_authenticate_requires_user_id_raises(self, mock_config):
        """authenticate must be called with user_id; otherwise it raises."""
        provider = MCPOAuth2Provider(mock_config)

        with pytest.raises(RuntimeError, match="User is not authorized to call the tool"):
            await provider.authenticate()

        with pytest.raises(RuntimeError, match="User is not authorized to call the tool"):
            await provider.authenticate(user_id=None)

    async def test_authenticate_normal_request_returns_empty_when_no_provider(self, mock_config):
        """Test that normal requests return empty auth result when no provider is set up."""
        provider = MCPOAuth2Provider(mock_config)

        result = await provider.authenticate(user_id="test_user")

        assert result.credentials == []
        assert result.token_expires_at is None
        assert result.raw == {}

    async def test_authenticate_with_manual_credentials(self, mock_config_with_credentials, mock_endpoints):
        """Test authentication with pre-registered credentials."""
        provider = MCPOAuth2Provider(mock_config_with_credentials)

        # Mock the discovery process
        with patch.object(provider._discoverer, 'discover') as mock_discover:
            mock_discover.return_value = (mock_endpoints, True)

            # Mock the OAuth2 flow
            mock_auth_result = AuthResult(
                credentials=[BearerTokenCred(token=SecretStr("test_token"))],
                token_expires_at=None,
                raw={},
            )

            with patch.object(provider, '_nat_oauth2_authenticate') as mock_flow:
                mock_flow.return_value = mock_auth_result

                result = await provider.authenticate(user_id="test_user")

                assert result == mock_auth_result
                # Discovery is not called in normal authentication flow (only on 401 responses)
                mock_discover.assert_not_called()
                mock_flow.assert_called_once()

    async def test_authenticate_with_dynamic_registration(self, mock_config, mock_endpoints, mock_credentials):
        """Test authentication with dynamic client registration."""
        provider = MCPOAuth2Provider(mock_config)

        # Mock the discovery process
        with patch.object(provider._discoverer, 'discover') as mock_discover:
            mock_discover.return_value = (mock_endpoints, True)

            # Mock the registration process
            with patch.object(provider._registrar, 'register') as mock_register:
                mock_register.return_value = mock_credentials

                # Mock the OAuth2 flow
                mock_auth_result = AuthResult(
                    credentials=[BearerTokenCred(token=SecretStr("test_token"))],
                    token_expires_at=None,
                    raw={},
                )

                with patch.object(provider, '_nat_oauth2_authenticate') as mock_flow:
                    mock_flow.return_value = mock_auth_result

                    result = await provider.authenticate(user_id="test_user")

                    assert result == mock_auth_result
                    # Discovery is not called in normal authentication flow (only on 401 responses)
                    mock_discover.assert_not_called()
                    mock_register.assert_not_called()
                    mock_flow.assert_called_once()

    async def test_authenticate_dynamic_registration_disabled(self, mock_endpoints):
        """Test authentication works when dynamic registration is disabled but valid credentials provided."""
        config = MCPOAuth2ProviderConfig(
            server_url="https://example.com/mcp",  # type: ignore
            redirect_uri="https://example.com/callback",  # type: ignore
            client_id="test_client_id",
            client_secret="test_client_secret",
            enable_dynamic_registration=False,
        )
        provider = MCPOAuth2Provider(config)

        # Mock the discovery process and OAuth flow
        with patch.object(provider._discoverer, 'discover') as mock_discover:
            mock_discover.return_value = (mock_endpoints, True)

            with patch.object(provider, '_nat_oauth2_authenticate') as mock_flow:
                mock_auth_result = AuthResult(credentials=[], token_expires_at=None, raw={})
                mock_flow.return_value = mock_auth_result

                # Should succeed with manual credentials
                result = await provider.authenticate(user_id="test_user")

                assert result == mock_auth_result
                # Discovery is not called in normal authentication flow (only on 401 responses)
                mock_discover.assert_not_called()
                mock_flow.assert_called_once()

    async def test_effective_scopes_uses_config_scopes(self):
        """Test that effective scopes uses config scopes when provided."""
        config = MCPOAuth2ProviderConfig(
            server_url="https://example.com/mcp",  # type: ignore
            redirect_uri="https://example.com/callback",  # type: ignore
            scopes=["read", "write"],
            enable_dynamic_registration=True,
        )
        provider = MCPOAuth2Provider(config)

        scopes = provider._effective_scopes
        assert scopes == ["read", "write"]

    async def test_effective_scopes_falls_back_to_discovered(self, mock_config):
        """Test that effective scopes falls back to discovered scopes when config scopes not provided."""
        provider = MCPOAuth2Provider(mock_config)

        # Mock discovered scopes
        provider._cached_endpoints = OAuth2Endpoints(
            authorization_url="https://auth.example.com/authorize",  # type: ignore
            token_url="https://auth.example.com/token",  # type: ignore
            scopes=["discovered_scope"])

        scopes = provider._effective_scopes
        assert scopes == ["discovered_scope"]

    async def test_effective_scopes_returns_empty_when_none_available(self, mock_config):
        """Test that effective scopes returns empty list when no scopes available."""
        provider = MCPOAuth2Provider(mock_config)

        scopes = provider._effective_scopes
        assert scopes == []

    async def test_fetch_pr_issuer_success(self, mock_config):
        """Test successful protected resource issuer fetching."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock(return_value=None)
            mock_resp.aread.return_value = b'{"resource": "https://example.com/api", "authorization_servers": ["https://auth.example.com"]}'
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp

            issuer = await discoverer._fetch_pr_issuer("https://example.com/.well-known/oauth-protected-resource")

            assert issuer == "https://auth.example.com/"

    async def test_fetch_pr_issuer_invalid_json(self, mock_config):
        """Test protected resource issuer fetching with invalid JSON."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock(return_value=None)
            mock_resp.aread.return_value = b'invalid json'
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp

            issuer = await discoverer._fetch_pr_issuer("https://example.com/.well-known/oauth-protected-resource")

            assert issuer is None

    async def test_fetch_pr_issuer_no_authorization_servers(self, mock_config):
        """Test protected resource issuer fetching with no authorization servers."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock(return_value=None)
            mock_resp.aread.return_value = b'{"resource": "https://example.com/api", "other_field": "value"}'
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp

            issuer = await discoverer._fetch_pr_issuer("https://example.com/.well-known/oauth-protected-resource")

            assert issuer is None

    async def test_discover_via_issuer_or_base_success(self, mock_config):
        """Test successful discovery via issuer or base URL."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_resp.aread.return_value = (b'{"issuer": "https://auth.example.com", '
                                            b'"authorization_endpoint": "https://auth.example.com/authorize", '
                                            b'"token_endpoint": "https://auth.example.com/token", '
                                            b'"registration_endpoint": "https://auth.example.com/register"}')
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp

            endpoints = await discoverer._discover_via_issuer_or_base("https://auth.example.com")

            assert endpoints is not None
            assert str(endpoints.authorization_url) == "https://auth.example.com/authorize"
            assert str(endpoints.token_url) == "https://auth.example.com/token"
            assert str(endpoints.registration_url) == "https://auth.example.com/register"

    async def test_discover_via_issuer_or_base_no_authorization_endpoint(self, mock_config):
        """Test discovery with missing authorization endpoint."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_resp.aread.return_value = b'{"token_endpoint": "https://auth.example.com/token"}'
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp

            endpoints = await discoverer._discover_via_issuer_or_base("https://auth.example.com")

            assert endpoints is None

    async def test_discover_via_issuer_or_base_no_token_endpoint(self, mock_config):
        """Test discovery with missing token endpoint."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_resp.aread.return_value = b'{"authorization_endpoint": "https://auth.example.com/authorize"}'
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp

            endpoints = await discoverer._discover_via_issuer_or_base("https://auth.example.com")

            assert endpoints is None

    async def test_discover_via_issuer_or_base_invalid_json(self, mock_config):
        """Test discovery with invalid JSON response."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_resp.aread.return_value = b'invalid json'
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp

            endpoints = await discoverer._discover_via_issuer_or_base("https://auth.example.com")

            assert endpoints is None

    async def test_discover_via_issuer_or_base_non_200_status(self, mock_config):
        """Test discovery with non-200 status code."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.status_code = 404
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp

            endpoints = await discoverer._discover_via_issuer_or_base("https://auth.example.com")

            assert endpoints is None

    async def test_discover_via_issuer_or_base_exception_handling(self, mock_config):
        """Test discovery with exception during request."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Network error")

            endpoints = await discoverer._discover_via_issuer_or_base("https://auth.example.com")

            assert endpoints is None

    async def test_register_with_scopes(self, mock_config, mock_endpoints):
        """Test registration with scopes."""
        registrar = DynamicClientRegistration(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock(return_value=None)
            mock_resp.aread.return_value = b'{"client_id": "test_client_id", "client_secret": "test_secret",\
                "redirect_uris": ["https://example.com/callback"]}'

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp

            credentials = await registrar.register(mock_endpoints, ["read", "write"])

            assert credentials.client_id == "test_client_id"
            assert credentials.client_secret == "test_secret"

    async def test_register_with_token_endpoint_auth_method(self, mock_config, mock_endpoints):
        """Test registration with custom token endpoint auth method."""
        config = mock_config.model_copy(update={'token_endpoint_auth_method': 'none'})
        registrar = DynamicClientRegistration(config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock(return_value=None)
            mock_resp.aread.return_value = b'{"client_id": "test_client_id", "client_secret": "test_secret",\
                "redirect_uris": ["https://example.com/callback"]}'

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp

            credentials = await registrar.register(mock_endpoints, None)

            assert credentials.client_id == "test_client_id"
            # Verify the correct auth method was used in the request
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            request_data = call_args[1]['json']
            assert request_data['token_endpoint_auth_method'] == 'none'

    async def test_discover_and_register_with_endpoints_changed(self, mock_config):
        """Test discover and register when endpoints change."""
        provider = MCPOAuth2Provider(mock_config)

        # Mock discovery returning changed endpoints
        with patch.object(provider._discoverer, 'discover') as mock_discover:
            mock_discover.return_value = (
                OAuth2Endpoints(
                    authorization_url="https://auth.example.com/authorize",  # type: ignore
                    token_url="https://auth.example.com/token",  # type: ignore
                    registration_url="https://auth.example.com/register",  # type: ignore
                ),
                True)

            with patch.object(provider._registrar, 'register') as mock_register:
                mock_register.return_value = OAuth2Credentials(client_id="test_client_id",
                                                               client_secret="test_client_secret")

                await provider._discover_and_register(response=None)

                # Should call register because endpoints changed
                mock_register.assert_called_once()

    async def test_discover_and_register_with_manual_credentials(self, mock_config):
        """Test discover and register with manual credentials."""
        config = mock_config.model_copy(update={
            'client_id': 'manual_client_id', 'client_secret': 'manual_client_secret'
        })
        provider = MCPOAuth2Provider(config)

        with patch.object(provider._discoverer, 'discover') as mock_discover:
            mock_discover.return_value = (
                OAuth2Endpoints(
                    authorization_url="https://auth.example.com/authorize",  # type: ignore
                    token_url="https://auth.example.com/token",  # type: ignore
                ),
                True)

            await provider._discover_and_register(response=None)

            # Should use manual credentials, not register
            assert provider._cached_credentials is not None
            assert provider._cached_credentials.client_id == 'manual_client_id'
            assert provider._cached_credentials.client_secret == 'manual_client_secret'

    async def test_discover_and_register_without_registration_endpoint(self, mock_config):
        """Test discover and register when no registration endpoint is available."""
        provider = MCPOAuth2Provider(mock_config)

        with patch.object(provider._discoverer, 'discover') as mock_discover:
            mock_discover.return_value = (
                OAuth2Endpoints(
                    authorization_url="https://auth.example.com/authorize",  # type: ignore
                    token_url="https://auth.example.com/token",  # type: ignore
                    registration_url=None,  # No registration endpoint
                ),
                True)

            with patch.object(provider._registrar, 'register') as mock_register:
                mock_register.return_value = OAuth2Credentials(client_id="test_client_id",
                                                               client_secret="test_client_secret")

                await provider._discover_and_register(response=None)

                # Should still call register (it will use fallback URL)
                mock_register.assert_called_once()

    async def test_authenticate_with_user_id_propagation(self, mock_config_with_credentials, mock_endpoints):
        """Test that user_id is properly propagated in auth request."""
        provider = MCPOAuth2Provider(mock_config_with_credentials)

        with patch.object(provider._discoverer, 'discover') as mock_discover:
            mock_discover.return_value = (mock_endpoints, True)

            with patch.object(provider, '_nat_oauth2_authenticate') as mock_flow:
                mock_flow.return_value = AuthResult(credentials=[], token_expires_at=None, raw={})

                # Call with different user_id
                await provider.authenticate(user_id="new_user")

                # Verify the flow was called
                mock_flow.assert_called_once()

    async def test_authenticate_without_user_id_in_request(self, mock_config_with_credentials, mock_endpoints):
        """Test authentication when user_id is not in the original request."""
        provider = MCPOAuth2Provider(mock_config_with_credentials)

        with patch.object(provider._discoverer, 'discover') as mock_discover:
            mock_discover.return_value = (mock_endpoints, True)

            with patch.object(provider, '_nat_oauth2_authenticate') as mock_flow:
                mock_flow.return_value = AuthResult(credentials=[], token_expires_at=None, raw={})

                # Call with user_id
                await provider.authenticate(user_id="test_user")

                # Verify the flow was called
                mock_flow.assert_called_once()

    async def test_authenticate_retry_after_401_clears_auth_code_provider(self,
                                                                          mock_config_with_credentials,
                                                                          mock_endpoints):
        """Test that authentication works correctly with auth code provider."""
        provider = MCPOAuth2Provider(mock_config_with_credentials)

        # Set up a mock auth code provider
        provider._auth_code_provider = AsyncMock()

        with patch.object(provider._discoverer, 'discover') as mock_discover:
            mock_discover.return_value = (mock_endpoints, True)

            with patch.object(provider, '_nat_oauth2_authenticate') as mock_flow:
                mock_flow.return_value = AuthResult(credentials=[], token_expires_at=None, raw={})

                result = await provider.authenticate(user_id="test_user")

                # Verify authentication completed successfully
                assert result.credentials == []
                assert result.token_expires_at is None
                assert result.raw == {}
                mock_flow.assert_called_once()

    async def test_effective_scopes_with_config_scopes(self, mock_config):
        """Test effective scopes when config has scopes."""
        config = mock_config.model_copy(update={'scopes': ['config_scope']})
        provider = MCPOAuth2Provider(config)

        scopes = provider._effective_scopes
        assert scopes == ['config_scope']

    async def test_effective_scopes_with_discovered_scopes(self, mock_config):
        """Test effective scopes when using discovered scopes."""
        provider = MCPOAuth2Provider(mock_config)
        provider._cached_endpoints = OAuth2Endpoints(
            authorization_url="https://auth.example.com/authorize",  # type: ignore
            token_url="https://auth.example.com/token",  # type: ignore
            scopes=['discovered_scope'])

        scopes = provider._effective_scopes
        assert scopes == ['discovered_scope']

    async def test_effective_scopes_config_overrides_discovered(self, mock_config):
        """Test that config scopes override discovered scopes."""
        config = mock_config.model_copy(update={'scopes': ['config_scope']})
        provider = MCPOAuth2Provider(config)
        provider._cached_endpoints = OAuth2Endpoints(
            authorization_url="https://auth.example.com/authorize",  # type: ignore
            token_url="https://auth.example.com/token",  # type: ignore
            scopes=['discovered_scope'])

        scopes = provider._effective_scopes
        assert scopes == ['config_scope']  # Config should take precedence
