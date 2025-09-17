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

import asyncio
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from pydantic import HttpUrl

from nat.data_models.authentication import AuthReason
from nat.data_models.authentication import AuthRequest
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
        server_url=HttpUrl("https://example.com/mcp"),
        redirect_uri=HttpUrl("https://example.com/callback"),
        client_name="Test Client",
        enable_dynamic_registration=True,
    )


@pytest.fixture
def mock_config_with_credentials() -> MCPOAuth2ProviderConfig:
    """Create a mock config with pre-registered credentials."""
    return MCPOAuth2ProviderConfig(
        server_url=HttpUrl("https://example.com/mcp"),
        redirect_uri=HttpUrl("https://example.com/callback"),
        client_id="test_client_id",
        client_secret="test_client_secret",
        client_name="Test Client",
        enable_dynamic_registration=False,
    )


@pytest.fixture
def mock_endpoints() -> OAuth2Endpoints:
    """Create mock OAuth2 endpoints for testing."""
    return OAuth2Endpoints(
        authorization_url=HttpUrl("https://auth.example.com/authorize"),
        token_url=HttpUrl("https://auth.example.com/token"),
        registration_url=HttpUrl("https://auth.example.com/register"),
    )


@pytest.fixture
def mock_credentials() -> OAuth2Credentials:
    """Create mock OAuth2 credentials for testing."""
    return OAuth2Credentials(
        client_id="test_client_id",
        client_secret="test_client_secret",
    )


@pytest.fixture
def mock_auth_request() -> AuthRequest:
    """Create a mock auth request for testing."""
    return AuthRequest(
        user_id="test_user",
        reason=AuthReason.NORMAL,
    )


@pytest.fixture
def mock_retry_auth_request() -> AuthRequest:
    """Create a mock retry auth request for testing."""
    return AuthRequest(
        user_id="test_user",
        reason=AuthReason.RETRY_AFTER_401,
        www_authenticate='Bearer realm="api", resource_metadata="https://auth.example.com/.well-known/oauth-protected-resource"',
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
            authorization_url=HttpUrl("https://auth.example.com/authorize"),
            token_url=HttpUrl("https://auth.example.com/token"),
        )
        discoverer._cached_endpoints = cached_endpoints

        # Test normal request returns cached endpoints
        endpoints, changed = await discoverer.discover(
            reason=AuthReason.NORMAL,
            www_authenticate=None
        )

        assert endpoints == cached_endpoints
        assert changed is False

    async def test_discover_with_www_authenticate_hint(self, mock_config):
        """Test discovery using WWW-Authenticate header hint."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        # Mock the protected resource metadata response
        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.aread.return_value = b'{"authorization_servers": ["https://auth.example.com"]}'
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp

            # Mock OAuth metadata response
            with patch.object(discoverer, '_discover_via_issuer_or_base') as mock_discover:
                mock_discover.return_value = OAuth2Endpoints(
                    authorization_url=HttpUrl("https://auth.example.com/authorize"),
                    token_url=HttpUrl("https://auth.example.com/token"),
                    registration_url=HttpUrl("https://auth.example.com/register"),
                )

                endpoints, changed = await discoverer.discover(
                    reason=AuthReason.RETRY_AFTER_401,
                    www_authenticate='Bearer realm="api", resource_metadata="https://auth.example.com/.well-known/oauth-protected-resource"'
                )

                assert endpoints is not None
                assert changed is True

    async def test_discover_fallback_to_server_base(self, mock_config):
        """Test discovery falls back to server base URL when no hint provided."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        with patch.object(discoverer, '_discover_via_issuer_or_base') as mock_discover:
            mock_discover.return_value = OAuth2Endpoints(
                authorization_url=HttpUrl("https://auth.example.com/authorize"),
                token_url=HttpUrl("https://auth.example.com/token"),
            )

            endpoints, changed = await discoverer.discover(
                reason=AuthReason.NORMAL,
                www_authenticate=None
            )

            assert endpoints is not None
            assert changed is True
            mock_discover.assert_called_once_with("https://example.com/mcp")

    def test_extract_from_www_authenticate_header(self, mock_config):
        """Test extracting resource_metadata URL from WWW-Authenticate header."""
        discoverer = DiscoverOAuth2Endpoints(mock_config)

        # Test with double quotes
        url = discoverer._extract_from_www_authenticate_header(
            'Bearer realm="api", resource_metadata="https://auth.example.com/.well-known/oauth-protected-resource"'
        )
        assert url == "https://auth.example.com/.well-known/oauth-protected-resource"

        # Test with single quotes
        url = discoverer._extract_from_www_authenticate_header(
            "Bearer realm='api', resource_metadata='https://auth.example.com/.well-known/oauth-protected-resource'"
        )
        assert url == "https://auth.example.com/.well-known/oauth-protected-resource"

        # Test without quotes
        url = discoverer._extract_from_www_authenticate_header(
            "Bearer realm=api, resource_metadata=https://auth.example.com/.well-known/oauth-protected-resource"
        )
        assert url == "https://auth.example.com/.well-known/oauth-protected-resource"

        # Test case insensitive
        url = discoverer._extract_from_www_authenticate_header(
            "Bearer realm=api, RESOURCE_METADATA=https://auth.example.com/.well-known/oauth-protected-resource"
        )
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
            "https://auth.example.com/.well-known/oauth-authorization-server/api/v1",
            "https://auth.example.com/.well-known/oauth-authorization-server",
            "https://auth.example.com/.well-known/openid-configuration/api/v1",
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
            mock_resp.raise_for_status.return_value = None
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
            authorization_url=HttpUrl("https://auth.example.com/authorize"),
            token_url=HttpUrl("https://auth.example.com/token"),
            registration_url=None,
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status.return_value = None
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
            mock_resp.raise_for_status.return_value = None
            mock_resp.aread.return_value = b'invalid json'
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp

            with pytest.raises(RuntimeError, match="Registration response was not valid"):
                await registrar.register(mock_endpoints, None)

    async def test_register_missing_client_id(self, mock_config, mock_endpoints):
        """Test registration fails when no client_id is returned."""
        registrar = DynamicClientRegistration(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.aread.return_value = b'{"client_secret": "secret", "redirect_uris": ["https://example.com/callback"]}'
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp

            with pytest.raises(RuntimeError):
                await registrar.register(mock_endpoints, None)


# --------------------------------------------------------------------------- #
# MCPOAuth2Provider Tests
# --------------------------------------------------------------------------- #

class TestMCPOAuth2Provider:
    """Test the MCPOAuth2Provider class."""

    async def test_authenticate_normal_request_returns_empty_when_no_provider(self, mock_config):
        """Test that normal requests return empty auth result when no provider is set up."""
        provider = MCPOAuth2Provider(mock_config)

        auth_request = AuthRequest(
            user_id="test_user",
            reason=AuthReason.NORMAL,
        )

        result = await provider.authenticate(user_id="test_user", auth_request=auth_request)

        assert result.credentials == []
        assert result.token_expires_at is None
        assert result.raw == {}

    async def test_authenticate_requires_auth_request(self, mock_config):
        """Test that authenticate requires an auth request."""
        provider = MCPOAuth2Provider(mock_config)

        with pytest.raises(RuntimeError, match="Auth request is required"):
            await provider.authenticate(user_id="test_user", auth_request=None)

    async def test_authenticate_with_manual_credentials(self, mock_config_with_credentials, mock_endpoints, monkeypatch):
        """Test authentication with pre-registered credentials."""
        provider = MCPOAuth2Provider(mock_config_with_credentials)

        # Mock the discovery process
        with patch.object(provider._discoverer, 'discover') as mock_discover:
            mock_discover.return_value = (mock_endpoints, True)

            # Mock the OAuth2 flow
            mock_auth_result = AuthResult(
                credentials=[BearerTokenCred(token="test_token")],
                token_expires_at=None,
                raw={},
            )

            with patch.object(provider, '_perform_oauth2_flow') as mock_flow:
                mock_flow.return_value = mock_auth_result

                auth_request = AuthRequest(
                    user_id="test_user",
                    reason=AuthReason.RETRY_AFTER_401,
                )

                result = await provider.authenticate(user_id="test_user", auth_request=auth_request)

                assert result == mock_auth_result
                mock_discover.assert_called_once()
                mock_flow.assert_called_once()

    async def test_authenticate_with_dynamic_registration(self, mock_config, mock_endpoints, mock_credentials, monkeypatch):
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
                    credentials=[BearerTokenCred(token="test_token")],
                    token_expires_at=None,
                    raw={},
                )

                with patch.object(provider, '_perform_oauth2_flow') as mock_flow:
                    mock_flow.return_value = mock_auth_result

                    auth_request = AuthRequest(
                        user_id="test_user",
                        reason=AuthReason.RETRY_AFTER_401,
                    )

                    result = await provider.authenticate(user_id="test_user", auth_request=auth_request)

                    assert result == mock_auth_result
                    mock_discover.assert_called_once()
                    mock_register.assert_called_once_with(mock_endpoints, None)
                    mock_flow.assert_called_once()

    async def test_authenticate_dynamic_registration_disabled(self, mock_config, mock_endpoints, monkeypatch):
        """Test authentication works when dynamic registration is disabled but valid credentials provided."""
        config = MCPOAuth2ProviderConfig(
            server_url=HttpUrl("https://example.com/mcp"),
            redirect_uri=HttpUrl("https://example.com/callback"),
            client_id="test_client_id",
            client_secret="test_client_secret",
            enable_dynamic_registration=False,
        )
        provider = MCPOAuth2Provider(config)

        # Mock the discovery process and OAuth flow
        with patch.object(provider._discoverer, 'discover') as mock_discover:
            mock_discover.return_value = (mock_endpoints, True)

            with patch.object(provider, '_perform_oauth2_flow') as mock_flow:
                mock_auth_result = AuthResult(credentials=[], token_expires_at=None, raw={})
                mock_flow.return_value = mock_auth_result

                auth_request = AuthRequest(
                    user_id="test_user",
                    reason=AuthReason.RETRY_AFTER_401,
                )

                # Should succeed with manual credentials
                result = await provider.authenticate(user_id="test_user", auth_request=auth_request)

                assert result == mock_auth_result
                mock_discover.assert_called_once()
                mock_flow.assert_called_once()

    async def test_effective_scopes_uses_config_scopes(self, mock_config):
        """Test that effective scopes uses config scopes when provided."""
        config = MCPOAuth2ProviderConfig(
            server_url=HttpUrl("https://example.com/mcp"),
            redirect_uri=HttpUrl("https://example.com/callback"),
            scopes=["read", "write"],
            enable_dynamic_registration=True,
        )
        provider = MCPOAuth2Provider(config)

        scopes = provider._effective_scopes()
        assert scopes == ["read", "write"]

    async def test_effective_scopes_falls_back_to_discovered(self, mock_config):
        """Test that effective scopes falls back to discovered scopes when config scopes not provided."""
        provider = MCPOAuth2Provider(mock_config)

        # Mock discovered scopes
        provider._discoverer._last_oauth_scopes = ["discovered_scope"]

        scopes = provider._effective_scopes()
        assert scopes == ["discovered_scope"]

    async def test_effective_scopes_returns_none_when_none_available(self, mock_config):
        """Test that effective scopes returns None when no scopes available."""
        provider = MCPOAuth2Provider(mock_config)

        scopes = provider._effective_scopes()
        assert scopes is None

    async def test_perform_oauth2_flow_requires_discovery(self, mock_config):
        """Test that OAuth2 flow requires discovery to be completed first."""
        provider = MCPOAuth2Provider(mock_config)

        auth_request = AuthRequest(
            user_id="test_user",
            reason=AuthReason.NORMAL,
        )

        with pytest.raises(RuntimeError, match="OAuth2 flow called before discovery"):
            await provider._perform_oauth2_flow(auth_request=auth_request)


    async def test_perform_oauth2_flow_prevents_retry_after_401(self, mock_config, mock_endpoints, mock_credentials):
        """Test that OAuth2 flow prevents being called for RETRY_AFTER_401."""
        provider = MCPOAuth2Provider(mock_config)

        provider._cached_endpoints = mock_endpoints
        provider._cached_credentials = mock_credentials

        auth_request = AuthRequest(
            user_id="test_user",
            reason=AuthReason.RETRY_AFTER_401,
        )

        with pytest.raises(RuntimeError, match="should not be called for RETRY_AFTER_401"):
            await provider._perform_oauth2_flow(auth_request=auth_request)

    async def test_state_lock_prevents_concurrent_discovery(self, mock_config, mock_endpoints, mock_credentials):
        """Test that state lock prevents concurrent 401 retries from clearing each other's endpoints."""
        provider = MCPOAuth2Provider(mock_config)

        # Create different endpoints for each request to simulate different 401 responses
        endpoints1 = OAuth2Endpoints(
            authorization_url=HttpUrl("https://auth1.example.com/authorize"),
            token_url=HttpUrl("https://auth1.example.com/token"),
            registration_url=HttpUrl("https://auth1.example.com/register")
        )
        endpoints2 = OAuth2Endpoints(
            authorization_url=HttpUrl("https://auth2.example.com/authorize"),
            token_url=HttpUrl("https://auth2.example.com/token"),
            registration_url=HttpUrl("https://auth2.example.com/register")
        )

        async def discover_with_delay(*args, **kwargs):
            # Simulate discovery taking time
            await asyncio.sleep(0.1)
            # Return different endpoints based on which request this is
            if not hasattr(discover_with_delay, 'call_count'):
                discover_with_delay.call_count = 0
            discover_with_delay.call_count += 1

            if discover_with_delay.call_count == 1:
                return endpoints1, True
            else:
                return endpoints2, True

        with patch.object(provider._discoverer, 'discover', side_effect=discover_with_delay):
            with patch.object(provider._registrar, 'register', return_value=mock_credentials):
                with patch.object(provider, '_perform_oauth2_flow') as mock_flow:
                    mock_flow.return_value = AuthResult(credentials=[], token_expires_at=None, raw={})

                    # Create two 401 retry requests
                    auth_request1 = AuthRequest(
                        user_id="test_user1",
                        reason=AuthReason.RETRY_AFTER_401,
                        www_authenticate='Bearer realm="api1"'
                    )
                    auth_request2 = AuthRequest(
                        user_id="test_user2",
                        reason=AuthReason.RETRY_AFTER_401,
                        www_authenticate='Bearer realm="api2"'
                    )

                    # Start two concurrent 401 retry operations
                    task1 = asyncio.create_task(provider._discover_and_register(auth_request1))
                    task2 = asyncio.create_task(provider._discover_and_register(auth_request2))

                    # Both should complete without issues due to locking
                    results = await asyncio.gather(task1, task2)

                    # Verify both operations completed successfully
                    assert len(results) == 2

                    # Verify that the final cached endpoints are consistent
                    # (not overwritten by the second request due to race conditions)
                    assert provider._cached_endpoints is not None
                    assert provider._cached_credentials is not None

                    # The lock should ensure that one request's endpoints don't get
                    # overwritten by the other request's endpoints
                    # We can't predict which one wins, but we should have consistent state
                    final_endpoints = provider._cached_endpoints
                    assert final_endpoints in [endpoints1, endpoints2]
