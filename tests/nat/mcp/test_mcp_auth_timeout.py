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

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from mcp.client.session import ClientSession
from pydantic import SecretStr

from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.plugins.mcp.auth.auth_provider import MCPOAuth2Provider
from nat.plugins.mcp.auth.auth_provider_config import MCPOAuth2ProviderConfig
from nat.plugins.mcp.client_base import AuthAdapter
from nat.plugins.mcp.client_base import MCPBaseClient


class MockMCPClient(MCPBaseClient):
    """Mock MCP client for testing authentication timeout functionality."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect_call_count = 0
        self.call_tool_side_effect = None

    def connect_to_server(self):  # type: ignore
        """Mock connection."""
        return MockAsyncContextManager(self)


class MockAsyncContextManager:
    """Mock async context manager for testing."""

    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        self.client.connect_call_count += 1
        mock_session = AsyncMock(spec=ClientSession)

        if self.client.call_tool_side_effect:
            mock_session.call_tool.side_effect = self.client.call_tool_side_effect
        else:
            mock_session.call_tool = AsyncMock()

        return mock_session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# ============================================================================
# Configuration Tests
# ============================================================================


async def test_auth_flow_timeout_configuration():
    """Test that auth_flow_timeout parameter is properly configured."""
    auth_timeout = timedelta(seconds=300)
    tool_timeout = timedelta(seconds=60)

    client = MockMCPClient(transport="streamable-http", tool_call_timeout=tool_timeout, auth_flow_timeout=auth_timeout)

    assert client._tool_call_timeout == tool_timeout
    assert client._auth_flow_timeout == auth_timeout


async def test_default_timeout_values():
    """Test that default timeout values are set correctly."""
    client = MockMCPClient(transport="streamable-http")

    assert client._tool_call_timeout == timedelta(seconds=60)
    assert client._auth_flow_timeout == timedelta(seconds=300)


# ============================================================================
# _has_cached_auth_token Tests
# ============================================================================


async def test_has_cached_auth_token_no_auth_provider():
    """Test _has_cached_auth_token returns True when no auth provider is configured."""
    client = MockMCPClient(transport="streamable-http")

    async with client:
        has_token = await client._has_cached_auth_token()
        assert has_token is True


async def test_has_cached_auth_token_with_valid_token():
    """Test _has_cached_auth_token returns True when valid token is cached."""
    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    # Create mock OAuth2 provider with cached token
    mock_oauth_provider = MagicMock()
    mock_auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("test_token"))],
                                  token_expires_at=None,
                                  raw={})
    mock_oauth_provider._authenticated_tokens = {"user1": mock_auth_result}
    auth_provider._auth_code_provider = mock_oauth_provider

    client = MockMCPClient(transport="streamable-http", auth_provider=auth_provider)

    async with client:
        has_token = await client._has_cached_auth_token()
        assert has_token is True


async def test_has_cached_auth_token_with_expired_token():
    """Test _has_cached_auth_token returns False when token is expired."""
    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    # Create mock OAuth2 provider with expired token
    mock_oauth_provider = MagicMock()
    expired_time = datetime.now(UTC)  # Already expired
    mock_auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("test_token"))],
                                  token_expires_at=expired_time,
                                  raw={})
    mock_oauth_provider._authenticated_tokens = {"user1": mock_auth_result}
    auth_provider._auth_code_provider = mock_oauth_provider

    client = MockMCPClient(transport="streamable-http", auth_provider=auth_provider)

    async with client:
        has_token = await client._has_cached_auth_token()
        assert has_token is False


async def test_has_cached_auth_token_no_cached_tokens():
    """Test _has_cached_auth_token returns False when no tokens are cached."""
    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    # Create mock OAuth2 provider with no cached tokens
    mock_oauth_provider = MagicMock()
    mock_oauth_provider._authenticated_tokens = {}
    auth_provider._auth_code_provider = mock_oauth_provider

    client = MockMCPClient(transport="streamable-http", auth_provider=auth_provider)

    async with client:
        has_token = await client._has_cached_auth_token()
        assert has_token is False


async def test_has_cached_auth_token_multiple_tokens_one_valid():
    """Test _has_cached_auth_token returns True when at least one token is valid."""
    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    # Create mock OAuth2 provider with one expired and one valid token
    mock_oauth_provider = MagicMock()
    expired_time = datetime.now(UTC)
    expired_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("expired_token"))],
                                token_expires_at=expired_time,
                                raw={})
    valid_result = AuthResult(
        credentials=[BearerTokenCred(token=SecretStr("valid_token"))],
        token_expires_at=None,  # No expiration
        raw={})
    mock_oauth_provider._authenticated_tokens = {"user1": expired_result, "user2": valid_result}
    auth_provider._auth_code_provider = mock_oauth_provider

    client = MockMCPClient(transport="streamable-http", auth_provider=auth_provider)

    async with client:
        has_token = await client._has_cached_auth_token()
        assert has_token is True


# ============================================================================
# _get_tool_call_timeout Tests
# ============================================================================


async def test_get_tool_call_timeout_no_auth_provider():
    """Test _get_tool_call_timeout returns normal timeout when no auth provider."""
    tool_timeout = timedelta(seconds=10)
    auth_timeout = timedelta(seconds=300)

    client = MockMCPClient(transport="streamable-http", tool_call_timeout=tool_timeout, auth_flow_timeout=auth_timeout)

    async with client:
        timeout = await client._get_tool_call_timeout()
        assert timeout == tool_timeout


async def test_get_tool_call_timeout_with_cached_token():
    """Test _get_tool_call_timeout returns normal timeout when token is cached."""
    tool_timeout = timedelta(seconds=10)
    auth_timeout = timedelta(seconds=300)

    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    # Mock cached token
    mock_oauth_provider = MagicMock()
    mock_auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("test_token"))],
                                  token_expires_at=None,
                                  raw={})
    mock_oauth_provider._authenticated_tokens = {"user1": mock_auth_result}
    auth_provider._auth_code_provider = mock_oauth_provider

    client = MockMCPClient(transport="streamable-http",
                           auth_provider=auth_provider,
                           tool_call_timeout=tool_timeout,
                           auth_flow_timeout=auth_timeout)

    async with client:
        timeout = await client._get_tool_call_timeout()
        assert timeout == tool_timeout  # Should use normal timeout


async def test_get_tool_call_timeout_without_cached_token():
    """Test _get_tool_call_timeout returns auth timeout when no token is cached."""
    tool_timeout = timedelta(seconds=10)
    auth_timeout = timedelta(seconds=300)

    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    # Mock no cached tokens
    mock_oauth_provider = MagicMock()
    mock_oauth_provider._authenticated_tokens = {}
    auth_provider._auth_code_provider = mock_oauth_provider

    client = MockMCPClient(transport="streamable-http",
                           auth_provider=auth_provider,
                           tool_call_timeout=tool_timeout,
                           auth_flow_timeout=auth_timeout)

    async with client:
        timeout = await client._get_tool_call_timeout()
        assert timeout == auth_timeout  # Should use extended auth timeout


# ============================================================================
# AuthAdapter Tests
# ============================================================================


async def test_auth_adapter_tracks_authentication_state():
    """Test that AuthAdapter properly tracks authentication state."""
    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)
    adapter = AuthAdapter(auth_provider)

    # Initially not authenticating
    assert adapter.is_authenticating is False


async def test_auth_adapter_initializes_with_auth_provider():
    """Test that AuthAdapter is properly initialized with auth provider."""
    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)
    adapter = AuthAdapter(auth_provider)

    assert adapter.auth_provider == auth_provider
    assert adapter.is_authenticating is False


# ============================================================================
# _with_reconnect During Authentication Tests
# ============================================================================


async def test_with_reconnect_timeout_during_auth_no_reconnect():
    """Test that _with_reconnect doesn't reconnect during authentication timeout."""
    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    client = MockMCPClient(transport="streamable-http",
                           auth_provider=auth_provider,
                           reconnect_enabled=True,
                           reconnect_max_attempts=2)

    async with client:
        # Simulate authentication in progress
        client._httpx_auth.is_authenticating = True

        reconnect_called = False

        async def mock_reconnect():
            nonlocal reconnect_called
            reconnect_called = True

        client._reconnect = mock_reconnect

        async def timeout_operation():
            raise TimeoutError("Auth timeout")

        # Should raise RuntimeError about auth timeout, not reconnect
        with pytest.raises(RuntimeError, match="Authentication timed out"):
            await client._with_reconnect(timeout_operation)

        # Verify reconnect was NOT called
        assert reconnect_called is False


async def test_with_reconnect_error_during_auth_no_reconnect():
    """Test that _with_reconnect doesn't reconnect during authentication errors."""
    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    client = MockMCPClient(transport="streamable-http",
                           auth_provider=auth_provider,
                           reconnect_enabled=True,
                           reconnect_max_attempts=2)

    async with client:
        # Simulate authentication in progress
        client._httpx_auth.is_authenticating = True

        reconnect_called = False

        async def mock_reconnect():
            nonlocal reconnect_called
            reconnect_called = True

        client._reconnect = mock_reconnect

        async def error_operation():
            raise ValueError("Auth error")

        # Should raise the original error, not reconnect
        with pytest.raises(ValueError, match="Auth error"):
            await client._with_reconnect(error_operation)

        # Verify reconnect was NOT called
        assert reconnect_called is False


async def test_with_reconnect_timeout_not_during_auth_does_reconnect():
    """Test that _with_reconnect does reconnect for timeouts when not authenticating."""
    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    client = MockMCPClient(transport="streamable-http",
                           auth_provider=auth_provider,
                           reconnect_enabled=True,
                           reconnect_max_attempts=2,
                           reconnect_initial_backoff=0.01)

    async with client:
        # NOT authenticating
        client._httpx_auth.is_authenticating = False

        reconnect_called = False
        call_count = 0

        async def mock_reconnect():
            nonlocal reconnect_called
            reconnect_called = True

        client._reconnect = mock_reconnect

        async def timeout_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Network timeout")
            return "success"

        # Should reconnect and succeed
        result = await client._with_reconnect(timeout_operation)
        assert result == "success"
        assert reconnect_called is True
        assert call_count == 2  # Failed once, then succeeded


async def test_with_reconnect_error_not_during_auth_does_reconnect():
    """Test that _with_reconnect does reconnect for non-timeout errors when not authenticating."""
    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    client = MockMCPClient(transport="streamable-http",
                           auth_provider=auth_provider,
                           reconnect_enabled=True,
                           reconnect_max_attempts=2,
                           reconnect_initial_backoff=0.01)

    async with client:
        # NOT authenticating
        client._httpx_auth.is_authenticating = False

        reconnect_called = False
        call_count = 0

        async def mock_reconnect():
            nonlocal reconnect_called
            reconnect_called = True

        client._reconnect = mock_reconnect

        async def error_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return "success"

        # Should reconnect and succeed
        result = await client._with_reconnect(error_operation)
        assert result == "success"
        assert reconnect_called is True
        assert call_count == 2  # Failed once, then succeeded


# ============================================================================
# Integration Tests
# ============================================================================


async def test_call_tool_uses_correct_timeout_with_cached_token():
    """Test that call_tool uses appropriate timeout based on auth state."""
    tool_timeout = timedelta(seconds=10)
    auth_timeout = timedelta(seconds=300)

    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    # Mock cached token
    mock_oauth_provider = MagicMock()
    mock_auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("test_token"))],
                                  token_expires_at=None,
                                  raw={})
    mock_oauth_provider._authenticated_tokens = {"user1": mock_auth_result}
    auth_provider._auth_code_provider = mock_oauth_provider

    client = MockMCPClient(transport="streamable-http",
                           auth_provider=auth_provider,
                           tool_call_timeout=tool_timeout,
                           auth_flow_timeout=auth_timeout)

    call_args = []

    async def mock_call_tool(*args, **kwargs):
        call_args.append((args, kwargs))
        return MagicMock(content=[])

    client.call_tool_side_effect = mock_call_tool

    async with client:
        await client.call_tool("test_tool", {"arg": "value"})

        # Should use normal timeout (not auth timeout) since token is cached
        assert len(call_args) == 1
        args, kwargs = call_args[0]
        assert kwargs.get("read_timeout_seconds") == tool_timeout


async def test_call_tool_uses_extended_timeout_without_token():
    """Test that call_tool uses extended timeout when no token is cached."""
    tool_timeout = timedelta(seconds=10)
    auth_timeout = timedelta(seconds=300)

    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    # Mock no cached tokens
    mock_oauth_provider = MagicMock()
    mock_oauth_provider._authenticated_tokens = {}
    auth_provider._auth_code_provider = mock_oauth_provider

    client = MockMCPClient(transport="streamable-http",
                           auth_provider=auth_provider,
                           tool_call_timeout=tool_timeout,
                           auth_flow_timeout=auth_timeout)

    call_args = []

    async def mock_call_tool(*args, **kwargs):
        call_args.append((args, kwargs))
        return MagicMock(content=[])

    client.call_tool_side_effect = mock_call_tool

    async with client:
        await client.call_tool("test_tool", {"arg": "value"})

        # Should use extended auth timeout since no token is cached
        assert len(call_args) == 1
        args, kwargs = call_args[0]
        assert kwargs.get("read_timeout_seconds") == auth_timeout


async def test_timeout_switches_after_authentication():
    """Test that timeout switches from auth to normal after authentication completes."""
    tool_timeout = timedelta(seconds=10)
    auth_timeout = timedelta(seconds=300)

    auth_config = MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_id="test_client",
        client_secret="test_secret")
    auth_provider = MCPOAuth2Provider(auth_config)

    # Start with no tokens
    mock_oauth_provider = MagicMock()
    mock_oauth_provider._authenticated_tokens = {}
    auth_provider._auth_code_provider = mock_oauth_provider

    client = MockMCPClient(transport="streamable-http",
                           auth_provider=auth_provider,
                           tool_call_timeout=tool_timeout,
                           auth_flow_timeout=auth_timeout)

    async with client:
        # First call - no token, should use auth timeout
        timeout1 = await client._get_tool_call_timeout()
        assert timeout1 == auth_timeout

        # Simulate authentication completing
        mock_auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("test_token"))],
                                      token_expires_at=None,
                                      raw={})
        mock_oauth_provider._authenticated_tokens = {"user1": mock_auth_result}

        # Second call - token cached, should use normal timeout
        timeout2 = await client._get_tool_call_timeout()
        assert timeout2 == tool_timeout
