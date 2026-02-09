# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""Tests for AuthenticationRef-based token resolver creation in A365 telemetry."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.data_models.authentication import AuthResult, BearerTokenCred, HeaderCred
from nat.data_models.component_ref import AuthenticationRef
from nat.plugins.a365.exceptions import A365AuthenticationError
from nat.plugins.a365.telemetry.register import (
    _create_token_resolver_from_auth_ref,
    _extract_token_from_auth_result,
)


class TestCreateTokenResolverFromAuthRef:
    """Tests for _create_token_resolver_from_auth_ref function."""

    @pytest.fixture
    def mock_auth_provider(self):
        """Create a mock auth provider."""
        provider = Mock()
        provider.authenticate = AsyncMock()
        return provider

    @pytest.fixture
    def mock_builder(self, mock_auth_provider):
        """Create a mock builder."""
        builder = Mock(spec=Builder)
        builder.get_auth_provider = AsyncMock(return_value=mock_auth_provider)
        return builder

    @pytest.fixture
    def mock_context(self):
        """Create a mock context."""
        context = Mock(spec=Context)
        context.user_id = "test_user_123"
        return context

    @pytest.mark.asyncio
    async def test_create_token_resolver_success(self, mock_builder, mock_auth_provider, mock_context):
        """Test successful token resolver creation with BearerTokenCred."""
        from pydantic import SecretStr

        auth_ref = AuthenticationRef("test_auth")
        cred = BearerTokenCred(token=SecretStr("test_token_123"))
        auth_result = Mock(spec=AuthResult)
        auth_result.credentials = [cred]
        auth_result.token_expires_at = None

        mock_auth_provider.authenticate.return_value = auth_result

        with patch("nat.builder.context.Context") as mock_context_class:
            mock_context_class.get.return_value = mock_context

            token_resolver, auth_provider, token_cache = await _create_token_resolver_from_auth_ref(
                auth_ref, mock_builder
            )

            assert token_resolver is not None
            assert callable(token_resolver)
            assert auth_provider == mock_auth_provider
            assert token_cache is not None

            # Test the resolver returns the cached token
            token = token_resolver("agent-123", "tenant-456")
            assert token == "test_token_123"

            # Verify auth provider was called with user_id
            mock_auth_provider.authenticate.assert_called_once_with(user_id="test_user_123")

    @pytest.mark.asyncio
    async def test_create_token_resolver_with_expiration(self, mock_builder, mock_auth_provider, mock_context):
        """Test token resolver creation with token expiration."""
        from datetime import datetime, timedelta, timezone
        from pydantic import SecretStr

        auth_ref = AuthenticationRef("test_auth")
        cred = BearerTokenCred(token=SecretStr("test_token_123"))
        auth_result = Mock(spec=AuthResult)
        auth_result.credentials = [cred]
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        auth_result.token_expires_at = expires_at

        mock_auth_provider.authenticate.return_value = auth_result

        with patch("nat.builder.context.Context") as mock_context_class:
            mock_context_class.get.return_value = mock_context

            token_resolver, _, token_cache = await _create_token_resolver_from_auth_ref(
                auth_ref, mock_builder
            )

            # Token should be valid
            token = token_resolver("agent-123", "tenant-456")
            assert token == "test_token_123"

            # Cache should have expiration
            assert token_cache.is_expiring_soon() is False

    @pytest.mark.asyncio
    async def test_create_token_resolver_no_credentials(self, mock_builder, mock_auth_provider, mock_context):
        """Test that error is raised when auth provider returns no credentials."""
        auth_ref = AuthenticationRef("test_auth")
        auth_result = Mock(spec=AuthResult)
        auth_result.credentials = []

        mock_auth_provider.authenticate.return_value = auth_result

        with patch("nat.builder.context.Context") as mock_context_class:
            mock_context_class.get.return_value = mock_context

            with pytest.raises(A365AuthenticationError, match="No credentials available"):
                await _create_token_resolver_from_auth_ref(auth_ref, mock_builder)

    @pytest.mark.asyncio
    async def test_token_resolver_returns_none_when_expired(self, mock_builder, mock_auth_provider, mock_context):
        """Test that token resolver returns None when token is expired."""
        from datetime import datetime, timedelta, timezone
        from pydantic import SecretStr

        auth_ref = AuthenticationRef("test_auth")
        cred = BearerTokenCred(token=SecretStr("test_token_123"))
        auth_result = Mock(spec=AuthResult)
        auth_result.credentials = [cred]
        # Token expired 1 minute ago
        auth_result.token_expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)

        mock_auth_provider.authenticate.return_value = auth_result

        with patch("nat.builder.context.Context") as mock_context_class:
            mock_context_class.get.return_value = mock_context

            token_resolver, _, _ = await _create_token_resolver_from_auth_ref(auth_ref, mock_builder)

            # Resolver should return None for expired token
            token = token_resolver("agent-123", "tenant-456")
            assert token is None
