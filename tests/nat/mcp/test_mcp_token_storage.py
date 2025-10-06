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
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.in_memory_object_store import InMemoryObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.plugins.mcp.auth.auth_provider import MCPOAuth2Provider
from nat.plugins.mcp.auth.auth_provider import OAuth2Credentials
from nat.plugins.mcp.auth.auth_provider import OAuth2Endpoints
from nat.plugins.mcp.auth.token_storage import InMemoryTokenStorage
from nat.plugins.mcp.auth.token_storage import ObjectStoreTokenStorage

# --------------------------------------------------------------------------- #
# Test Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_auth_result() -> AuthResult:
    """Create a sample AuthResult for testing."""
    return AuthResult(credentials=[BearerTokenCred(token=SecretStr("test_token_12345"))],
                      token_expires_at=datetime.now(UTC) + timedelta(hours=1),
                      raw={
                          "access_token": "test_token_12345",
                          "refresh_token": "refresh_token_67890",
                          "expires_at": 1234567890
                      })


@pytest.fixture
def expired_auth_result() -> AuthResult:
    """Create an expired AuthResult for testing."""
    return AuthResult(credentials=[BearerTokenCred(token=SecretStr("expired_token"))],
                      token_expires_at=datetime.now(UTC) - timedelta(hours=1),
                      raw={"access_token": "expired_token"})


@pytest.fixture
def mock_object_store():
    """Create a mock object store for testing."""
    mock = AsyncMock()
    mock.upsert_object = AsyncMock()
    mock.get_object = AsyncMock()
    mock.delete_object = AsyncMock()
    return mock


@pytest.fixture
def mock_config():
    """Create a mock MCP OAuth2 provider config for testing."""
    from nat.plugins.mcp.auth.auth_provider_config import MCPOAuth2ProviderConfig
    return MCPOAuth2ProviderConfig(
        server_url="https://example.com/mcp",  # type: ignore
        redirect_uri="https://example.com/callback",  # type: ignore
        client_name="Test Client",
        enable_dynamic_registration=True,
    )


# --------------------------------------------------------------------------- #
# ObjectStoreTokenStorage Tests
# --------------------------------------------------------------------------- #


class TestObjectStoreTokenStorage:
    """Test the ObjectStoreTokenStorage class."""

    async def test_store_and_retrieve(self, mock_object_store, sample_auth_result):
        """Test storing and retrieving a token."""
        storage = ObjectStoreTokenStorage(mock_object_store)
        user_id = "test_user"

        # Store the token
        await storage.store(user_id, sample_auth_result)

        # Verify upsert was called
        assert mock_object_store.upsert_object.called
        call_args = mock_object_store.upsert_object.call_args
        key, item = call_args[0]

        # Verify key is hashed
        assert key.startswith("tokens/")
        assert len(key) > 20  # SHA256 hash should be long

        # Verify item structure
        assert isinstance(item, ObjectStoreItem)
        assert item.content_type == "application/json"
        assert item.metadata is not None
        assert "expires_at" in item.metadata

        # Setup mock retrieval
        mock_object_store.get_object.return_value = item

        # Retrieve the token
        retrieved = await storage.retrieve(user_id)

        # Verify the retrieved token
        assert retrieved is not None
        assert len(retrieved.credentials) == 1
        assert isinstance(retrieved.credentials[0], BearerTokenCred)
        assert retrieved.credentials[0].token.get_secret_value() == "test_token_12345"  # type: ignore[union-attr]

    async def test_retrieve_nonexistent_token(self, mock_object_store):
        """Test retrieving a token that doesn't exist."""
        storage = ObjectStoreTokenStorage(mock_object_store)
        mock_object_store.get_object.side_effect = NoSuchKeyError("test_key")

        result = await storage.retrieve("nonexistent_user")

        assert result is None

    async def test_delete_token(self, mock_object_store):
        """Test deleting a token."""
        storage = ObjectStoreTokenStorage(mock_object_store)
        user_id = "test_user"

        await storage.delete(user_id)

        # Verify delete was called with hashed key
        assert mock_object_store.delete_object.called
        call_args = mock_object_store.delete_object.call_args
        key = call_args[0][0]
        assert key.startswith("tokens/")

    async def test_delete_nonexistent_token(self, mock_object_store):
        """Test deleting a token that doesn't exist (should not raise)."""
        storage = ObjectStoreTokenStorage(mock_object_store)
        mock_object_store.delete_object.side_effect = NoSuchKeyError("test_key")

        # Should not raise an exception
        await storage.delete("nonexistent_user")

    async def test_key_hashing_consistency(self, mock_object_store, sample_auth_result):
        """Test that the same user_id always produces the same hashed key."""
        storage = ObjectStoreTokenStorage(mock_object_store)
        user_id = "test_user@example.com"

        # Store twice
        await storage.store(user_id, sample_auth_result)
        first_key = mock_object_store.upsert_object.call_args[0][0]

        await storage.store(user_id, sample_auth_result)
        second_key = mock_object_store.upsert_object.call_args[0][0]

        # Keys should be identical
        assert first_key == second_key

    async def test_secret_str_serialization(self, mock_object_store, sample_auth_result):
        """Test that SecretStr values are properly serialized and deserialized."""
        storage = ObjectStoreTokenStorage(mock_object_store)
        user_id = "test_user"

        # Store the token
        await storage.store(user_id, sample_auth_result)

        # Get the stored item
        call_args = mock_object_store.upsert_object.call_args
        stored_item = call_args[0][1]

        # Verify the data contains the actual token value, not masked
        data_str = stored_item.data.decode('utf-8')
        assert "test_token_12345" in data_str
        assert "**********" not in data_str  # Should not be masked

        # Setup retrieval
        mock_object_store.get_object.return_value = stored_item

        # Retrieve and verify
        retrieved = await storage.retrieve(user_id)
        assert retrieved.credentials[0].token.get_secret_value() == "test_token_12345"  # type: ignore[union-attr]

    async def test_clear_all_not_supported(self, mock_object_store):
        """Test that clear_all logs a warning (not supported for generic object stores)."""
        storage = ObjectStoreTokenStorage(mock_object_store)

        # Should complete without error but log warning
        await storage.clear_all()

        # No object store operations should be called
        assert not mock_object_store.delete_object.called


# --------------------------------------------------------------------------- #
# InMemoryTokenStorage Tests
# --------------------------------------------------------------------------- #


class TestInMemoryTokenStorage:
    """Test the InMemoryTokenStorage class."""

    async def test_store_and_retrieve(self, sample_auth_result):
        """Test storing and retrieving a token in memory."""
        storage = InMemoryTokenStorage()
        user_id = "test_user"

        # Store the token
        await storage.store(user_id, sample_auth_result)

        # Retrieve the token
        retrieved = await storage.retrieve(user_id)

        # Verify the retrieved token
        assert retrieved is not None
        assert len(retrieved.credentials) == 1
        assert isinstance(retrieved.credentials[0], BearerTokenCred)
        assert retrieved.credentials[0].token.get_secret_value() == "test_token_12345"  # type: ignore[union-attr]

    async def test_retrieve_nonexistent_token(self):
        """Test retrieving a token that doesn't exist."""
        storage = InMemoryTokenStorage()

        result = await storage.retrieve("nonexistent_user")

        assert result is None

    async def test_delete_token(self, sample_auth_result):
        """Test deleting a token."""
        storage = InMemoryTokenStorage()
        user_id = "test_user"

        # Store then delete
        await storage.store(user_id, sample_auth_result)
        await storage.delete(user_id)

        # Verify token is gone
        result = await storage.retrieve(user_id)
        assert result is None

    async def test_delete_nonexistent_token(self):
        """Test deleting a token that doesn't exist (should not raise)."""
        storage = InMemoryTokenStorage()

        # Should not raise an exception
        await storage.delete("nonexistent_user")

    async def test_clear_all(self, sample_auth_result):
        """Test clearing all stored tokens."""
        storage = InMemoryTokenStorage()

        # Store multiple tokens
        await storage.store("user1", sample_auth_result)
        await storage.store("user2", sample_auth_result)

        # Clear all
        await storage.clear_all()

        # Verify all tokens are gone
        assert await storage.retrieve("user1") is None
        assert await storage.retrieve("user2") is None

    async def test_multiple_users(self, sample_auth_result):
        """Test storing tokens for multiple users."""
        storage = InMemoryTokenStorage()

        # Create different auth results
        auth1 = AuthResult(credentials=[BearerTokenCred(token=SecretStr("token1"))], token_expires_at=None, raw={})
        auth2 = AuthResult(credentials=[BearerTokenCred(token=SecretStr("token2"))], token_expires_at=None, raw={})

        # Store for different users
        await storage.store("user1", auth1)
        await storage.store("user2", auth2)

        # Retrieve and verify isolation
        retrieved1 = await storage.retrieve("user1")
        retrieved2 = await storage.retrieve("user2")

        assert retrieved1.credentials[0].token.get_secret_value() == "token1"  # type: ignore[union-attr]
        assert retrieved2.credentials[0].token.get_secret_value() == "token2"  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Integration Tests
# --------------------------------------------------------------------------- #


class TestTokenStorageIntegration:
    """Integration tests for token storage with OAuth2 flow."""

    async def test_oauth2_provider_with_in_memory_storage(self, mock_config):
        """Test that MCPOAuth2Provider uses in-memory storage by default."""
        provider = MCPOAuth2Provider(mock_config)

        # Verify in-memory storage is initialized
        assert provider._token_storage is not None
        assert isinstance(provider._token_storage, InMemoryTokenStorage)

    async def test_oauth2_provider_with_object_store_reference(self, mock_config):
        """Test that MCPOAuth2Provider can be configured with an object store reference."""
        # Configure with object store reference
        mock_config.token_storage_object_store = "test_store"

        mock_builder = MagicMock()
        mock_builder.get_object_store_client = AsyncMock(return_value=InMemoryObjectStore())

        provider = MCPOAuth2Provider(mock_config, builder=mock_builder)

        # Verify object store name is stored
        assert provider._token_storage_object_store_name == "test_store"
        assert provider._token_storage is None  # Not resolved yet

    async def test_token_storage_lazy_resolution(self, mock_config, sample_auth_result):
        """Test that object store is lazily resolved during authentication."""
        mock_config.token_storage_object_store = "test_store"

        mock_builder = MagicMock()
        mock_object_store = InMemoryObjectStore()
        mock_builder.get_object_store_client = AsyncMock(return_value=mock_object_store)

        provider = MCPOAuth2Provider(mock_config, builder=mock_builder)

        # Mock the cached endpoints and credentials to allow authentication
        provider._cached_endpoints = OAuth2Endpoints(
            authorization_url="https://auth.example.com/authorize",  # type: ignore
            token_url="https://auth.example.com/token",  # type: ignore
        )
        provider._cached_credentials = OAuth2Credentials(client_id="test", client_secret="secret")

        # Trigger authentication which should resolve the object store
        with patch('nat.authentication.oauth2.oauth2_auth_code_flow_provider.OAuth2AuthCodeFlowProvider'
                   ) as mock_provider_class:
            mock_instance = AsyncMock()
            mock_instance.authenticate = AsyncMock(return_value=sample_auth_result)
            mock_instance._set_custom_auth_callback = MagicMock()
            mock_provider_class.return_value = mock_instance

            await provider._nat_oauth2_authenticate(user_id="test_user")

            # Verify object store was resolved
            assert provider._token_storage is not None
            assert isinstance(provider._token_storage, ObjectStoreTokenStorage)
            assert mock_builder.get_object_store_client.called

    async def test_token_persistence_across_provider_instances(self):
        """Test that tokens stored in object store can be retrieved by different provider instances."""
        # Create a shared object store
        object_store = InMemoryObjectStore()
        storage1 = ObjectStoreTokenStorage(object_store)
        storage2 = ObjectStoreTokenStorage(object_store)

        # Create and store auth result with first storage
        auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("persistent_token"))],
                                 token_expires_at=None,
                                 raw={})

        await storage1.store("shared_user", auth_result)

        # Retrieve with second storage instance
        retrieved = await storage2.retrieve("shared_user")

        # Verify token was persisted and retrieved
        assert retrieved is not None
        assert retrieved.credentials[0].token.get_secret_value() == "persistent_token"  # type: ignore[union-attr]

    async def test_url_user_id_compatibility(self, mock_object_store):
        """Test that URL-based user IDs are properly hashed to S3-safe keys."""
        storage = ObjectStoreTokenStorage(mock_object_store)
        url_user_id = "https://example.com/mcp/server"

        auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("token"))], token_expires_at=None, raw={})

        await storage.store(url_user_id, auth_result)

        # Verify the key doesn't contain invalid characters
        call_args = mock_object_store.upsert_object.call_args
        key = call_args[0][0]

        # Key should not contain ://, ?, &, or other invalid S3 characters
        assert "://" not in key
        assert "?" not in key
        assert "&" not in key
        # Key should be in format tokens/{hash}
        assert key.startswith("tokens/")
        assert len(key.split("/")[1]) == 64  # SHA256 produces 64 hex characters
