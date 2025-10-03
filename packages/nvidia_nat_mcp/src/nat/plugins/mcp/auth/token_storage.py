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

import hashlib
import json
import logging
from abc import ABC
from abc import abstractmethod

from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BasicAuthCred
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.authentication import CookieCred
from nat.data_models.authentication import HeaderCred
from nat.data_models.authentication import QueryCred
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem

logger = logging.getLogger(__name__)


class TokenStorageBase(ABC):
    """
    Abstract base class for token storage implementations.

    Token storage implementations handle the secure persistence of authentication
    tokens for MCP OAuth2 flows. Implementations can use various backends such as
    object stores, databases, or in-memory storage.
    """

    @abstractmethod
    async def store(self, user_id: str, auth_result: AuthResult) -> None:
        """
        Store an authentication result for a user.

        Args:
            user_id: The unique identifier for the user
            auth_result: The authentication result to store
        """
        pass

    @abstractmethod
    async def retrieve(self, user_id: str) -> AuthResult | None:
        """
        Retrieve an authentication result for a user.

        Args:
            user_id: The unique identifier for the user

        Returns:
            The authentication result if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete(self, user_id: str) -> None:
        """
        Delete an authentication result for a user.

        Args:
            user_id: The unique identifier for the user
        """
        pass

    @abstractmethod
    async def clear_all(self) -> None:
        """
        Clear all stored authentication results.
        """
        pass


class ObjectStoreTokenStorage(TokenStorageBase):
    """
    Token storage implementation backed by a NeMo Agent toolkit object store.

    This implementation uses the object store infrastructure to persist tokens,
    which provides encryption at rest, access controls, and persistence across
    restarts when using backends like S3, MySQL, or Redis.
    """

    def __init__(self, object_store: ObjectStore):
        """
        Initialize the object store token storage.

        Args:
            object_store: The object store instance to use for token persistence
        """
        self._object_store = object_store

    def _get_key(self, user_id: str) -> str:
        """
        Generate the object store key for a user's token.

        Uses SHA256 hash to ensure the key is S3-compatible and doesn't
        contain special characters like "://" that are invalid in object keys.

        Args:
            user_id: The user identifier

        Returns:
            The object store key
        """
        # Hash the user_id to create an S3-safe key
        user_hash = hashlib.sha256(user_id.encode('utf-8')).hexdigest()
        return f"tokens/{user_hash}"

    async def store(self, user_id: str, auth_result: AuthResult) -> None:
        """
        Store an authentication result in the object store.

        Args:
            user_id: The unique identifier for the user
            auth_result: The authentication result to store
        """
        key = self._get_key(user_id)

        # Serialize the AuthResult to JSON with secrets exposed
        # SecretStr values are masked by default, so we need to expose them manually
        # Create a serializable dict with exposed secrets
        auth_dict = auth_result.model_dump(mode='json')
        # Manually expose SecretStr values in credentials
        for i, cred_obj in enumerate(auth_result.credentials):
            if isinstance(cred_obj, BearerTokenCred):
                auth_dict['credentials'][i]['token'] = cred_obj.token.get_secret_value()
            elif isinstance(cred_obj, BasicAuthCred):
                auth_dict['credentials'][i]['username'] = cred_obj.username.get_secret_value()
                auth_dict['credentials'][i]['password'] = cred_obj.password.get_secret_value()
            elif isinstance(cred_obj, HeaderCred | QueryCred | CookieCred):
                auth_dict['credentials'][i]['value'] = cred_obj.value.get_secret_value()

        data = json.dumps(auth_dict).encode('utf-8')

        # Prepare metadata
        metadata = {}
        if auth_result.token_expires_at:
            metadata["expires_at"] = auth_result.token_expires_at.isoformat()

        # Create the object store item
        item = ObjectStoreItem(data=data, content_type="application/json", metadata=metadata if metadata else None)

        # Store using upsert to handle both new and existing tokens
        await self._object_store.upsert_object(key, item)

    async def retrieve(self, user_id: str) -> AuthResult | None:
        """
        Retrieve an authentication result from the object store.

        Args:
            user_id: The unique identifier for the user

        Returns:
            The authentication result if found, None otherwise
        """
        key = self._get_key(user_id)

        try:
            item = await self._object_store.get_object(key)
            # Deserialize the AuthResult from JSON
            auth_result = AuthResult.model_validate_json(item.data)
            return auth_result
        except NoSuchKeyError:
            return None
        except Exception as e:
            logger.error(f"Error deserializing token for user {user_id}: {e}", exc_info=True)
            return None

    async def delete(self, user_id: str) -> None:
        """
        Delete an authentication result from the object store.

        Args:
            user_id: The unique identifier for the user
        """
        key = self._get_key(user_id)

        try:
            await self._object_store.delete_object(key)
        except NoSuchKeyError:
            # Token doesn't exist, which is fine for delete operations
            pass

    async def clear_all(self) -> None:
        """
        Clear all stored authentication results.

        Note: This implementation does not support clearing all tokens as the
        object store interface doesn't provide a list operation. Individual
        tokens must be deleted explicitly.
        """
        logger.warning("clear_all() is not supported for ObjectStoreTokenStorage")


class InMemoryTokenStorage(TokenStorageBase):
    """
    In-memory token storage using NeMo Agent toolkit's built-in object store.

    This implementation uses the in-memory object store for token persistence,
    which provides a secure default option that doesn't require external storage
    configuration. Tokens are stored in memory and cleared when the process exits.
    """

    def __init__(self):
        """
        Initialize the in-memory token storage.
        """
        from nat.object_store.in_memory_object_store import InMemoryObjectStore

        # Create a dedicated in-memory object store for tokens
        self._object_store = InMemoryObjectStore()

        # Wrap with ObjectStoreTokenStorage for the actual implementation
        self._storage = ObjectStoreTokenStorage(self._object_store)
        logger.debug("Initialized in-memory token storage")

    async def store(self, user_id: str, auth_result: AuthResult) -> None:
        """
        Store an authentication result in memory.

        Args:
            user_id: The unique identifier for the user
            auth_result: The authentication result to store
        """
        await self._storage.store(user_id, auth_result)

    async def retrieve(self, user_id: str) -> AuthResult | None:
        """
        Retrieve an authentication result from memory.

        Args:
            user_id: The unique identifier for the user

        Returns:
            The authentication result if found, None otherwise
        """
        return await self._storage.retrieve(user_id)

    async def delete(self, user_id: str) -> None:
        """
        Delete an authentication result from memory.

        Args:
            user_id: The unique identifier for the user
        """
        await self._storage.delete(user_id)

    async def clear_all(self) -> None:
        """
        Clear all stored authentication results from memory.
        """
        # For in-memory storage, we can access the internal storage
        self._object_store._store.clear()
