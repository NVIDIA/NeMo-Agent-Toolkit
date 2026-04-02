# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging
import time
from abc import ABC
from abc import abstractmethod

from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem

logger = logging.getLogger(__name__)

_EXPIRY_BUFFER_SECONDS = 60


class OAuthTokenCacheBase(ABC):
    """Async cache abstraction for WebSocket OAuth tokens.

    A cache key encodes the (session, provider) pair so that different users and
    different OAuth providers are kept isolated. Implementations may be local
    (in-process dict) or distributed (e.g. Redis-backed object store) to support
    multi-replica deployments.
    """

    @abstractmethod
    async def get(self, key: str) -> AuthenticatedContext | None:
        """Return a cached, non-expired token, or None if absent or expired."""

    @abstractmethod
    async def set(self, key: str, ctx: AuthenticatedContext, expires_at: float | None) -> None:
        """Store *ctx* under *key*. Overwrites any existing entry."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove *key* from the cache. A missing key is silently ignored."""


class InMemoryOAuthTokenCache(OAuthTokenCacheBase):
    """In-process dict-backed token cache.

    Suitable for single-process deployments only. All state is lost on restart
    and is not shared across replicas.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[AuthenticatedContext, float | None]] = {}

    async def get(self, key: str) -> AuthenticatedContext | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ctx, expires_at = entry
        if expires_at is not None and time.time() >= expires_at - _EXPIRY_BUFFER_SECONDS:
            del self._store[key]
            return None
        return ctx

    async def set(self, key: str, ctx: AuthenticatedContext, expires_at: float | None) -> None:
        self._store[key] = (ctx, expires_at)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


class ObjectStoreOAuthTokenCache(OAuthTokenCacheBase):
    """Object-store-backed token cache.

    Stores tokens as JSON blobs in a NAT object store (e.g. Redis, S3, MySQL),
    which makes the cache durable and shared across all replicas.
    """

    def __init__(self, object_store: ObjectStore) -> None:
        self._object_store = object_store

    async def get(self, key: str) -> AuthenticatedContext | None:
        try:
            item = await self._object_store.get_object(key)
        except NoSuchKeyError:
            return None
        except Exception:
            logger.exception("Failed to read OAuth token from object store (key=%s)", key)
            return None

        try:
            payload = json.loads(item.data)
            expires_at = payload.get("expires_at")
            if expires_at is not None and time.time() >= float(expires_at) - _EXPIRY_BUFFER_SECONDS:
                await self.delete(key)
                return None
            return AuthenticatedContext.model_validate(payload["ctx"])
        except Exception:
            logger.exception("Failed to deserialize OAuth token from object store (key=%s)", key)
            return None

    async def set(self, key: str, ctx: AuthenticatedContext, expires_at: float | None) -> None:
        try:
            payload = json.dumps({
                "ctx": ctx.model_dump(mode="json"),
                "expires_at": expires_at,
            }).encode("utf-8")
            metadata: dict[str, str] = {}
            if expires_at is not None:
                metadata["expires_at"] = str(expires_at)
            item = ObjectStoreItem(data=payload,
                                   content_type="application/json",
                                   metadata=metadata if metadata else None)
            await self._object_store.upsert_object(key, item)
        except Exception:
            logger.exception("Failed to store OAuth token in object store (key=%s)", key)

    async def delete(self, key: str) -> None:
        try:
            await self._object_store.delete_object(key)
        except NoSuchKeyError:
            pass
        except Exception:
            logger.exception("Failed to delete OAuth token from object store (key=%s)", key)
