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

import logging

import redis.asyncio as redis

from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


class RedisObjectStore(ObjectStore):
    """
    Implementation of ObjectStore that stores objects in Redis.

    Each object is stored as a single binary value at key "nat/object_store/{bucket_name}/{object_key}".
    """

    def __init__(self, *, bucket_name: str, host: str, port: int, db: int):

        super().__init__()

        self._bucket_name = bucket_name
        self._host = host
        self._port = port
        self._db = db
        self._client: redis.Redis | None = None

    async def __aenter__(self) -> "RedisObjectStore":

        if self._client is not None:
            raise RuntimeError("Connection already established")

        self._client = redis.Redis(
            host=self._host,
            port=self._port,
            db=self._db,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

        # Ping to ensure connectivity
        res = await self._client.ping()
        if not res:
            raise RuntimeError("Failed to connect to Redis")

        logger.info("Connected Redis client for %s at %s:%s/%s", self._bucket_name, self._host, self._port, self._db)

        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:

        if not self._client:
            raise RuntimeError("Connection not established")

        await self._client.close()
        self._client = None

    def _make_key(self, key: str) -> str:
        return f"nat/object_store/{self._bucket_name}/{key}"

    @override
    async def put_object(self, key: str, item: ObjectStoreItem):

        if not self._client:
            raise RuntimeError("Connection not established")

        full_key = self._make_key(key)

        item_json = item.model_dump_json()
        # Redis SET with NX ensures we do not overwrite existing keys
        if not await self._client.set(full_key, item_json, nx=True):
            raise KeyAlreadyExistsError(key=key,
                                        additional_message=f"Redis bucket {self._bucket_name} already has key {key}")

    @override
    async def upsert_object(self, key: str, item: ObjectStoreItem):

        if not self._client:
            raise RuntimeError("Connection not established")

        full_key = self._make_key(key)
        item_json = item.model_dump_json()
        await self._client.set(full_key, item_json)

    @override
    async def get_object(self, key: str) -> ObjectStoreItem:

        if not self._client:
            raise RuntimeError("Connection not established")

        full_key = self._make_key(key)
        data = await self._client.get(full_key)
        if data is None:
            raise NoSuchKeyError(key=key,
                                 additional_message=f"Redis bucket {self._bucket_name} does not have key {key}")
        return ObjectStoreItem.model_validate_json(data)

    @override
    async def delete_object(self, key: str):

        if not self._client:
            raise RuntimeError("Connection not established")

        full_key = self._make_key(key)
        deleted = await self._client.delete(full_key)
        if deleted == 0:
            raise NoSuchKeyError(key=key,
                                 additional_message=f"Redis bucket {self._bucket_name} does not have key {key}")
