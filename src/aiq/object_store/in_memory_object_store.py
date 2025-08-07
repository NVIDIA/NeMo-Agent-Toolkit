# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_object_store
from aiq.data_models.object_store import KeyAlreadyExistsError
from aiq.data_models.object_store import NoSuchKeyError
from aiq.data_models.object_store import ObjectStoreBaseConfig
from aiq.utils.type_utils import override

from .interfaces import ObjectStore
from .models import ObjectStoreItem


class InMemoryObjectStoreConfig(ObjectStoreBaseConfig, name="in_memory"):
    """
    Object store that stores objects in memory. Objects are not persisted when the process shuts down.
    """
    pass


class InMemoryObjectStore(ObjectStore):
    """
    Implementation of ObjectStore that stores objects in memory. Objects are not persisted when the process shuts down.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._store: dict[str, ObjectStoreItem] = {}

    @override
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        async with self._lock:
            if key in self._store:
                raise KeyAlreadyExistsError(key)
            self._store[key] = item

    @override
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        async with self._lock:
            self._store[key] = item

    @override
    async def get_object(self, key: str) -> ObjectStoreItem:
        async with self._lock:
            value = self._store.get(key)
            if value is None:
                raise NoSuchKeyError(key)
            return value

    @override
    async def delete_object(self, key: str) -> None:
        try:
            async with self._lock:
                self._store.pop(key)
        except KeyError:
            raise NoSuchKeyError(key)


@register_object_store(config_type=InMemoryObjectStoreConfig)
async def in_memory_object_store(config: InMemoryObjectStoreConfig, builder: Builder):
    yield InMemoryObjectStore()
