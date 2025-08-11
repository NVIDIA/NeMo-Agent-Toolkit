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

import uuid
from abc import abstractmethod
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio

from aiq.data_models.object_store import KeyAlreadyExistsError
from aiq.data_models.object_store import NoSuchKeyError
from aiq.object_store.interfaces import ObjectStore
from aiq.object_store.models import ObjectStoreItem


@pytest.mark.asyncio(loop_scope="class")
class ObjectStoreTests:

    @abstractmethod
    @asynccontextmanager
    async def _get_store(self):
        yield

    @pytest_asyncio.fixture(loop_scope="class", scope="class")
    async def store(self):

        async with self._get_store() as store:
            yield store

    async def test_create_object_store(self, store: ObjectStore):
        assert isinstance(store, ObjectStore)

    async def test_put_object(self, store: ObjectStore):

        # Use a random key to avoid conflicts with other tests
        key = f"test_key_{uuid.uuid4()}"

        initial_item = ObjectStoreItem(data=b"test_value")
        await store.put_object(key, initial_item)

        # Try to put the same object again
        with pytest.raises(KeyAlreadyExistsError):
            await store.put_object(key, initial_item)

    async def test_upsert_object(self, store: ObjectStore):
        key = f"test_key_{uuid.uuid4()}"

        initial_item = ObjectStoreItem(data=b"test_value", content_type="text/plain", metadata={"key": "value"})

        await store.upsert_object(key, initial_item)

        # Check that the object exists
        retrieved_item = await store.get_object(key)
        assert retrieved_item.data == initial_item.data
        assert retrieved_item.content_type == initial_item.content_type
        assert retrieved_item.metadata == initial_item.metadata

        # Upsert the object with a new value
        new_item = ObjectStoreItem(data=b"new_value", content_type="application/json", metadata={"key": "new_value"})
        await store.upsert_object(key, new_item)

        # Check that the object was updated
        retrieved_item = await store.get_object(key)
        assert retrieved_item.data == new_item.data
        assert retrieved_item.content_type == new_item.content_type
        assert retrieved_item.metadata == new_item.metadata

    async def test_get_object(self, store: ObjectStore):

        key = f"test_key_{uuid.uuid4()}"

        initial_item = ObjectStoreItem(data=b"test_value", content_type="text/plain", metadata={"key": "value"})
        await store.put_object(key, initial_item)

        retrieved_item = await store.get_object(key)
        assert retrieved_item.data == initial_item.data
        assert retrieved_item.content_type == initial_item.content_type
        assert retrieved_item.metadata == initial_item.metadata

        # Try to get an object that doesn't exist
        with pytest.raises(NoSuchKeyError):
            await store.get_object(f"test_key_{uuid.uuid4()}")

    async def test_delete_object(self, store: ObjectStore):

        key = f"test_key_{uuid.uuid4()}"

        initial_item = ObjectStoreItem(data=b"test_value")
        await store.put_object(key, initial_item)

        # Check that the object exists
        retrieved_item = await store.get_object(key)
        assert retrieved_item.data == initial_item.data

        # Delete the object
        await store.delete_object(key)

        # Try to get the object again
        with pytest.raises(NoSuchKeyError):
            await store.get_object(key)

        # Try to delete the object again
        with pytest.raises(NoSuchKeyError):
            await store.delete_object(key)
