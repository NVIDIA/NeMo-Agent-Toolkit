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

from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.object_store.models import ObjectStoreListItem


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

    async def test_list_objects(self, store: ObjectStore):
        """Test listing objects with and without prefix filtering"""

        test_id = str(uuid.uuid4())[:8]

        test_objects = {
            f"videos/{test_id}/video1.mp4":
                ObjectStoreItem(data=b"video1_data", content_type="video/mp4", metadata={"title": "Video 1"}),
            f"videos/{test_id}/video2.mp4":
                ObjectStoreItem(data=b"video2_data", content_type="video/mp4", metadata={"title": "Video 2"}),
            f"images/{test_id}/image1.png":
                ObjectStoreItem(data=b"image1_data", content_type="image/png", metadata={"title": "Image 1"}),
            f"docs/{test_id}/doc1.txt":
                ObjectStoreItem(data=b"doc1_data", content_type="text/plain")
        }

        for key, item in test_objects.items():
            await store.put_object(key, item)

        # Test 1: List all objects (no prefix)
        all_objects = await store.list_objects()
        all_keys = {obj.key for obj in all_objects}

        for key in test_objects.keys():
            assert key in all_keys, f"Expected key {key} not found in all_objects"

        # Test 2: List with videos prefix
        video_objects = await store.list_objects(prefix=f"videos/{test_id}/")
        assert len(video_objects) == 2, f"Expected 2 video objects, got {len(video_objects)}"

        video_keys = {obj.key for obj in video_objects}
        assert f"videos/{test_id}/video1.mp4" in video_keys
        assert f"videos/{test_id}/video2.mp4" in video_keys

        for obj in video_objects:
            assert isinstance(obj, ObjectStoreListItem)
            assert obj.key.startswith(f"videos/{test_id}/")
            assert obj.size > 0
            assert obj.content_type == "video/mp4"
            assert obj.metadata is not None
            assert "title" in obj.metadata

        # Test 3: List with images prefix
        image_objects = await store.list_objects(prefix=f"images/{test_id}/")
        assert len(image_objects) == 1
        assert image_objects[0].key == f"images/{test_id}/image1.png"
        assert image_objects[0].content_type == "image/png"

        # Test 4: List with non-existent prefix
        empty_objects = await store.list_objects(prefix=f"nonexistent/{test_id}/")
        assert len(empty_objects) == 0

        # Test 5: List with partial prefix
        all_test_objects = await store.list_objects(prefix=f"videos/{test_id}")
        assert len(all_test_objects) >= 2  # At least our video objects

        # Cleanup
        for key in test_objects.keys():
            await store.delete_object(key)
