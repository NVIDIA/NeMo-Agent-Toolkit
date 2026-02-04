# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import pytest

from nat.data_models.object_store import KeyAlreadyExistsError, NoSuchKeyError
from nat.object_store.local_file import LocalFileObjectStore
from nat.object_store.models import ObjectStoreItem


class TestLocalFileObjectStore:
    async def test_create_local_file_object_store(self, tmp_path: Path):
        """Test LocalFileObjectStore can be instantiated with a base path."""
        store = LocalFileObjectStore(base_path=tmp_path)
        assert store is not None
        assert store.base_path == tmp_path

    async def test_put_object(self, tmp_path: Path):
        """Test putting a new object creates data and metadata files."""
        store = LocalFileObjectStore(base_path=tmp_path)

        item = ObjectStoreItem(
            data=b'{"test": "data"}',
            content_type="application/json",
            metadata={"key": "value"}
        )

        await store.put_object("test.json", item)

        # Verify data file
        data_path = tmp_path / "test.json"
        assert data_path.exists()
        assert data_path.read_bytes() == b'{"test": "data"}'

        # Verify metadata file
        meta_path = tmp_path / "test.json.meta"
        assert meta_path.exists()
        meta_data = json.loads(meta_path.read_text())
        assert meta_data["content_type"] == "application/json"
        assert meta_data["metadata"] == {"key": "value"}

    async def test_put_object_raises_on_existing_key(self, tmp_path: Path):
        """Test put_object raises KeyAlreadyExistsError if key exists."""
        store = LocalFileObjectStore(base_path=tmp_path)

        item = ObjectStoreItem(data=b"data", content_type="text/plain")

        await store.put_object("existing.txt", item)

        with pytest.raises(KeyAlreadyExistsError):
            await store.put_object("existing.txt", item)

    async def test_upsert_object_creates_new(self, tmp_path: Path):
        """Test upsert_object creates object if it doesn't exist."""
        store = LocalFileObjectStore(base_path=tmp_path)

        item = ObjectStoreItem(data=b"new data", content_type="text/plain")
        await store.upsert_object("new.txt", item)

        data_path = tmp_path / "new.txt"
        assert data_path.exists()
        assert data_path.read_bytes() == b"new data"

    async def test_upsert_object_updates_existing(self, tmp_path: Path):
        """Test upsert_object overwrites existing object."""
        store = LocalFileObjectStore(base_path=tmp_path)

        # Create initial object
        item1 = ObjectStoreItem(data=b"old data", content_type="text/plain")
        await store.upsert_object("file.txt", item1)

        # Update object
        item2 = ObjectStoreItem(data=b"new data", content_type="application/json")
        await store.upsert_object("file.txt", item2)

        # Verify updated
        data_path = tmp_path / "file.txt"
        assert data_path.read_bytes() == b"new data"

        meta_path = tmp_path / "file.txt.meta"
        meta_data = json.loads(meta_path.read_text())
        assert meta_data["content_type"] == "application/json"

    async def test_get_object(self, tmp_path: Path):
        """Test getting an existing object returns correct data and metadata."""
        store = LocalFileObjectStore(base_path=tmp_path)

        # Put object
        original = ObjectStoreItem(
            data=b"test data",
            content_type="text/plain",
            metadata={"author": "test"}
        )
        await store.put_object("file.txt", original)

        # Get object
        retrieved = await store.get_object("file.txt")

        assert retrieved.data == b"test data"
        assert retrieved.content_type == "text/plain"
        assert retrieved.metadata == {"author": "test"}

    async def test_get_object_missing_metadata(self, tmp_path: Path):
        """Test get_object handles missing .meta file gracefully."""
        store = LocalFileObjectStore(base_path=tmp_path)

        # Manually create data file without metadata
        data_path = tmp_path / "no_meta.txt"
        data_path.write_bytes(b"data without meta")

        # Should still retrieve with None metadata
        retrieved = await store.get_object("no_meta.txt")

        assert retrieved.data == b"data without meta"
        assert retrieved.content_type is None
        assert retrieved.metadata is None

    async def test_get_object_raises_on_missing_key(self, tmp_path: Path):
        """Test get_object raises NoSuchKeyError if key doesn't exist."""
        store = LocalFileObjectStore(base_path=tmp_path)

        with pytest.raises(NoSuchKeyError):
            await store.get_object("nonexistent.txt")
