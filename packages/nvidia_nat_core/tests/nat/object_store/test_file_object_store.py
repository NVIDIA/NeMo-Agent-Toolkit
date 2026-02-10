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

from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.file_object_store import FileObjectStore
from nat.object_store.file_object_store import FileObjectStoreConfig
from nat.object_store.models import ObjectStoreItem


@pytest.mark.asyncio
class TestFileObjectStore:

    @pytest.fixture
    def store(self, tmp_path: Path):
        return FileObjectStore(base_path=tmp_path)

    async def test_put_and_get(self, store: FileObjectStore, tmp_path: Path):
        item = ObjectStoreItem(data=b"hello world", content_type="text/plain")
        await store.put_object("test.txt", item)

        result = await store.get_object("test.txt")
        assert result.data == b"hello world"

    async def test_put_creates_subdirectories(self, store: FileObjectStore, tmp_path: Path):
        item = ObjectStoreItem(data=b"nested")
        await store.put_object("a/b/c/file.txt", item)

        result = await store.get_object("a/b/c/file.txt")
        assert result.data == b"nested"

    async def test_put_duplicate_raises(self, store: FileObjectStore):
        item = ObjectStoreItem(data=b"data")
        await store.put_object("dup.txt", item)
        with pytest.raises(KeyAlreadyExistsError):
            await store.put_object("dup.txt", item)

    async def test_get_missing_raises(self, store: FileObjectStore):
        with pytest.raises(NoSuchKeyError):
            await store.get_object("nonexistent.txt")

    async def test_upsert_creates_and_overwrites(self, store: FileObjectStore):
        item1 = ObjectStoreItem(data=b"v1")
        await store.upsert_object("upserting.txt", item1)
        result = await store.get_object("upserting.txt")
        assert result.data == b"v1"

        item2 = ObjectStoreItem(data=b"v2")
        await store.upsert_object("upserting.txt", item2)
        result = await store.get_object("upserting.txt")
        assert result.data == b"v2"

    async def test_delete(self, store: FileObjectStore):
        item = ObjectStoreItem(data=b"to delete")
        await store.put_object("deleteme.txt", item)
        await store.delete_object("deleteme.txt")
        with pytest.raises(NoSuchKeyError):
            await store.get_object("deleteme.txt")

    async def test_delete_missing_raises(self, store: FileObjectStore):
        with pytest.raises(NoSuchKeyError):
            await store.delete_object("nope.txt")

    async def test_config_name(self):
        config = FileObjectStoreConfig(base_path=Path("/tmp"))
        assert config.base_path == Path("/tmp")


class TestFileObjectStoreReadDataframe:
    """Tests for read_dataframe override (efficient local file reads)."""

    @pytest.fixture
    def store(self, tmp_path: Path):
        return FileObjectStore(base_path=tmp_path)

    @pytest.mark.asyncio
    async def test_read_csv_file(self, store: FileObjectStore, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,question\n1,hello\n2,world")

        df = await store.read_dataframe("data.csv")
        assert len(df) == 2
        assert list(df.columns) == ["id", "question"]

    @pytest.mark.asyncio
    async def test_read_json_file(self, store: FileObjectStore, tmp_path: Path):
        json_path = tmp_path / "data.json"
        json_path.write_text(json.dumps([{"a": 1}, {"a": 2}]))

        df = await store.read_dataframe("data.json")
        assert len(df) == 2

    @pytest.mark.asyncio
    async def test_read_with_explicit_format(self, store: FileObjectStore, tmp_path: Path):
        csv_path = tmp_path / "mydata"
        csv_path.write_text("x,y\n1,2")

        df = await store.read_dataframe("mydata", format="csv")
        assert list(df.columns) == ["x", "y"]
