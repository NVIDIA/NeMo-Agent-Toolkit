import json

import pytest

from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem


class SimpleMemoryStore(ObjectStore):
    """Minimal in-memory ObjectStore for testing without triggering type registry."""

    def __init__(self):
        self._store: dict[str, ObjectStoreItem] = {}

    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        self._store[key] = item

    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        self._store[key] = item

    async def get_object(self, key: str) -> ObjectStoreItem:
        if key not in self._store:
            raise NoSuchKeyError(key)
        return self._store[key]

    async def delete_object(self, key: str) -> None:
        if key not in self._store:
            raise NoSuchKeyError(key)
        del self._store[key]


@pytest.mark.asyncio
class TestObjectStoreReadDataframe:

    async def test_read_csv_from_object_store(self):
        store = SimpleMemoryStore()
        csv_data = b"id,question,answer\n1,What is 2+2?,4\n2,What is 3+3?,6"
        await store.put_object("data.csv", ObjectStoreItem(data=csv_data))

        df = await store.read_dataframe("data.csv")
        assert len(df) == 2
        assert list(df.columns) == ["id", "question", "answer"]

    async def test_read_json_from_object_store(self):
        store = SimpleMemoryStore()
        json_data = json.dumps([{"a": 1}, {"a": 2}]).encode()
        await store.put_object("data.json", ObjectStoreItem(data=json_data))

        df = await store.read_dataframe("data.json")
        assert len(df) == 2

    async def test_read_with_explicit_format(self):
        store = SimpleMemoryStore()
        csv_data = b"x,y\n1,2"
        await store.put_object("mydata", ObjectStoreItem(data=csv_data))

        df = await store.read_dataframe("mydata", format="csv")
        assert list(df.columns) == ["x", "y"]

    async def test_read_infers_format_from_key(self):
        store = SimpleMemoryStore()
        csv_data = b"col1\nval1"
        await store.put_object("path/to/file.csv", ObjectStoreItem(data=csv_data))

        df = await store.read_dataframe("path/to/file.csv")
        assert "col1" in df.columns

    async def test_read_unknown_format_raises(self):
        store = SimpleMemoryStore()
        await store.put_object("file.xyz", ObjectStoreItem(data=b"data"))

        with pytest.raises(ValueError):
            await store.read_dataframe("file.xyz")
