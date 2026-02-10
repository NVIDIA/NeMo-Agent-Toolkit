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

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from nat.plugins.langchain.object_store.langsmith_object_store import LangSmithObjectStore
from nat.plugins.langchain.object_store.langsmith_object_store import LangSmithObjectStoreConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example(inputs: dict, outputs: dict | None = None, example_id: str | None = None):
    """Create a mock LangSmith Example object."""
    return SimpleNamespace(
        id=uuid.UUID(example_id) if example_id else uuid.uuid4(),
        inputs=inputs,
        outputs=outputs,
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestLangSmithObjectStoreConfig:

    def test_default_config(self):
        config = LangSmithObjectStoreConfig()
        assert config.input_key == "input"
        assert config.output_key == "output"
        assert config.key_is_id is False
        assert config.split is None
        assert config.limit is None

    def test_custom_config(self):
        config = LangSmithObjectStoreConfig(
            input_key="prompt",
            output_key="response",
            split="test",
            as_of="v2",
            limit=50,
        )
        assert config.input_key == "prompt"
        assert config.output_key == "response"
        assert config.split == "test"
        assert config.as_of == "v2"
        assert config.limit == 50

    def test_key_is_id(self):
        config = LangSmithObjectStoreConfig(key_is_id=True)
        assert config.key_is_id is True


# ---------------------------------------------------------------------------
# ObjectStore tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLangSmithObjectStore:

    @patch("nat.plugins.langchain.object_store.langsmith_object_store.LangSmithObjectStore._get_client")
    async def test_read_dataframe_by_name(self, mock_get_client):
        examples = [
            _make_example({"input": f"q{i}"}, {"output": f"a{i}"}, f"00000000-0000-0000-0000-00000000000{i}")
            for i in range(1, 4)
        ]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter(examples)
        mock_get_client.return_value = mock_client

        config = LangSmithObjectStoreConfig()
        store = LangSmithObjectStore(config=config)
        df = await store.read_dataframe("my-dataset")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns[:3]) == ["id", "question", "answer"]
        assert df["question"].tolist() == ["q1", "q2", "q3"]
        assert df["answer"].tolist() == ["a1", "a2", "a3"]
        mock_client.list_examples.assert_called_once_with(dataset_name="my-dataset")

    @patch("nat.plugins.langchain.object_store.langsmith_object_store.LangSmithObjectStore._get_client")
    async def test_read_dataframe_by_id(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter([])
        mock_get_client.return_value = mock_client

        config = LangSmithObjectStoreConfig(key_is_id=True)
        store = LangSmithObjectStore(config=config)
        await store.read_dataframe("abc-123-uuid")

        mock_client.list_examples.assert_called_once_with(dataset_id="abc-123-uuid")

    @patch("nat.plugins.langchain.object_store.langsmith_object_store.LangSmithObjectStore._get_client")
    async def test_read_dataframe_custom_keys(self, mock_get_client):
        examples = [_make_example({"prompt": "hello"}, {"response": "world"})]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter(examples)
        mock_get_client.return_value = mock_client

        config = LangSmithObjectStoreConfig(input_key="prompt", output_key="response")
        store = LangSmithObjectStore(config=config)
        df = await store.read_dataframe("ds", question_col="q", answer_col="a")

        assert df["q"].tolist() == ["hello"]
        assert df["a"].tolist() == ["world"]

    @patch("nat.plugins.langchain.object_store.langsmith_object_store.LangSmithObjectStore._get_client")
    async def test_read_dataframe_limit(self, mock_get_client):
        examples = [_make_example({"input": f"q{i}"}, {"output": f"a{i}"}) for i in range(10)]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter(examples)
        mock_get_client.return_value = mock_client

        config = LangSmithObjectStoreConfig(limit=2)
        store = LangSmithObjectStore(config=config)
        df = await store.read_dataframe("ds")

        assert len(df) == 2

    @patch("nat.plugins.langchain.object_store.langsmith_object_store.LangSmithObjectStore._get_client")
    async def test_read_dataframe_split(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter([])
        mock_get_client.return_value = mock_client

        config = LangSmithObjectStoreConfig(split="test")
        store = LangSmithObjectStore(config=config)
        await store.read_dataframe("ds")

        call_kwargs = mock_client.list_examples.call_args[1]
        assert call_kwargs["splits"] == ["test"]

    @patch("nat.plugins.langchain.object_store.langsmith_object_store.LangSmithObjectStore._get_client")
    async def test_read_dataframe_as_of(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter([])
        mock_get_client.return_value = mock_client

        config = LangSmithObjectStoreConfig(as_of="v2")
        store = LangSmithObjectStore(config=config)
        await store.read_dataframe("ds")

        call_kwargs = mock_client.list_examples.call_args[1]
        assert call_kwargs["as_of"] == "v2"

    @patch("nat.plugins.langchain.object_store.langsmith_object_store.LangSmithObjectStore._get_client")
    async def test_read_dataframe_empty(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter([])
        mock_get_client.return_value = mock_client

        config = LangSmithObjectStoreConfig()
        store = LangSmithObjectStore(config=config)
        df = await store.read_dataframe("ds")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["id", "question", "answer"]

    @patch("nat.plugins.langchain.object_store.langsmith_object_store.LangSmithObjectStore._get_client")
    async def test_read_dataframe_null_outputs(self, mock_get_client):
        examples = [_make_example({"input": "q1"}, None)]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter(examples)
        mock_get_client.return_value = mock_client

        config = LangSmithObjectStoreConfig()
        store = LangSmithObjectStore(config=config)
        df = await store.read_dataframe("ds")

        assert len(df) == 1
        assert df["question"].tolist() == ["q1"]
        assert df["answer"].tolist() == [""]

    @patch("nat.plugins.langchain.object_store.langsmith_object_store.LangSmithObjectStore._get_client")
    async def test_read_dataframe_extra_fields_preserved(self, mock_get_client):
        examples = [_make_example({"input": "q1", "context": "ctx1"}, {"output": "a1", "score": 0.9})]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter(examples)
        mock_get_client.return_value = mock_client

        config = LangSmithObjectStoreConfig()
        store = LangSmithObjectStore(config=config)
        df = await store.read_dataframe("ds")

        assert "context" in df.columns
        assert "score" in df.columns
        assert df["context"].tolist() == ["ctx1"]
        assert df["score"].tolist() == [0.9]

    @patch("nat.plugins.langchain.object_store.langsmith_object_store.LangSmithObjectStore._get_client")
    async def test_get_object_returns_json(self, mock_get_client):
        examples = [_make_example({"input": "q1"}, {"output": "a1"}, "00000000-0000-0000-0000-000000000001")]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter(examples)
        mock_get_client.return_value = mock_client

        config = LangSmithObjectStoreConfig()
        store = LangSmithObjectStore(config=config)
        item = await store.get_object("ds")

        assert item.content_type == "application/json"
        assert b"q1" in item.data

    async def test_put_raises(self):
        config = LangSmithObjectStoreConfig()
        store = LangSmithObjectStore(config=config)
        with pytest.raises(NotImplementedError):
            from nat.object_store.models import ObjectStoreItem
            await store.put_object("key", ObjectStoreItem(data=b"x"))

    async def test_delete_raises(self):
        config = LangSmithObjectStoreConfig()
        store = LangSmithObjectStore(config=config)
        with pytest.raises(NotImplementedError):
            await store.delete_object("key")


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:

    def test_registration(self):
        import nat.plugins.langchain.object_store.langsmith_object_store  # noqa: F401
        from nat.cli.type_registry import GlobalTypeRegistry

        registry = GlobalTypeRegistry.get()
        info = registry.get_object_store(LangSmithObjectStoreConfig)
        assert info is not None
        assert info.build_fn is not None
