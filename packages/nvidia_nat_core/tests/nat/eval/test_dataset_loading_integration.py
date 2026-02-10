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

from nat.data_models.dataset_handler import EvalDatasetConfig
from nat.object_store.file_object_store import FileObjectStore


class TestDatasetLoadingIntegration:
    """End-to-end tests for the new ObjectStore-based dataset loading."""

    async def test_load_csv_via_file_path(self, tmp_path: Path):
        """The most common case: file_path shorthand."""
        csv_file = tmp_path / "eval.csv"
        csv_file.write_text("id,question,answer\n1,What is 2+2?,4\n2,What is 3+3?,6")

        store = FileObjectStore(base_path=tmp_path)
        df = await store.read_dataframe("eval.csv")

        assert len(df) == 2
        assert "question" in df.columns

    async def test_load_json_via_file_path(self, tmp_path: Path):
        json_file = tmp_path / "eval.json"
        json_file.write_text(
            json.dumps([
                {
                    "id": "1", "question": "Q1", "answer": "A1"
                },
                {
                    "id": "2", "question": "Q2", "answer": "A2"
                },
            ]))

        store = FileObjectStore(base_path=tmp_path)
        df = await store.read_dataframe("eval.json")

        assert len(df) == 2

    async def test_load_jsonl_via_file_path(self, tmp_path: Path):
        jsonl_file = tmp_path / "eval.jsonl"
        jsonl_file.write_text('{"id": "1", "q": "Q1"}\n{"id": "2", "q": "Q2"}\n')

        store = FileObjectStore(base_path=tmp_path)
        df = await store.read_dataframe("eval.jsonl")

        assert len(df) == 2

    async def test_load_via_subdirectory(self, tmp_path: Path):
        """Simulate loading from a named ObjectStore via subdirectory."""
        csv_file = tmp_path / "datasets" / "eval.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("id,question\n1,hello")

        store = FileObjectStore(base_path=tmp_path)
        df = await store.read_dataframe("datasets/eval.csv")

        assert len(df) == 1
        assert "question" in df.columns

    async def test_format_inference(self, tmp_path: Path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2")

        store = FileObjectStore(base_path=tmp_path)
        df = await store.read_dataframe("data.csv")
        assert list(df.columns) == ["a", "b"]

    async def test_explicit_format_overrides_extension(self, tmp_path: Path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("x,y\n1,2")

        store = FileObjectStore(base_path=tmp_path)
        df = await store.read_dataframe("data.txt", format="csv")
        assert list(df.columns) == ["x", "y"]

    async def test_in_memory_store_read_dataframe(self):
        """Test that InMemoryObjectStore also gets read_dataframe for free."""
        from nat.object_store.in_memory_object_store import InMemoryObjectStore
        from nat.object_store.models import ObjectStoreItem

        store = InMemoryObjectStore()
        csv_data = b"id,name\n1,Alice\n2,Bob"
        await store.put_object("test.csv", ObjectStoreItem(data=csv_data))

        df = await store.read_dataframe("test.csv")
        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]

    async def test_eval_dataset_config_with_file_object_store(self, tmp_path: Path):
        """Test the full config -> store -> dataframe flow."""
        csv_file = tmp_path / "eval.csv"
        csv_file.write_text("id,question,answer\n1,Q1,A1\n2,Q2,A2")

        config = EvalDatasetConfig(file_path=str(csv_file))

        # Simulate what DatasetHandler.load_dataset_df does
        store = FileObjectStore(base_path=Path("."))
        df = await store.read_dataframe(str(config.file_path), format=config.format)

        assert len(df) == 2
        assert "question" in df.columns
        assert "answer" in df.columns
