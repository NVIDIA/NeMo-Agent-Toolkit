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

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_object_store
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.utils.type_utils import override


class LangSmithObjectStoreConfig(ObjectStoreBaseConfig, name="langsmith"):
    """ObjectStore backed by the LangSmith API.

    The ``key`` passed to ``read_dataframe()`` is treated as the LangSmith
    dataset name.  To use a dataset UUID instead, set ``key_is_id: true``.

    Example YAML::

        object_stores:
          langsmith:
            _type: langsmith

        eval:
          general:
            dataset:
              object_store: langsmith
              key: my-eval-dataset
    """

    key_is_id: bool = Field(
        default=False,
        description="When true, interpret the key as a LangSmith dataset UUID instead of a name.",
    )
    input_key: str = Field(default="input", description="Key within Example.inputs to map to the question column.")
    output_key: str = Field(default="output", description="Key within Example.outputs to map to the answer column.")
    split: str | None = Field(default=None, description="LangSmith dataset split to fetch.")
    as_of: str | None = Field(default=None, description="LangSmith dataset version tag.")
    limit: int | None = Field(default=None, description="Maximum number of examples to fetch.")


class LangSmithObjectStore(ObjectStore):
    """Read-only ObjectStore that fetches data from the LangSmith API."""

    def __init__(self, config: LangSmithObjectStoreConfig) -> None:
        self._config = config

    def _get_client(self):
        from langsmith import Client
        return Client()  # reads LANGCHAIN_API_KEY / LANGSMITH_API_KEY from env

    @override
    async def read_dataframe(self, key: str, format: str | None = None, **kwargs):
        """Fetch a LangSmith dataset and return as a DataFrame.

        Args:
            key: Dataset name (or UUID if ``key_is_id`` is set on the config).
            format: Ignored — data always comes from the LangSmith API.
            **kwargs: ``question_col`` and ``answer_col`` can be passed to
                      override the column names (default: "question", "answer").
        """
        import pandas as pd

        cfg = self._config
        client = self._get_client()

        question_col = kwargs.pop("question_col", "question")
        answer_col = kwargs.pop("answer_col", "answer")
        id_col = kwargs.pop("id_col", "id")

        list_kwargs: dict = {}
        if cfg.key_is_id:
            list_kwargs["dataset_id"] = key
        else:
            list_kwargs["dataset_name"] = key

        if cfg.split:
            list_kwargs["splits"] = [cfg.split]
        if cfg.as_of:
            list_kwargs["as_of"] = cfg.as_of

        rows: list[dict] = []
        for i, ex in enumerate(client.list_examples(**list_kwargs)):
            if cfg.limit is not None and i >= cfg.limit:
                break
            row = {
                id_col: str(ex.id),
                question_col: ex.inputs.get(cfg.input_key, ""),
                answer_col: (ex.outputs or {}).get(cfg.output_key, ""),
            }
            # Preserve all original fields for full_dataset_entry
            for k, v in ex.inputs.items():
                if k not in row:
                    row[k] = v
            if ex.outputs:
                for k, v in ex.outputs.items():
                    if k not in row:
                        row[k] = v
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=[id_col, question_col, answer_col])

        return pd.DataFrame(rows)

    @override
    async def get_object(self, key: str) -> ObjectStoreItem:
        """Fetch a LangSmith dataset and return as JSON bytes."""
        df = await self.read_dataframe(key)
        return ObjectStoreItem(
            data=df.to_json(orient="records").encode(),
            content_type="application/json",
        )

    @override
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        raise NotImplementedError("LangSmithObjectStore is read-only")

    @override
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        raise NotImplementedError("LangSmithObjectStore is read-only")

    @override
    async def delete_object(self, key: str) -> None:
        raise NotImplementedError("LangSmithObjectStore is read-only")


@register_object_store(config_type=LangSmithObjectStoreConfig)
async def langsmith_object_store(config: LangSmithObjectStoreConfig, builder: Builder):
    yield LangSmithObjectStore(config=config)
