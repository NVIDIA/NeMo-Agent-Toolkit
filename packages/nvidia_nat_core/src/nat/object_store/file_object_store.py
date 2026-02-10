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

import mimetypes
from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_object_store
from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.utils.type_utils import override

from .interfaces import ObjectStore
from .models import ObjectStoreItem


class FileObjectStoreConfig(ObjectStoreBaseConfig, name="file"):
    """ObjectStore backed by the local filesystem."""
    base_path: Path = Field(default=Path("."), description="Root directory for file storage.")


class FileObjectStore(ObjectStore):
    """ObjectStore implementation that reads/writes to the local filesystem."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)

    def _resolve(self, key: str) -> Path:
        return self._base_path / key

    @override
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        path = self._resolve(key)
        if path.exists():
            raise KeyAlreadyExistsError(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(item.data)

    @override
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(item.data)

    @override
    async def get_object(self, key: str) -> ObjectStoreItem:
        path = self._resolve(key)
        if not path.exists():
            raise NoSuchKeyError(key)
        content_type, _ = mimetypes.guess_type(str(path))
        return ObjectStoreItem(data=path.read_bytes(), content_type=content_type)

    @override
    async def delete_object(self, key: str) -> None:
        path = self._resolve(key)
        if not path.exists():
            raise NoSuchKeyError(key)
        path.unlink()

    @override
    async def read_dataframe(self, key: str, format: str | None = None, **kwargs):
        """Override: read directly from file path for efficiency.

        Local files are passed directly to pandas readers instead of
        going through bytes -> BytesIO.
        """
        import pandas as pd

        from nat.object_store.format_parsers import infer_format

        path = self._resolve(key)
        if not path.exists():
            raise NoSuchKeyError(key)

        fmt = format or infer_format(key)

        direct_readers: dict = {
            "csv": pd.read_csv,
            "json": pd.read_json,
            "parquet": pd.read_parquet,
            "xls": lambda p, **kw: pd.read_excel(p, engine="openpyxl", **kw),
        }

        if fmt in direct_readers:
            return direct_readers[fmt](path, **kwargs)

        # Fallback for formats without direct file path support (e.g. jsonl)
        return await super().read_dataframe(key, fmt, **kwargs)


@register_object_store(config_type=FileObjectStoreConfig)
async def file_object_store(config: FileObjectStoreConfig, builder: Builder):
    yield FileObjectStore(base_path=config.base_path)
