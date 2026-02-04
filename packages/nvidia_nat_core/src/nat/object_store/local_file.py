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
from typing import Any

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_object_store
from nat.data_models.object_store import KeyAlreadyExistsError  # noqa: F401
from nat.data_models.object_store import NoSuchKeyError  # noqa: F401
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.utils.type_utils import override

from .interfaces import ObjectStore
from .models import ObjectStoreItem


class LocalFileObjectStoreConfig(ObjectStoreBaseConfig, name="local_file"):
    """
    Configuration for LocalFileObjectStore.

    Attributes:
        base_path: Base directory for all storage operations
    """
    base_path: str


class LocalFileObjectStore(ObjectStore):
    """
    Object store implementation using local filesystem.

    Stores data and metadata as separate files:
    - Data: {base_path}/{key}
    - Metadata: {base_path}/{key}.meta (JSON)

    Args:
        base_path: Base directory for all storage operations
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @override
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        """
        Save object to filesystem. Raises KeyAlreadyExistsError if key exists.

        Args:
            key: Storage key (can include slashes for nested paths)
            item: Object to store

        Raises:
            KeyAlreadyExistsError: If key already exists
        """
        data_path = self.base_path / key
        meta_path = self.base_path / f"{key}.meta"

        # Check if key already exists
        if data_path.exists():
            raise KeyAlreadyExistsError(key)

        # Create parent directories
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Write data file
        data_path.write_bytes(item.data)

        # Write metadata file
        meta_dict: dict[str, Any] = {
            "content_type": item.content_type,
            "metadata": item.metadata
        }
        meta_path.write_text(json.dumps(meta_dict, indent=2))

    @override
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        """
        Save or update object in filesystem.

        Args:
            key: Storage key (can include slashes for nested paths)
            item: Object to store
        """
        data_path = self.base_path / key
        meta_path = self.base_path / f"{key}.meta"

        # Create parent directories
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Write data file (overwrite if exists)
        data_path.write_bytes(item.data)

        # Write metadata file
        meta_dict: dict[str, Any] = {
            "content_type": item.content_type,
            "metadata": item.metadata
        }
        meta_path.write_text(json.dumps(meta_dict, indent=2))

    @override
    async def get_object(self, key: str) -> ObjectStoreItem:
        """
        Retrieve object from filesystem.

        Args:
            key: Storage key

        Returns:
            ObjectStoreItem with data and metadata

        Raises:
            NoSuchKeyError: If key doesn't exist
        """
        data_path = self.base_path / key
        meta_path = self.base_path / f"{key}.meta"

        # Check if data file exists
        if not data_path.exists():
            raise NoSuchKeyError(key)

        # Read data
        data = data_path.read_bytes()

        # Read metadata if exists
        content_type = None
        metadata = None
        if meta_path.exists():
            meta_dict = json.loads(meta_path.read_text())
            content_type = meta_dict.get("content_type")
            metadata = meta_dict.get("metadata")

        return ObjectStoreItem(
            data=data,
            content_type=content_type,
            metadata=metadata
        )

    @override
    async def delete_object(self, key: str) -> None:
        """
        Delete object from filesystem.

        Args:
            key: Storage key

        Raises:
            NoSuchKeyError: If key doesn't exist
        """
        data_path = self.base_path / key
        meta_path = self.base_path / f"{key}.meta"

        # Check if data file exists
        if not data_path.exists():
            raise NoSuchKeyError(key)

        # Delete data file
        data_path.unlink()

        # Delete metadata file if exists
        if meta_path.exists():
            meta_path.unlink()


@register_object_store(config_type=LocalFileObjectStoreConfig)
async def local_file_object_store(
    config: LocalFileObjectStoreConfig,
    _builder: Builder
):
    """
    Factory function to create LocalFileObjectStore from config.

    Args:
        config: LocalFileObjectStoreConfig instance
        _builder: Builder instance (unused)

    Yields:
        LocalFileObjectStore instance
    """
    yield LocalFileObjectStore(base_path=Path(config.base_path))
