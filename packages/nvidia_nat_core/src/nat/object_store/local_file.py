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

from pathlib import Path

from nat.data_models.object_store import KeyAlreadyExistsError  # noqa: F401
from nat.data_models.object_store import NoSuchKeyError  # noqa: F401
from nat.utils.type_utils import override

from .interfaces import ObjectStore
from .models import ObjectStoreItem


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
        raise NotImplementedError

    @override
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        raise NotImplementedError

    @override
    async def get_object(self, key: str) -> ObjectStoreItem:
        raise NotImplementedError

    @override
    async def delete_object(self, key: str) -> None:
        raise NotImplementedError
