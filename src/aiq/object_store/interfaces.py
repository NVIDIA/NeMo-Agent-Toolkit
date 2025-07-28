# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import ABC
from abc import abstractmethod

from .models import ObjectStoreItem


class ObjectStore(ABC):
    """
    Abstract interface for an object store.

    Implementations may integrate with various object stores,
    such as S3, MySQL, etc.
    """

    @abstractmethod
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        """
        Save an ObjectStoreItem in the object store with the given key.
        If the key already exists, raise an error.

        Args:
            key (str): The key to save the item under.
            item (ObjectStoreItem): The item to save.

        Raises:
            KeyAlreadyExistsError: If the key already exists.
        """
        pass

    @abstractmethod
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        """
        Save an ObjectStoreItem in the object store with the given key.
        If the key already exists, update the item.

        Args:
            key (str): The key to save the item under.
            item (ObjectStoreItem): The item to save.
        """
        pass

    @abstractmethod
    async def get_object(self, key: str) -> ObjectStoreItem:
        """
        Get an ObjectStoreItem from the object store by key.

        Args:
            key (str): The key to get the item from.

        Returns:
            ObjectStoreItem: The item retrieved from the object store.

        Raises:
            NoSuchKeyError: If the item does not exist.
        """
        pass

    @abstractmethod
    async def delete_object(self, key: str) -> None:
        """
        Delete an ObjectStoreItem from the object store by key.

        Args:
            key (str): The key to delete the item from.

        Raises:
            NoSuchKeyError: If the item does not exist.
        """
        pass
