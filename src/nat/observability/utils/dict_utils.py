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

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)


class KeyedLock:
    """
    A lock manager that provides an asyncio-compatible lock for each unique key.

    This allows for fine-grained locking based on arbitrary keys, so that
    concurrent operations on different keys do not block each other.

    Attributes:
        _locks (AsyncDictionary): A dictionary to store locks per key.
    """

    def __init__(self):
        """
        Initialize the KeyedLock with an internal AsyncSafeWeakKeyDictionary to store locks per key.
        """
        self._locks: AsyncDictionary = AsyncDictionary()

    @asynccontextmanager
    async def get_lock(self, key: Any) -> AsyncGenerator[None]:
        """
        Async context manager to acquire a lock for a specific key.

        Args:
            key (Any): The key to lock on.

        Yields:
            None: Control is yielded while the lock is held.
        """
        lock = await self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            await self._locks.set(key, lock)
        async with lock:
            yield

    async def delete(self, key: Any) -> None:
        """
        Remove the lock associated with the given key, if it exists.

        Args:
            key (Any): The key whose lock should be removed.
        """
        await self._locks.delete(key)

    async def clear(self) -> None:
        """
        Remove all locks managed by this KeyedLock instance.
        """
        await self._locks.clear()


class AsyncDictionary:
    """
    An asyncio-safe dictionary.

    This class wraps a regular dictionary with an asyncio.Lock to ensure
    thread safety for concurrent async operations.

    Attributes:
        _dict (dict): A dictionary to store the key-value pairs.
        _lock (asyncio.Lock): A lock to synchronize access to the dictionary.
    """

    def __init__(self):
        """
        Initialize the AsyncDictionary with a regular dictionary and an asyncio.Lock.
        """
        self._dict: dict = {}
        self._lock = asyncio.Lock()

    async def get(self, key: Any, default: Any | None = None) -> Any | None:
        """
        Get the value associated with the given key, or return default if not found.

        Args:
            key (Any): The key to look up.
            default (Any | None, optional): The value to return if key is not found. Defaults to None.

        Returns:
            Any | None: The value associated with the key, or default.
        """
        async with self._lock:
            return self._dict.get(key, default)

    async def keys(self) -> list[Any]:
        """
        Get a list of all keys currently in the dictionary.

        Returns:
            list[Any]: A list of keys.
        """
        async with self._lock:
            return list(self._dict.keys())

    async def values(self) -> list[Any]:
        """
        Get a list of all values currently in the dictionary.

        Returns:
            list[Any]: A list of values.
        """
        async with self._lock:
            return list(self._dict.values())

    async def set(self, key: Any, value: Any) -> None:
        """
        Set the value for the given key, overwriting any existing value.

        Args:
            key (Any): The key to set.
            value (Any): The value to associate with the key.
        """
        async with self._lock:
            self._dict[key] = value

    async def set_strict(self, key: Any, value: Any) -> None:
        """
        Set the value for the given key only if the key does not already exist.

        Args:
            key (Any): The key to set.
            value (Any): The value to associate with the key.

        Raises:
            ValueError: If the key already exists in the dictionary.
        """
        async with self._lock:
            if key in self._dict:
                raise ValueError(f"Key '{key}' already exists")
            self._dict[key] = value

    async def delete(self, key: Any) -> None:
        """
        Remove the value associated with the given key, if it exists.

        Args:
            key (Any): The key to remove.
        """
        async with self._lock:
            self._dict.pop(key, None)

    async def delete_strict(self, key: Any) -> None:
        """
        Remove the value associated with the given key, raising an error if the key does not exist.

        Args:
            key (Any): The key to remove.

        Raises:
            ValueError: If the key does not exist in the dictionary.
        """
        async with self._lock:
            if key not in self._dict:
                raise ValueError(f"Key '{key}' does not exist")
            self._dict.pop(key)

    async def clear(self) -> None:
        """
        Remove all items from the dictionary.
        """
        async with self._lock:
            self._dict.clear()

    async def items(self) -> dict[Any, Any]:
        """
        Get a copy of the dictionary's items as a regular dict.

        Returns:
            dict[Any, Any]: A copy of the dictionary's items.
        """
        async with self._lock:
            return dict(self._dict)  # Return a copy to prevent external modification


class AsyncSafeWeakKeyDictionary(AsyncDictionary):
    """
    An asyncio-safe, weakly-referenced dictionary.

    This class wraps a WeakKeyDictionary with an asyncio.Lock to ensure
    thread safety for concurrent async operations.

    Attributes:
        _dict (WeakKeyDictionary): A dictionary to store the key-value pairs.
        _lock (asyncio.Lock): A lock to synchronize access to the dictionary.
    """

    def __init__(self):
        """
        Initialize the AsyncSafeWeakKeyDictionary with a WeakKeyDictionary and an asyncio.Lock.
        """
        super().__init__()
        self._dict: WeakKeyDictionary = WeakKeyDictionary()
        self._lock = asyncio.Lock()


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dictionaries, prioritizing non-null values from the first dictionary.

    Args:
        dict1 (dict): First dictionary (higher priority)
        dict2 (dict): Second dictionary (lower priority)

    Returns:
        dict: Merged dictionary with non-null values from dict1 taking precedence
    """
    result = dict2.copy()  # Start with a copy of the second dictionary
    for key, value in dict1.items():
        if value is not None:  # Only update if value is not None
            result[key] = value
    return result
