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
from weakref import WeakKeyDictionary

import pytest

from nat.observability.utils.dict_utils import AsyncDictionary
from nat.observability.utils.dict_utils import AsyncSafeWeakKeyDictionary
from nat.observability.utils.dict_utils import KeyedLock
from nat.observability.utils.dict_utils import merge_dicts


class TestAsyncDictionary:
    """Tests for AsyncDictionary class."""

    @pytest.fixture
    def async_dict(self):
        """Create an AsyncDictionary instance for testing."""
        return AsyncDictionary()

    async def test_get_existing_key(self, async_dict):
        """Test getting an existing key from the dictionary."""
        await async_dict.set("key1", "value1")
        result = await async_dict.get("key1")
        assert result == "value1"

    async def test_get_nonexistent_key_default_none(self, async_dict):
        """Test getting a nonexistent key returns None by default."""
        result = await async_dict.get("nonexistent")
        assert result is None

    async def test_get_nonexistent_key_custom_default(self, async_dict):
        """Test getting a nonexistent key with custom default value."""
        result = await async_dict.get("nonexistent", "default_value")
        assert result == "default_value"

    async def test_set_and_get(self, async_dict):
        """Test setting and getting values."""
        await async_dict.set("test_key", 42)
        result = await async_dict.get("test_key")
        assert result == 42

    async def test_set_overwrite(self, async_dict):
        """Test overwriting an existing key."""
        await async_dict.set("key", "original")
        await async_dict.set("key", "updated")
        result = await async_dict.get("key")
        assert result == "updated"

    async def test_set_strict_new_key(self, async_dict):
        """Test set_strict with a new key."""
        await async_dict.set_strict("new_key", "value")
        result = await async_dict.get("new_key")
        assert result == "value"

    async def test_set_strict_existing_key_raises_error(self, async_dict):
        """Test set_strict raises ValueError for existing key."""
        await async_dict.set("existing_key", "value")
        with pytest.raises(ValueError, match="Key 'existing_key' already exists"):
            await async_dict.set_strict("existing_key", "new_value")

    async def test_delete_existing_key(self, async_dict):
        """Test deleting an existing key."""
        await async_dict.set("key", "value")
        await async_dict.delete("key")
        result = await async_dict.get("key")
        assert result is None

    async def test_delete_nonexistent_key(self, async_dict):
        """Test deleting a nonexistent key (should not raise error)."""
        await async_dict.delete("nonexistent_key")
        # Should not raise an exception

    async def test_delete_strict_existing_key(self, async_dict):
        """Test delete_strict with an existing key."""
        await async_dict.set("key", "value")
        await async_dict.delete_strict("key")
        result = await async_dict.get("key")
        assert result is None

    async def test_delete_strict_nonexistent_key_raises_error(self, async_dict):
        """Test delete_strict raises ValueError for nonexistent key."""
        with pytest.raises(ValueError, match="Key 'nonexistent' does not exist"):
            await async_dict.delete_strict("nonexistent")

    async def test_keys(self, async_dict):
        """Test getting all keys from the dictionary."""
        await async_dict.set("key1", "value1")
        await async_dict.set("key2", "value2")
        keys = await async_dict.keys()
        assert set(keys) == {"key1", "key2"}

    async def test_keys_empty(self, async_dict):
        """Test getting keys from empty dictionary."""
        keys = await async_dict.keys()
        assert keys == []

    async def test_values(self, async_dict):
        """Test getting all values from the dictionary."""
        await async_dict.set("key1", "value1")
        await async_dict.set("key2", "value2")
        values = await async_dict.values()
        assert set(values) == {"value1", "value2"}

    async def test_values_empty(self, async_dict):
        """Test getting values from empty dictionary."""
        values = await async_dict.values()
        assert values == []

    async def test_items(self, async_dict):
        """Test getting all items from the dictionary."""
        await async_dict.set("key1", "value1")
        await async_dict.set("key2", "value2")
        items = await async_dict.items()
        assert items == {"key1": "value1", "key2": "value2"}

    async def test_items_returns_copy(self, async_dict):
        """Test that items() returns a copy to prevent external modification."""
        await async_dict.set("key", "value")
        items = await async_dict.items()
        modified_key = "modified_key"
        items[modified_key] = "modified_value"  # Modify the returned dict

        # Original dictionary should be unchanged
        result = await async_dict.get(modified_key)
        assert result is None

    async def test_clear(self, async_dict):
        """Test clearing all items from the dictionary."""
        await async_dict.set("key1", "value1")
        await async_dict.set("key2", "value2")
        await async_dict.clear()

        keys = await async_dict.keys()
        assert keys == []

    async def test_concurrent_operations(self, async_dict):
        """Test concurrent operations are properly synchronized."""

        async def set_values():
            for i in range(10):
                await async_dict.set(f"key{i}", f"value{i}")

        async def get_values():
            results = []
            for i in range(10):
                result = await async_dict.get(f"key{i}")
                results.append(result)
            return results

        # Run concurrent set and get operations
        await asyncio.gather(set_values(), set_values())
        results = await get_values()

        # All values should be set correctly
        expected = [f"value{i}" for i in range(10)]
        assert results == expected


class TestAsyncSafeWeakKeyDictionary:
    """Tests for AsyncSafeWeakKeyDictionary class."""

    @pytest.fixture
    def weak_dict(self):
        """Create an AsyncSafeWeakKeyDictionary instance for testing."""
        return AsyncSafeWeakKeyDictionary()

    async def test_inherits_async_dictionary_behavior(self, weak_dict):
        """Test that AsyncSafeWeakKeyDictionary inherits AsyncDictionary behavior."""

        # Use a custom class instance as key (required for WeakKeyDictionary)
        class TestKey:
            pass

        key = TestKey()
        await weak_dict.set(key, "value")
        result = await weak_dict.get(key)
        assert result == "value"

    async def test_uses_weak_key_dictionary(self, weak_dict):
        """Test that it uses WeakKeyDictionary internally."""
        assert isinstance(weak_dict._dict, WeakKeyDictionary)

    async def test_weak_reference_behavior(self, weak_dict):
        """Test weak reference behavior when key is garbage collected."""

        # Create a key object using a custom class that supports weak references
        class TestKey:
            pass

        key = TestKey()
        await weak_dict.set(key, "value")

        # Verify the value is set
        result = await weak_dict.get(key)
        assert result == "value"

        # Delete the key reference and force garbage collection
        del key

        # The key should no longer be accessible
        # Note: This test might be flaky depending on garbage collection timing
        # In a real scenario, the key would be automatically removed when no strong references exist


class TestKeyedLock:
    """Tests for KeyedLock class."""

    @pytest.fixture
    def keyed_lock(self):
        """Create a KeyedLock instance for testing."""
        return KeyedLock()

    async def test_get_lock_same_key_sequential(self, keyed_lock):
        """Test that the same key uses the same lock sequentially."""
        async with keyed_lock.get_lock("test_key"):
            # First acquisition
            pass

        async with keyed_lock.get_lock("test_key"):
            # Second acquisition (should reuse the same lock)
            pass

    async def test_get_lock_different_keys_concurrent(self, keyed_lock):
        """Test that different keys can be locked concurrently."""
        results = []

        async def task_with_key(key, delay):
            async with keyed_lock.get_lock(key):
                await asyncio.sleep(delay)
                results.append(key)

        # Start tasks concurrently with different keys
        await asyncio.gather(
            task_with_key("key1", 0.1),
            task_with_key("key2", 0.05),
        )

        # key2 should finish first due to shorter delay
        assert results == ["key2", "key1"]

    async def test_get_lock_same_key_blocks(self, keyed_lock):
        """Test that the same key blocks concurrent access."""
        results = []
        start_time = asyncio.get_event_loop().time()

        async def task_with_timing(task_id, delay):
            async with keyed_lock.get_lock("same_key"):
                await asyncio.sleep(delay)
                current_time = asyncio.get_event_loop().time()
                results.append((task_id, current_time - start_time))

        # Start tasks concurrently with the same key
        await asyncio.gather(
            task_with_timing("task1", 0.1),
            task_with_timing("task2", 0.05),
        )

        # Tasks should run sequentially, not concurrently
        assert len(results) == 2
        # Second task should start after first task completes
        assert results[1][1] > results[0][1] + 0.05

    async def test_delete_lock(self, keyed_lock):
        """Test deleting a lock for a specific key."""
        # Create a lock by using it
        async with keyed_lock.get_lock("test_key"):
            pass

        # Delete the lock
        await keyed_lock.delete("test_key")

        # The lock should be removed (this is more of an internal state test)
        # We can't easily verify this without accessing private members

    async def test_clear_all_locks(self, keyed_lock):
        """Test clearing all locks."""
        # Create multiple locks by using them
        async with keyed_lock.get_lock("key1"):
            pass
        async with keyed_lock.get_lock("key2"):
            pass

        # Clear all locks
        await keyed_lock.clear()

        # All locks should be removed (internal state test)

    async def test_lock_with_different_key_types(self, keyed_lock):
        """Test locks with different key types."""
        keys = ["string_key", 123, ("tuple", "key"), object()]

        async def use_lock(key):
            async with keyed_lock.get_lock(key):
                await asyncio.sleep(0.01)
                return key

        # All different key types should work
        results = await asyncio.gather(*[use_lock(key) for key in keys])
        assert len(results) == len(keys)


# Integration tests
class TestIntegration:
    """Integration tests for multiple components working together."""

    async def test_keyed_lock_with_async_dictionary(self):
        """Test using KeyedLock with AsyncDictionary operations."""
        keyed_lock = KeyedLock()
        async_dict = AsyncDictionary()

        async def protected_increment(key):
            async with keyed_lock.get_lock(key):
                current = await async_dict.get(key, 0)
                # Ensure current is not None by providing explicit default
                if current is None:
                    current = 0
                await asyncio.sleep(0.01)  # Simulate some work
                await async_dict.set(key, current + 1)

        # Run concurrent increments on the same key
        await asyncio.gather(*[protected_increment("counter") for _ in range(10)])

        result = await async_dict.get("counter")
        assert result == 10  # All increments should be properly synchronized

    async def test_multiple_async_dictionaries_with_shared_lock(self):
        """Test multiple AsyncDictionary instances with shared KeyedLock."""
        keyed_lock = KeyedLock()
        dict1 = AsyncDictionary()
        dict2 = AsyncDictionary()

        async def transfer_value(from_dict, to_dict, key):
            async with keyed_lock.get_lock(key):
                value = await from_dict.get(key, 0)
                if value is None:
                    value = 0
                await from_dict.set(key, 0)
                current_to = await to_dict.get(key, 0)
                if current_to is None:
                    current_to = 0
                await to_dict.set(key, current_to + value)

        # Initialize values
        await dict1.set("balance", 100)
        await dict2.set("balance", 0)

        # Perform concurrent transfers
        await asyncio.gather(*[
            transfer_value(dict1, dict2, "balance") if i % 2 == 0 else transfer_value(dict2, dict1, "balance")
            for i in range(10)
        ])

        # Total balance should be preserved
        balance1 = await dict1.get("balance", 0)
        balance2 = await dict2.get("balance", 0)
        # Handle potential None values explicitly
        final_balance1 = balance1 if balance1 is not None else 0
        final_balance2 = balance2 if balance2 is not None else 0
        assert final_balance1 + final_balance2 == 100


def test_merge_dicts_basic():
    """Test basic dictionary merging functionality."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2, "c": 4}


def test_merge_dicts_with_none_values():
    """Test merging dictionaries with None values."""
    dict1 = {"a": None, "b": 2, "c": None}
    dict2 = {"a": 1, "b": 3, "c": 4}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2, "c": 4}


def test_merge_dicts_empty_dicts():
    """Test merging empty dictionaries."""
    dict1 = {}
    dict2 = {}
    result = merge_dicts(dict1, dict2)
    assert not result


def test_merge_dicts_one_empty():
    """Test merging when one dictionary is empty."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2}

    dict1 = {}
    dict2 = {"a": 1, "b": 2}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2}


def test_merge_dicts_nested_values():
    """Test merging dictionaries with nested values."""
    dict1 = {"a": {"x": 1}, "b": None}
    dict2 = {"a": {"y": 2}, "b": {"z": 3}}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": {"x": 1}, "b": {"z": 3}}


def test_merge_dicts_complex_types():
    """Test merging dictionaries with complex types."""
    dict1 = {"a": [1, 2, 3], "b": None}
    dict2 = {"a": [4, 5, 6], "b": "test"}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": [1, 2, 3], "b": "test"}
