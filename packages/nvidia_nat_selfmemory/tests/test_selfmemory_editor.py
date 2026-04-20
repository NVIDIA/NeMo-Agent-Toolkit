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

from unittest.mock import MagicMock

import pytest

from nat.memory.models import MemoryItem
from nat.plugins.selfmemory.selfmemory_editor import SelfMemoryEditor


@pytest.fixture(name="mock_backend")
def mock_backend_fixture() -> MagicMock:
    """Fixture to provide a mocked SelfMemory instance."""
    backend = MagicMock()
    backend.add.return_value = {"id": "mem_123", "success": True}
    backend.search.return_value = {"results": []}
    backend.delete.return_value = {"success": True}
    backend.delete_all.return_value = {"success": True}
    return backend


@pytest.fixture(name="editor")
def editor_fixture(mock_backend: MagicMock):
    """Fixture to provide a SelfMemoryEditor with a mocked backend."""
    return SelfMemoryEditor(backend=mock_backend)


@pytest.fixture(name="sample_memory_item")
def sample_memory_item_fixture():
    """Fixture to provide a sample MemoryItem."""
    conversation = [
        {"role": "user", "content": "I love Italian food, especially pizza."},
        {"role": "assistant", "content": "Noted! You love Italian food and pizza."},
    ]

    return MemoryItem(
        conversation=conversation,
        user_id="user123",
        memory="Loves Italian food",
        metadata={"topic_category": "preferences"},
        tags=["food", "preferences"],
    )


async def test_add_items_success(editor: SelfMemoryEditor, mock_backend: MagicMock, sample_memory_item: MemoryItem):
    """Test adding multiple MemoryItem objects successfully."""
    items = [sample_memory_item, sample_memory_item]
    await editor.add_items(items)

    assert mock_backend.add.call_count == len(items)


async def test_add_items_translates_tags(editor: SelfMemoryEditor, mock_backend: MagicMock, sample_memory_item: MemoryItem):
    """Test that tags are converted from list to comma-separated string."""
    await editor.add_items([sample_memory_item])

    call_kwargs = mock_backend.add.call_args[1]
    assert call_kwargs["tags"] == "food,preferences"


async def test_add_items_translates_metadata(editor: SelfMemoryEditor, mock_backend: MagicMock, sample_memory_item: MemoryItem):
    """Test that metadata fields are extracted to SelfMemory kwargs."""
    await editor.add_items([sample_memory_item])

    call_kwargs = mock_backend.add.call_args[1]
    assert call_kwargs["topic_category"] == "preferences"
    assert call_kwargs["user_id"] == "user123"


async def test_add_items_empty_list(editor: SelfMemoryEditor, mock_backend: MagicMock):
    """Test adding an empty list of MemoryItem objects."""
    await editor.add_items([])

    mock_backend.add.assert_not_called()


async def test_search_success(editor: SelfMemoryEditor, mock_backend: MagicMock):
    """Test searching with a valid query and user ID."""
    mock_backend.search.return_value = {
        "results": [
            {
                "id": "mem_1",
                "content": "Loves Italian food",
                "score": 0.95,
                "metadata": {
                    "data": "Loves Italian food",
                    "user_id": "user123",
                    "tags": "food,preferences",
                },
            }
        ]
    }

    result = await editor.search(query="food preferences", user_id="user123", top_k=5)

    assert len(result) == 1
    assert result[0].memory == "Loves Italian food"
    assert result[0].user_id == "user123"
    assert result[0].tags == ["food", "preferences"]

    mock_backend.search.assert_called_once_with("food preferences", user_id="user123", limit=5)


async def test_search_empty_results(editor: SelfMemoryEditor, mock_backend: MagicMock):
    """Test searching with no results."""
    mock_backend.search.return_value = {"results": []}

    result = await editor.search(query="nonexistent", user_id="user123")

    assert len(result) == 0


async def test_search_missing_user_id(editor: SelfMemoryEditor):
    """Test searching without providing a user ID."""
    with pytest.raises(KeyError, match="user_id"):
        await editor.search(query="test query")


async def test_remove_items_by_memory_id(editor: SelfMemoryEditor, mock_backend: MagicMock):
    """Test removing items by memory ID."""
    await editor.remove_items(memory_id="mem_123")

    mock_backend.delete.assert_called_once_with("mem_123")


async def test_remove_items_by_user_id(editor: SelfMemoryEditor, mock_backend: MagicMock):
    """Test removing all items for a specific user ID."""
    await editor.remove_items(user_id="user123")

    mock_backend.delete_all.assert_called_once_with(user_id="user123")


async def test_remove_items_missing_arguments(editor: SelfMemoryEditor):
    """Test removing items with missing required arguments."""
    result = await editor.remove_items()

    assert result is None
