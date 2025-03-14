# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import AsyncMock

import pytest
from mem0 import AsyncMemoryClient

from aiq.memory.models import MemoryItem
from aiq.plugins.mem0ai.mem0_editor import Mem0Editor


@pytest.fixture(name="mock_mem0_client")
def mock_mem0_client_fixture() -> AsyncMemoryClient:
    """Fixture to provide a mocked AsyncMemoryClient."""
    return AsyncMock()


@pytest.fixture(name="mem0_editor")
def mem0_editor_fixture(mock_mem0_client: AsyncMock):
    """Fixture to provide an instance of Mem0Editor with a mocked client."""
    return Mem0Editor(mem0_client=mock_mem0_client)


@pytest.fixture(name="sample_memory_item")
def sample_memory_item_fixture():
    """Fixture to provide a sample MemoryItem."""

    conversation = [
        {
            "role": "user",
            "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts.",
        },
        {
            "role": "assistant",
            "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy.",
        },
    ]

    return MemoryItem(conversation=conversation,
                      user_id="user123",
                      memory="Sample memory",
                      metadata={"key1": "value1"},
                      tags=["tag1", "tag2"])


async def test_add_items_success(mem0_editor: Mem0Editor, mock_mem0_client: AsyncMock, sample_memory_item: MemoryItem):
    """Test adding multiple MemoryItem objects successfully."""
    items = [sample_memory_item, sample_memory_item]
    await mem0_editor.add_items(items)

    assert mock_mem0_client.add.call_count == len(items)
    mock_mem0_client.add.assert_any_call(sample_memory_item.conversation,
                                         user_id=sample_memory_item.user_id,
                                         run_id=None,
                                         tags=sample_memory_item.tags,
                                         metadata=sample_memory_item.metadata,
                                         output_format="v1.1")


async def test_add_items_empty_list(mem0_editor: Mem0Editor, mock_mem0_client: AsyncMock):
    """Test adding an empty list of MemoryItem objects."""
    await mem0_editor.add_items([])

    mock_mem0_client.add.assert_not_called()


async def test_search_success(mem0_editor: Mem0Editor, mock_mem0_client: AsyncMock):
    """Test searching with a valid query and user ID."""
    mock_mem0_client.search.return_value = {
        "results": [{
            "input": [{
                "role": "system", "content": "Hello"
            }, {
                "role": "system", "content": "Hi"
            }],
            "memory": "Sample memory",
            "categories": ["tag1", "tag2"],
            "metadata": {
                "key1": "value1"
            }
        }]
    }

    result = await mem0_editor.search(query="test query", user_id="user123", top_k=1)

    assert len(result) == 1
    assert result[0].conversation == [{"role": "system", "content": "Hello"}, {"role": "system", "content": "Hi"}]
    assert result[0].memory == "Sample memory"
    assert result[0].tags == ["tag1", "tag2"]
    assert result[0].metadata == {"key1": "value1"}


async def test_search_missing_user_id(mem0_editor: Mem0Editor):
    """Test searching without providing a user ID."""
    with pytest.raises(KeyError, match="user_id"):
        await mem0_editor.search(query="test query")


async def test_remove_items_by_memory_id(mem0_editor: Mem0Editor, mock_mem0_client: AsyncMock):
    """Test removing items by memory ID."""
    await mem0_editor.remove_items(memory_id="memory123")

    mock_mem0_client.delete.assert_called_once_with("memory123")


async def test_remove_items_by_user_id(mem0_editor: Mem0Editor, mock_mem0_client: AsyncMock):
    """Test removing all items for a specific user ID."""
    await mem0_editor.remove_items(user_id="user123")

    mock_mem0_client.delete_all.assert_called_once_with(user_id="user123")


async def test_remove_items_missing_arguments(mem0_editor: Mem0Editor):
    """Test removing items with missing required arguments."""
    result = await mem0_editor.remove_items()

    assert result is None
