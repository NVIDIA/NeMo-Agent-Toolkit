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

from unittest.mock import AsyncMock

import pytest

from nat.memory.models import MemoryItem
from nat.plugins.mem0ai.mem0_editor import Mem0Editor
from nat.utils.exception_handlers.automatic_retries import patch_with_retry


@pytest.fixture(name="mock_mem0_client")
def mock_mem0_client_fixture() -> AsyncMock:
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


@pytest.fixture(name="run_scoped_memory_item")
def run_scoped_memory_item_fixture(sample_memory_item: MemoryItem) -> MemoryItem:
    """Fixture to provide a MemoryItem whose metadata carries a Mem0 run ID."""
    return sample_memory_item.model_copy(update={"metadata": {"run_id": "run-1", "key1": "value1"}})


async def test_add_items_keeps_run_id_in_item_metadata(mem0_editor: Mem0Editor,
                                                       mock_mem0_client: AsyncMock,
                                                       run_scoped_memory_item: MemoryItem):
    """Every add receives the run ID, and the caller's item metadata is left unchanged."""
    await mem0_editor.add_items([run_scoped_memory_item, run_scoped_memory_item])

    assert run_scoped_memory_item.metadata == {"run_id": "run-1", "key1": "value1"}
    assert mock_mem0_client.add.call_count == 2
    for call in mock_mem0_client.add.call_args_list:
        assert call.kwargs["run_id"] == "run-1"
        assert call.kwargs["metadata"] == {"key1": "value1"}


async def test_add_items_retry_keeps_run_id(mock_mem0_client: AsyncMock, run_scoped_memory_item: MemoryItem):
    """A retried add (as configured by ``do_auto_retry``) still sends the run ID."""
    mock_mem0_client.add.side_effect = [RuntimeError("503 Service Unavailable"), None]
    editor = patch_with_retry(Mem0Editor(mem0_client=mock_mem0_client), retries=2, base_delay=0)

    await editor.add_items([run_scoped_memory_item])

    assert mock_mem0_client.add.call_count == 2
    assert [call.kwargs["run_id"] for call in mock_mem0_client.add.call_args_list] == ["run-1", "run-1"]


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
