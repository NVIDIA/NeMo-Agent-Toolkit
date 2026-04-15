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
from unittest.mock import MagicMock

import pytest

from nat.memory.models import MemoryItem
from nat.plugins.agent_memory_server.agent_memory_editor import AgentMemoryServerEditor


@pytest.fixture(name="mock_client")
def mock_client_fixture() -> AsyncMock:
    """Fixture to provide a mocked agent-memory-client (MemoryAPIClient)."""
    client = AsyncMock()
    client.create_long_term_memory = AsyncMock()
    client.search_long_term_memory = AsyncMock()
    return client


@pytest.fixture(name="agent_memory_editor")
def agent_memory_editor_fixture(mock_client: AsyncMock):
    """Fixture to provide an instance of AgentMemoryServerEditor with a mocked client."""
    return AgentMemoryServerEditor(client=mock_client)


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
    return MemoryItem(
        conversation=conversation,
        user_id="user123",
        memory="Sample memory",
        metadata={"key1": "value1"},
        tags=["tag1", "tag2"],
    )


async def test_add_items_success(
    agent_memory_editor: AgentMemoryServerEditor,
    mock_client: AsyncMock,
    sample_memory_item: MemoryItem,
):
    """Test adding multiple MemoryItem objects successfully."""
    items = [sample_memory_item]
    await agent_memory_editor.add_items(items)

    mock_client.create_long_term_memory.assert_called_once()
    call_args = mock_client.create_long_term_memory.call_args[0]
    records = call_args[0]
    assert len(records) == 1
    assert records[0].text == sample_memory_item.memory
    assert records[0].user_id == sample_memory_item.user_id
    assert list(records[0].topics) == sample_memory_item.tags


async def test_add_items_uses_conversation_when_memory_empty(
    agent_memory_editor: AgentMemoryServerEditor,
    mock_client: AsyncMock,
):
    """Test that add_items derives text from conversation when memory field is empty."""
    item = MemoryItem(
        conversation=[{
            "role": "user", "content": "I love pizza."
        }],
        user_id="user456",
        memory=None,
        tags=[],
    )
    await agent_memory_editor.add_items([item])

    mock_client.create_long_term_memory.assert_called_once()
    records = mock_client.create_long_term_memory.call_args[0][0]
    assert len(records) == 1
    assert records[0].text == "I love pizza."
    assert records[0].user_id == "user456"


async def test_add_items_empty_list(
    agent_memory_editor: AgentMemoryServerEditor,
    mock_client: AsyncMock,
):
    """Test adding an empty list of MemoryItem objects."""
    await agent_memory_editor.add_items([])

    mock_client.create_long_term_memory.assert_not_called()


async def test_add_items_skips_item_with_no_text(
    agent_memory_editor: AgentMemoryServerEditor,
    mock_client: AsyncMock,
):
    """Test that add_items skips MemoryItems with no memory text or conversation content."""
    item = MemoryItem(user_id="user789", memory=None, conversation=None, tags=[])
    await agent_memory_editor.add_items([item])

    mock_client.create_long_term_memory.assert_not_called()


@pytest.mark.asyncio
async def test_search_success(
    agent_memory_editor: AgentMemoryServerEditor,
    mock_client: AsyncMock,
):
    """Test searching with a valid query and user ID."""
    mock_memory = MagicMock()
    mock_memory.text = "Sample memory"
    mock_memory.dist = 0.25
    mock_memory.metadata = {"key1": "value1"}
    mock_memory.topics = ["tag1", "tag2"]

    mock_results = MagicMock()
    mock_results.memories = [mock_memory]
    mock_client.search_long_term_memory.return_value = mock_results

    result = await agent_memory_editor.search(
        query="test query",
        top_k=5,
        user_id="user123",
    )

    assert len(result) == 1
    assert result[0].memory == "Sample memory"
    assert result[0].user_id == "user123"
    assert result[0].similarity_score == 0.25
    assert result[0].tags == ["tag1", "tag2"]
    assert result[0].metadata == {"key1": "value1"}
    assert result[0].conversation == [{"role": "user", "content": "Sample memory"}]

    mock_client.search_long_term_memory.assert_called_once()
    call_kwargs = mock_client.search_long_term_memory.call_args[1]
    assert call_kwargs["text"] == "test query"
    assert call_kwargs["limit"] == 5
    assert call_kwargs["user_id"].eq == "user123"


async def test_search_topics_string_becomes_single_tag(
    agent_memory_editor: AgentMemoryServerEditor,
    mock_client: AsyncMock,
):
    """Topics returned as a string must become one tag, not a list of characters."""
    mock_memory = MagicMock()
    mock_memory.text = "Sample memory"
    mock_memory.dist = None
    mock_memory.metadata = {}
    mock_memory.topics = "single-topic"

    mock_results = MagicMock()
    mock_results.memories = [mock_memory]
    mock_client.search_long_term_memory.return_value = mock_results

    result = await agent_memory_editor.search(
        query="test query",
        top_k=5,
        user_id="user123",
    )

    assert len(result) == 1
    assert result[0].tags == ["single-topic"]


async def test_search_missing_user_id(agent_memory_editor: AgentMemoryServerEditor):
    """Test searching without providing a user ID raises ValueError."""
    with pytest.raises(ValueError, match="user_id"):
        await agent_memory_editor.search(query="test query", top_k=5)


async def test_search_empty_memories_returns_empty_list(
    agent_memory_editor: AgentMemoryServerEditor,
    mock_client: AsyncMock,
):
    """Test that search returns empty list when server returns no memories."""
    mock_results = MagicMock()
    mock_results.memories = []
    mock_client.search_long_term_memory.return_value = mock_results

    result = await agent_memory_editor.search(
        query="test query",
        top_k=5,
        user_id="user123",
    )

    assert result == []


async def test_remove_items_by_memory_id(
    agent_memory_editor: AgentMemoryServerEditor,
    mock_client: AsyncMock,
):
    """Test removing items by memory ID when client has delete_long_term_memory."""
    mock_client.delete_long_term_memory = AsyncMock()

    await agent_memory_editor.remove_items(memory_id="mem-123")

    mock_client.delete_long_term_memory.assert_called_once_with("mem-123")


async def test_remove_items_by_user_id(
    agent_memory_editor: AgentMemoryServerEditor,
    mock_client: AsyncMock,
):
    """Test removing all items for a user when client has forget."""
    mock_client.forget = AsyncMock()

    await agent_memory_editor.remove_items(user_id="user123")

    mock_client.forget.assert_called_once_with(user_id="user123")


async def test_remove_items_no_op_when_no_methods():
    """Test remove_items with user_id when client has neither delete_long_term_memory nor forget."""
    # Use a client that only has create/search so hasattr(..., "delete_long_term_memory") and
    # hasattr(..., "forget") are False.
    client = MagicMock(spec=["create_long_term_memory", "search_long_term_memory"])
    client.create_long_term_memory = AsyncMock()
    client.search_long_term_memory = AsyncMock()
    editor = AgentMemoryServerEditor(client=client)

    # Should not raise; editor logs warning and returns
    await editor.remove_items(user_id="user123")
