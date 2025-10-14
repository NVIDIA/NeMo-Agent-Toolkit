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
from unittest.mock import MagicMock

import pytest
from langchain_core.embeddings import Embeddings

from nat.memory.models import MemoryItem
from nat.plugins.redis.redis_editor import RedisEditor
from nat.utils.type_utils import override


class TestEmbeddings(Embeddings):

    @override
    def embed_query(self, text: str) -> list[float]:
        if not text or len(text) == 0:
            raise ValueError("No query passed to embedding model")
        return [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        res: list[list[float]] = []
        counter = 0
        for text in texts:
            embedding = [e + counter for e in self.embed_query(text)]
            res.append(embedding)
            counter += len(embedding)
        return res


@pytest.fixture(name="mock_redis_client")
def mock_redis_client_fixture() -> AsyncMock:
    """Fixture to provide a mocked AsyncMemoryClient."""
    mock_client = AsyncMock()

    # Create a mock for the JSON commands
    mock_json = AsyncMock()
    mock_json.set = AsyncMock()
    mock_json.get = AsyncMock()

    # Set up the json() method to return our mock
    mock_client.json = MagicMock(return_value=mock_json)

    return mock_client


@pytest.fixture(name="redis_editor")
def redis_editor_fixture(mock_redis_client: AsyncMock):
    """Fixture to provide an instance of RedisEditor with a mocked client."""

    editor = RedisEditor(
        redis_client=mock_redis_client,
        key_prefix="pytest",
        embedder=TestEmbeddings(),
    )
    return editor


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
                      memory="Sample memory",
                      metadata={"key1": "value1"},
                      tags=["tag1", "tag2"])


async def test_add_items_success(redis_editor: RedisEditor,
                                 mock_redis_client: AsyncMock,
                                 sample_memory_item: MemoryItem):
    """Test adding multiple MemoryItem objects successfully."""
    items = [sample_memory_item]

    # Pass user_id as positional argument
    await redis_editor.add_items(items, "user123")

    # Verify json().set was called once
    mock_redis_client.json().set.assert_called_once()

    # Get the actual call arguments
    call_args = mock_redis_client.json().set.call_args[0]

    # First argument should be the memory key (which starts with the prefix)
    assert call_args[0].startswith("pytest:memory:")

    # Second argument should be "$"
    assert call_args[1] == "$"

    # Third argument should be the memory data
    memory_data = call_args[2]
    assert memory_data["conversation"] == sample_memory_item.conversation
    assert memory_data["user_id"] == "user123"  # Now from Context, not MemoryItem
    assert memory_data["tags"] == sample_memory_item.tags
    assert memory_data["metadata"] == sample_memory_item.metadata
    assert memory_data["memory"] == sample_memory_item.memory


async def test_add_items_empty_list(redis_editor: RedisEditor, mock_redis_client: AsyncMock):
    """Test adding an empty list of MemoryItem objects."""
    await redis_editor.add_items([], "test_user")

    mock_redis_client.json().set.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_memory_success(redis_editor: RedisEditor, mock_redis_client: AsyncMock):
    """Test retrieving formatted memory with valid query and user ID."""
    # Create mock documents with the required attributes
    mock_doc1 = MagicMock()
    mock_doc1.id = "pytest:memory:001"
    mock_doc1.score = 0.95

    mock_doc2 = MagicMock()
    mock_doc2.id = "pytest:memory:002"
    mock_doc2.score = 0.87

    # Create a mock results object with a docs attribute
    mock_results = MagicMock()
    mock_results.docs = [mock_doc1, mock_doc2]

    # Create a mock for the ft method that returns an object with the search method
    mock_ft_index = MagicMock()
    mock_ft_index.search = AsyncMock(return_value=mock_results)

    # Set up the client mock to return the ft mock
    mock_redis_client.ft = MagicMock(return_value=mock_ft_index)

    # Mock Redis JSON get to return document data for each doc
    async def mock_json_get(doc_id):
        if doc_id == "pytest:memory:001":
            return {
                "conversation": [{
                    "role": "user", "content": "I like pizza"
                }],
                "user_id": "user123",
                "tags": ["food", "preferences"],
                "metadata": {
                    "key1": "value1"
                },
                "memory": "User likes pizza"
            }
        elif doc_id == "pytest:memory:002":
            return {
                "conversation": [{
                    "role": "user", "content": "I'm vegetarian"
                }],
                "user_id": "user123",
                "tags": ["dietary"],
                "metadata": {},
                "memory": "User is vegetarian"
            }
        return {}

    mock_redis_client.json().get = AsyncMock(side_effect=mock_json_get)

    result = await redis_editor.retrieve_memory(query="test query", user_id="user123", top_k=2)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "User likes pizza" in result
    assert "User is vegetarian" in result
    assert "Tags: food, preferences" in result
    assert "Tags: dietary" in result
    assert "Similarity: 0.950" in result
    assert "Similarity: 0.870" in result


@pytest.mark.asyncio
async def test_retrieve_memory_no_results(redis_editor: RedisEditor, mock_redis_client: AsyncMock):
    """Test retrieving memory when no results are found."""
    # Create a mock results object with no docs
    mock_results = MagicMock()
    mock_results.docs = []

    # Create a mock for the ft method
    mock_ft_index = MagicMock()
    mock_ft_index.search = AsyncMock(return_value=mock_results)

    mock_redis_client.ft = MagicMock(return_value=mock_ft_index)

    result = await redis_editor.retrieve_memory(query="test query", user_id="user123")

    assert isinstance(result, str)
    assert result == ""
