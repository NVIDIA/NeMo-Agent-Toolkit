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

from unittest.mock import MagicMock, Mock

import pytest

from nat.memory.models import MemoryItem
from nat.plugins.memmachine.memmachine_editor import MemMachineEditor


@pytest.fixture(name="mock_memory_instance")
def mock_memory_instance_fixture():
    """Fixture to provide a mocked Memory instance from MemMachine SDK."""
    mock_memory = Mock()
    mock_memory.add = Mock(return_value=True)
    mock_memory.search = Mock(return_value={
        "episodic_memory": [],
        "semantic_memory": [],
        "episode_summary": []
    })
    mock_memory.delete_episodic = Mock(return_value=True)
    mock_memory.delete_semantic = Mock(return_value=True)
    return mock_memory


@pytest.fixture(name="mock_project")
def mock_project_fixture(mock_memory_instance):
    """Fixture to provide a mocked Project instance from MemMachine SDK."""
    # Use spec to restrict attributes - Project should have 'memory' but NOT 'create_project'
    # This ensures hasattr checks work correctly
    mock_project = Mock(spec=['memory', 'org_id', 'project_id'])
    mock_project.memory = Mock(return_value=mock_memory_instance)
    mock_project.org_id = "test_org"
    mock_project.project_id = "test_project"
    # Explicitly ensure create_project doesn't exist (Mock with spec will raise AttributeError)
    return mock_project


@pytest.fixture(name="mock_client")
def mock_client_fixture(mock_project):
    """Fixture to provide a mocked MemMachineClient instance."""
    # Use spec to ensure create_project and get_or_create_project exist for hasattr checks
    mock_client = Mock(spec=['create_project', 'get_or_create_project', 'base_url'])
    mock_client.create_project = Mock(return_value=mock_project)
    mock_client.get_or_create_project = Mock(return_value=mock_project)
    mock_client.base_url = "http://localhost:8080"
    return mock_client


@pytest.fixture(name="memmachine_editor_with_client")
def memmachine_editor_with_client_fixture(mock_client):
    """Fixture to provide an instance of MemMachineEditor with a mocked client."""
    return MemMachineEditor(memmachine_instance=mock_client)


@pytest.fixture(name="memmachine_editor_with_project")
def memmachine_editor_with_project_fixture(mock_project):
    """Fixture to provide an instance of MemMachineEditor with a mocked project."""
    return MemMachineEditor(memmachine_instance=mock_project)


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
        metadata={"key1": "value1", "session_id": "session456", "agent_id": "agent789"},
        tags=["tag1", "tag2"]
    )


@pytest.fixture(name="sample_memory_item_with_semantic")
def sample_memory_item_with_semantic_fixture():
    """Fixture to provide a MemoryItem configured for semantic memory."""
    return MemoryItem(
        conversation=None,
        user_id="user123",
        memory="I prefer working in the morning",
        metadata={"use_semantic_memory": True, "session_id": "session456", "agent_id": "agent789"},
        tags=["preference"]
    )


async def test_add_items_with_conversation(
    memmachine_editor_with_client: MemMachineEditor,
    mock_client: Mock,
    mock_project: Mock,
    mock_memory_instance: Mock,
    sample_memory_item: MemoryItem
):
    """Test adding MemoryItem objects with conversation successfully."""
    items = [sample_memory_item]
    await memmachine_editor_with_client.add_items(items)

    # Verify project was created/retrieved
    mock_client.get_or_create_project.assert_called_once()
    
    # Verify memory instance was created
    mock_project.memory.assert_called_once_with(
        user_id="user123",
        agent_id="agent789",
        session_id="session456",
        group_id="default"
    )
    
    # Verify add was called for each message in conversation
    # The await above should have completed all async tasks
    assert mock_memory_instance.add.call_count == 2, f"Expected 2 calls, got {mock_memory_instance.add.call_count}. Calls: {mock_memory_instance.add.call_args_list}"
    
    # Get all calls
    all_calls = mock_memory_instance.add.call_args_list
    assert len(all_calls) == 2, f"Expected 2 calls in call_args_list, got {len(all_calls)}"
    
    # Extract roles and contents from all calls
    calls_data = []
    for call in all_calls:
        if call.kwargs:
            role = call.kwargs.get("role")
            content = call.kwargs.get("content")
            episode_type = call.kwargs.get("episode_type")
            metadata = call.kwargs.get("metadata", {})
        else:
            # Handle positional args if needed
            role = None
            content = call.args[0] if call.args else None
            episode_type = None
            metadata = {}
        if role and content:
            calls_data.append({"role": role, "content": content, "episode_type": episode_type, "metadata": metadata})
    
    # Verify we have both roles
    roles = [c["role"] for c in calls_data]
    assert "user" in roles, f"Expected 'user' role in calls, got: {roles}. Calls data: {calls_data}"
    assert "assistant" in roles, f"Expected 'assistant' role in calls, got: {roles}. Calls data: {calls_data}"
    
    # Verify user message
    user_call_data = next(c for c in calls_data if c["role"] == "user")
    assert user_call_data["content"] == "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."
    # Now uses memory_types instead of episode_type
    assert user_call_data["episode_type"] is None
    assert "tags" in user_call_data["metadata"]
    
    # Verify assistant message
    assistant_call_data = next(c for c in calls_data if c["role"] == "assistant")
    assert assistant_call_data["content"] == "Hello Alex! I've noted that you're a vegetarian and have a nut allergy."
    # Now uses memory_types instead of episode_type
    assert assistant_call_data["episode_type"] is None


async def test_add_items_with_semantic_memory(
    memmachine_editor_with_client: MemMachineEditor,
    mock_client: Mock,
    mock_project: Mock,
    mock_memory_instance: Mock,
    sample_memory_item_with_semantic: MemoryItem
):
    """Test adding MemoryItem configured for semantic memory."""
    items = [sample_memory_item_with_semantic]
    await memmachine_editor_with_client.add_items(items)

    # Verify add was called
    assert mock_memory_instance.add.call_count == 1
    
    # Verify memory_types is used (episode_type is None)
    call_kwargs = mock_memory_instance.add.call_args.kwargs
    assert call_kwargs["content"] == "I prefer working in the morning"
    assert call_kwargs["episode_type"] is None
    assert "memory_types" in call_kwargs
    assert call_kwargs["role"] == "user"


async def test_add_items_empty_list(
    memmachine_editor_with_client: MemMachineEditor,
    mock_memory_instance: Mock
):
    """Test adding an empty list of MemoryItem objects."""
    await memmachine_editor_with_client.add_items([])
    
    # Should not call add if list is empty
    mock_memory_instance.add.assert_not_called()


async def test_add_items_with_memory_text_only(
    memmachine_editor_with_client: MemMachineEditor,
    mock_client: Mock,
    mock_project: Mock,
    mock_memory_instance: Mock
):
    """Test adding MemoryItem with only memory text (no conversation)."""
    item = MemoryItem(
        conversation=None,
        user_id="user123",
        memory="This is a standalone memory",
        metadata={"session_id": "session456", "agent_id": "agent789"},
        tags=[]
    )
    
    await memmachine_editor_with_client.add_items([item])
    
    # Verify add was called once
    assert mock_memory_instance.add.call_count == 1
    
    # Verify memory_types is used (episode_type is None)
    call_kwargs = mock_memory_instance.add.call_args.kwargs
    assert call_kwargs["content"] == "This is a standalone memory"
    assert call_kwargs["episode_type"] is None
    assert "memory_types" in call_kwargs


async def test_search_success(
    memmachine_editor_with_client: MemMachineEditor,
    mock_client: Mock,
    mock_project: Mock,
    mock_memory_instance: Mock
):
    """Test searching with a valid query and user ID."""
    # Mock search results with the new nested structure
    # MemMachine SDK returns SearchResult with content containing nested episodic_memory
    mock_search_result = Mock()
    mock_search_result.content = {
        "episodic_memory": {
            "long_term_memory": {
                "episodes": [
                    {
                        "content": "I like pizza",
                        "metadata": {"key1": "value1", "tags": "food"}
                    }
                ]
            },
            "short_term_memory": {
                "episodes": []
            }
        },
        "semantic_memory": [
            {
                "feature": "User prefers Italian food",
                "metadata": {"key2": "value2"}
            }
        ]
    }
    mock_memory_instance.search.return_value = mock_search_result
    
    result = await memmachine_editor_with_client.search(
        query="What do I like to eat?",
        top_k=5,
        user_id="user123",
        session_id="session456",
        agent_id="agent789"
    )
    
    # Verify search was called
    mock_memory_instance.search.assert_called_once_with(query="What do I like to eat?", limit=5)
    
    # Verify results
    assert len(result) == 2  # One episodic + one semantic
    assert result[0].memory == "I like pizza"
    assert result[0].tags == ["food"]
    assert result[1].memory == "User prefers Italian food"


async def test_search_with_string_tags(
    memmachine_editor_with_client: MemMachineEditor,
    mock_memory_instance: Mock
):
    """Test searching when tags come back as comma-separated string from SDK."""
    # Mock search results with the new nested structure
    mock_search_result = Mock()
    mock_search_result.content = {
        "episodic_memory": {
            "long_term_memory": {
                "episodes": [
                    {
                        "content": "I like pizza and pasta",
                        "metadata": {"tags": "food, preference, italian"}  # String format
                    }
                ]
            },
            "short_term_memory": {
                "episodes": []
            }
        },
        "semantic_memory": []
    }
    mock_memory_instance.search.return_value = mock_search_result
    
    result = await memmachine_editor_with_client.search(
        query="What do I like?",
        top_k=5,
        user_id="user123"
    )
    
    assert len(result) == 1
    # Tags should be converted from string to list
    assert result[0].tags == ["food", "preference", "italian"]


async def test_search_empty_results(
    memmachine_editor_with_client: MemMachineEditor,
    mock_memory_instance: Mock
):
    """Test searching with empty results."""
    mock_search_result = Mock()
    mock_search_result.content = {
        "episodic_memory": {
            "long_term_memory": {"episodes": []},
            "short_term_memory": {"episodes": []}
        },
        "semantic_memory": []
    }
    mock_memory_instance.search.return_value = mock_search_result
    
    result = await memmachine_editor_with_client.search(
        query="test query",
        top_k=5,
        user_id="user123"
    )
    
    assert len(result) == 0


async def test_search_missing_user_id(memmachine_editor_with_client: MemMachineEditor):
    """Test searching without providing a user ID."""
    with pytest.raises(KeyError, match="user_id"):
        await memmachine_editor_with_client.search(query="test query")


async def test_search_with_defaults(
    memmachine_editor_with_client: MemMachineEditor,
    mock_memory_instance: Mock
):
    """Test searching with default session_id and agent_id."""
    mock_search_result = Mock()
    mock_search_result.content = {
        "episodic_memory": {
            "long_term_memory": {"episodes": []},
            "short_term_memory": {"episodes": []}
        },
        "semantic_memory": []
    }
    mock_memory_instance.search.return_value = mock_search_result
    
    await memmachine_editor_with_client.search(
        query="test query",
        user_id="user123"
    )
    
    # Verify memory instance was created with defaults
    # The editor should use default_session and default_agent
    mock_memory_instance.search.assert_called_once()


async def test_remove_items_by_memory_id_episodic(
    memmachine_editor_with_client: MemMachineEditor,
    mock_client: Mock,
    mock_project: Mock,
    mock_memory_instance: Mock
):
    """Test removing items by episodic memory ID."""
    await memmachine_editor_with_client.remove_items(
        memory_id="episodic_123",
        memory_type="episodic",
        user_id="user123",
        session_id="session456",
        agent_id="agent789"
    )
    
    # Verify delete_episodic was called
    mock_memory_instance.delete_episodic.assert_called_once_with(episodic_id="episodic_123")


async def test_remove_items_by_memory_id_semantic(
    memmachine_editor_with_client: MemMachineEditor,
    mock_client: Mock,
    mock_project: Mock,
    mock_memory_instance: Mock
):
    """Test removing items by semantic memory ID."""
    await memmachine_editor_with_client.remove_items(
        memory_id="semantic_123",
        memory_type="semantic",
        user_id="user123",
        session_id="session456",
        agent_id="agent789"
    )
    
    # Verify delete_semantic was called
    mock_memory_instance.delete_semantic.assert_called_once_with(semantic_id="semantic_123")


async def test_remove_items_by_memory_id_without_user_id(
    memmachine_editor_with_client: MemMachineEditor
):
    """Test that removing items by memory_id without user_id raises ValueError."""
    with pytest.raises(ValueError, match="user_id is required"):
        await memmachine_editor_with_client.remove_items(memory_id="episodic_123")


async def test_remove_items_by_user_id_not_implemented(
    memmachine_editor_with_client: MemMachineEditor
):
    """Test that removing all items by user_id raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Bulk deletion by user_id"):
        await memmachine_editor_with_client.remove_items(user_id="user123")


async def test_editor_with_project_instance(
    memmachine_editor_with_project: MemMachineEditor,
    mock_project: Mock,
    mock_memory_instance: Mock,
    sample_memory_item: MemoryItem
):
    """Test that editor works correctly when initialized with a Project instance."""
    items = [sample_memory_item]
    await memmachine_editor_with_project.add_items(items)
    
    # Verify project.memory was called directly (not create_project)
    mock_project.memory.assert_called_once()
    
    # Verify add was called
    assert mock_memory_instance.add.call_count == 2


async def test_add_items_with_custom_project_and_org(
    memmachine_editor_with_client: MemMachineEditor,
    mock_client: Mock,
    mock_project: Mock,
    mock_memory_instance: Mock
):
    """Test adding items with custom project_id and org_id in metadata."""
    item = MemoryItem(
        conversation=[{"role": "user", "content": "Test"}],
        user_id="user123",
        memory="Test memory",
        metadata={
            "session_id": "session456",
            "agent_id": "agent789",
            "project_id": "custom_project",
            "org_id": "custom_org"
        }
    )
    
    await memmachine_editor_with_client.add_items([item])
    
    # Verify project was created/retrieved with custom org_id and project_id
    mock_client.get_or_create_project.assert_called_once_with(
        org_id="custom_org",
        project_id="custom_project",
        description="Project for user123"
    )


async def test_search_with_custom_project_and_org(
    memmachine_editor_with_client: MemMachineEditor,
    mock_client: Mock,
    mock_project: Mock,
    mock_memory_instance: Mock
):
    """Test searching with custom project_id and org_id."""
    mock_memory_instance.search.return_value = {
        "episodic_memory": [],
        "semantic_memory": [],
        "episode_summary": []
    }
    
    await memmachine_editor_with_client.search(
        query="test",
        user_id="user123",
        project_id="custom_project",
        org_id="custom_org"
    )
    
    # Verify project was created/retrieved with custom IDs
    mock_client.get_or_create_project.assert_called_once_with(
        org_id="custom_org",
        project_id="custom_project",
        description="Project for user123"
    )
