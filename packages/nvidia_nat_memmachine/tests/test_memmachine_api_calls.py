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

"""
Tests that verify actual MemMachine SDK API calls are made correctly.

These tests use spies/wrappers to capture and verify:
1. The exact SDK methods called
2. The parameters passed to each method
3. The data transformations (NAT MemoryItem → MemMachine format)
4. That all memories are added to both episodic and semantic memory types
"""

from unittest.mock import Mock, patch
from typing import Any

import pytest

from nat.memory.models import MemoryItem
from nat.plugins.memmachine.memmachine_editor import MemMachineEditor


class APICallSpy:
    """Spy class to capture and verify actual SDK API calls."""
    
    def __init__(self):
        self.calls = []
        self.return_values = {}
    
    def record_call(self, method_name: str, args: tuple, kwargs: dict):
        """Record an API call."""
        self.calls.append({
            'method': method_name,
            'args': args,
            'kwargs': kwargs
        })
    
    def get_calls(self, method_name: str = None):
        """Get all calls, optionally filtered by method name."""
        if method_name:
            return [c for c in self.calls if c['method'] == method_name]
        return self.calls
    
    def assert_called_with(self, method_name: str, **expected_kwargs):
        """Assert a method was called with specific parameters."""
        calls = self.get_calls(method_name)
        assert len(calls) > 0, f"Expected {method_name} to be called, but it wasn't"
        
        for call in calls:
            call_kwargs = call['kwargs']
            # Check if all expected kwargs match
            matches = all(
                call_kwargs.get(key) == value 
                for key, value in expected_kwargs.items()
            )
            if matches:
                return call
        
        raise AssertionError(
            f"Expected {method_name} to be called with {expected_kwargs}, "
            f"but got calls: {[c['kwargs'] for c in calls]}"
        )


@pytest.fixture(name="api_spy")
def api_spy_fixture():
    """Fixture to provide an API call spy."""
    return APICallSpy()


@pytest.fixture(name="spied_memory_instance")
def spied_memory_instance_fixture(api_spy: APICallSpy):
    """Create a memory instance with spied methods."""
    mock_memory = Mock()
    
    # Wrap the add method to spy on calls
    original_add = Mock(return_value=True)
    def spied_add(*args, **kwargs):
        api_spy.record_call('add', args, kwargs)
        return original_add(*args, **kwargs)
    mock_memory.add = spied_add
    
    # Wrap the search method
    original_search = Mock(return_value={
        "episodic_memory": [],
        "semantic_memory": [],
        "episode_summary": []
    })
    def spied_search(*args, **kwargs):
        api_spy.record_call('search', args, kwargs)
        return original_search(*args, **kwargs)
    mock_memory.search = spied_search
    
    # Wrap delete methods
    original_delete_episodic = Mock(return_value=True)
    def spied_delete_episodic(*args, **kwargs):
        api_spy.record_call('delete_episodic', args, kwargs)
        return original_delete_episodic(*args, **kwargs)
    mock_memory.delete_episodic = spied_delete_episodic
    
    original_delete_semantic = Mock(return_value=True)
    def spied_delete_semantic(*args, **kwargs):
        api_spy.record_call('delete_semantic', args, kwargs)
        return original_delete_semantic(*args, **kwargs)
    mock_memory.delete_semantic = spied_delete_semantic
    
    return mock_memory


@pytest.fixture(name="spied_project")
def spied_project_fixture(spied_memory_instance: Mock, api_spy: APICallSpy):
    """Create a project instance with spied memory() method."""
    mock_project = Mock(spec=['memory', 'org_id', 'project_id'])
    
    def spied_memory(*args, **kwargs):
        api_spy.record_call('project.memory', args, kwargs)
        return spied_memory_instance
    
    mock_project.memory = spied_memory
    mock_project.org_id = "test_org"
    mock_project.project_id = "test_project"
    return mock_project


@pytest.fixture(name="spied_client")
def spied_client_fixture(spied_project: Mock, api_spy: APICallSpy):
    """Create a client instance with spied create_project and get_or_create_project methods."""
    mock_client = Mock(spec=['create_project', 'get_or_create_project', 'base_url'])
    
    def spied_create_project(*args, **kwargs):
        api_spy.record_call('create_project', args, kwargs)
        return spied_project
    
    def spied_get_or_create_project(*args, **kwargs):
        api_spy.record_call('get_or_create_project', args, kwargs)
        return spied_project
    
    mock_client.create_project = spied_create_project
    mock_client.get_or_create_project = spied_get_or_create_project
    mock_client.base_url = "http://localhost:8080"
    return mock_client


@pytest.fixture(name="editor_with_spy")
def editor_with_spy_fixture(spied_client: Mock):
    """Create an editor with spied SDK calls."""
    return MemMachineEditor(memmachine_instance=spied_client)


class TestAddItemsAPICalls:
    """Test that add_items makes correct API calls to MemMachine SDK."""
    
    async def test_add_conversation_calls_add_with_correct_parameters(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that adding a conversation calls memory.add() with correct parameters."""
        item = MemoryItem(
            conversation=[
                {"role": "user", "content": "I like pizza"},
                {"role": "assistant", "content": "Great! What's your favorite topping?"}
            ],
            user_id="user123",
            memory="User likes pizza",
            metadata={"session_id": "session1", "agent_id": "agent1"},
            tags=["food", "preference"]
        )
        
        await editor_with_spy.add_items([item])
        
        # Verify project.memory was called with correct parameters
        api_spy.assert_called_with(
            'project.memory',
            user_id="user123",
            session_id="session1",
            agent_id="agent1",
            group_id="default"
        )
        
        # Verify add was called twice (once per message)
        add_calls = api_spy.get_calls('add')
        assert len(add_calls) == 2, f"Expected 2 add calls, got {len(add_calls)}"
        
        # Verify first call (user message) - episodic by default
        user_call = next(
            (c for c in add_calls if c['kwargs'].get('role') == 'user'),
            None
        )
        assert user_call is not None, "Should have a call with role='user'"
        assert user_call['kwargs']['content'] == "I like pizza"
        assert user_call['kwargs']['role'] == "user"
        # Now uses memory_types instead of episode_type
        assert user_call['kwargs']['episode_type'] is None
        assert 'memory_types' in user_call['kwargs']
        assert 'tags' in user_call['kwargs'].get('metadata', {})
        # MemMachine SDK expects tags as comma-separated string
        assert user_call['kwargs']['metadata']['tags'] == "food, preference"
        
        # Verify second call (assistant message)
        assistant_call = next(
            (c for c in add_calls if c['kwargs'].get('role') == 'assistant'),
            None
        )
        assert assistant_call is not None, "Should have a call with role='assistant'"
        assert assistant_call['kwargs']['content'] == "Great! What's your favorite topping?"
        assert assistant_call['kwargs']['role'] == "assistant"
        # Now uses memory_types instead of episode_type
        assert assistant_call['kwargs']['episode_type'] is None
        assert 'memory_types' in assistant_call['kwargs']
    
    async def test_add_direct_memory_calls_add_with_both_types(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that direct memory (no conversation) calls add() with both memory types."""
        item = MemoryItem(
            conversation=None,
            user_id="user123",
            memory="User prefers working in the morning",
            metadata={
                "session_id": "session1",
                "agent_id": "agent1"
            },
            tags=["preference"]
        )
        
        await editor_with_spy.add_items([item])
        
        # Verify add was called with both memory types
        add_calls = api_spy.get_calls('add')
        assert len(add_calls) == 1
        assert add_calls[0]['kwargs']['content'] == "User prefers working in the morning"
        assert add_calls[0]['kwargs']['role'] == "user"
        assert add_calls[0]['kwargs']['episode_type'] is None
        # Verify memory_types contains both Episodic and Semantic
        memory_types = add_calls[0]['kwargs']['memory_types']
        assert len(memory_types) == 2, "Should have both episodic and semantic memory types"
        
        # Verify metadata includes tags (as comma-separated string)
        assert add_calls[0]['kwargs']['metadata']['tags'] == "preference"
    
    async def test_add_conversation_memory_calls_add_with_both_types(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that conversation memory calls add() with both memory types."""
        item = MemoryItem(
            conversation=[{"role": "user", "content": "Hello"}],
            user_id="user123",
            memory="Test",
            metadata={
                "session_id": "session1",
                "agent_id": "agent1"
            },
            tags=[]
        )
        
        await editor_with_spy.add_items([item])
        
        # Verify add was called with both memory types
        add_calls = api_spy.get_calls('add')
        assert len(add_calls) == 1
        assert add_calls[0]['kwargs']['content'] == "Hello"
        assert add_calls[0]['kwargs']['role'] == "user"
        assert add_calls[0]['kwargs']['episode_type'] is None
        # Verify memory_types contains both Episodic and Semantic
        memory_types = add_calls[0]['kwargs']['memory_types']
        assert len(memory_types) == 2, "Should have both episodic and semantic memory types"
    
    async def test_add_with_custom_project_org_calls_get_or_create_project(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that custom project_id/org_id triggers get_or_create_project call."""
        item = MemoryItem(
            conversation=[{"role": "user", "content": "Test"}],
            user_id="user123",
            memory="Test",
            metadata={
                "session_id": "session1",
                "agent_id": "agent1",
                "project_id": "custom_project",
                "org_id": "custom_org"
            }
        )
        
        await editor_with_spy.add_items([item])
        
        # Verify get_or_create_project was called with custom IDs
        api_spy.assert_called_with(
            'get_or_create_project',
            org_id="custom_org",
            project_id="custom_project",
            description="Project for user123"
        )
    
    async def test_add_preserves_metadata_except_special_fields(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that metadata is preserved except for special fields like session_id."""
        item = MemoryItem(
            conversation=[{"role": "user", "content": "Test"}],
            user_id="user123",
            memory="Test",
            metadata={
                "session_id": "session1",
                "agent_id": "agent1",
                "custom_field": "custom_value",
                "another_field": 123
            },
            tags=["tag1"]
        )
        
        await editor_with_spy.add_items([item])
        
        # Verify metadata in the API call
        add_calls = api_spy.get_calls('add')
        assert len(add_calls) == 1
        
        metadata = add_calls[0]['kwargs'].get('metadata', {})
        # Special fields should be removed (used for memory instance creation)
        assert 'session_id' not in metadata, "session_id should be removed from metadata"
        assert 'agent_id' not in metadata, "agent_id should be removed from metadata"
        # Custom fields should be preserved
        assert metadata['custom_field'] == "custom_value"
        assert metadata['another_field'] == 123
        # MemMachine SDK expects tags as comma-separated string
        assert metadata['tags'] == "tag1"


class TestSearchAPICalls:
    """Test that search makes correct API calls to MemMachine SDK."""
    
    async def test_search_calls_memory_search_with_correct_parameters(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy,
        spied_memory_instance: Mock
    ):
        """Verify that search calls memory.search() with correct parameters."""
        # Set up search return value
        spied_memory_instance.search.return_value = {
            "episodic_memory": [{"content": "I like pizza", "metadata": {}}],
            "semantic_memory": [],
            "episode_summary": []
        }
        
        results = await editor_with_spy.search(
            query="What do I like?",
            top_k=10,
            user_id="user123",
            session_id="session1",
            agent_id="agent1"
        )
        
        # Verify project.memory was called
        api_spy.assert_called_with(
            'project.memory',
            user_id="user123",
            session_id="session1",
            agent_id="agent1",
            group_id="default"
        )
        
        # Verify search was called with correct parameters
        api_spy.assert_called_with(
            'search',
            query="What do I like?",
            limit=10
        )
    
    async def test_search_with_custom_project_org(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy,
        spied_memory_instance: Mock
    ):
        """Verify search with custom project/org calls get_or_create_project."""
        spied_memory_instance.search.return_value = {
            "episodic_memory": [],
            "semantic_memory": [],
            "episode_summary": []
        }
        
        await editor_with_spy.search(
            query="test",
            user_id="user123",
            project_id="custom_project",
            org_id="custom_org"
        )
        
        # Verify get_or_create_project was called
        api_spy.assert_called_with(
            'get_or_create_project',
            org_id="custom_org",
            project_id="custom_project",
            description="Project for user123"
        )


class TestRemoveItemsAPICalls:
    """Test that remove_items makes correct API calls to MemMachine SDK."""
    
    async def test_remove_episodic_calls_delete_episodic(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that removing episodic memory calls delete_episodic()."""
        await editor_with_spy.remove_items(
            memory_id="episodic_123",
            memory_type="episodic",
            user_id="user123",
            session_id="session1",
            agent_id="agent1"
        )
        
        # Verify delete_episodic was called with correct ID
        api_spy.assert_called_with(
            'delete_episodic',
            episodic_id="episodic_123"
        )
    
    async def test_remove_semantic_calls_delete_semantic(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that removing semantic memory calls delete_semantic()."""
        await editor_with_spy.remove_items(
            memory_id="semantic_456",
            memory_type="semantic",
            user_id="user123",
            session_id="session1",
            agent_id="agent1"
        )
        
        # Verify delete_semantic was called with correct ID
        api_spy.assert_called_with(
            'delete_semantic',
            semantic_id="semantic_456"
        )


class TestAPICallParameterValidation:
    """Test that API calls use correct parameter names and formats."""
    
    async def test_add_uses_keyword_arguments_not_positional(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that add() is called with keyword arguments, not positional."""
        item = MemoryItem(
            conversation=[{"role": "user", "content": "Test"}],
            user_id="user123",
            memory="Test",
            metadata={"session_id": "session1", "agent_id": "agent1"}
        )
        
        await editor_with_spy.add_items([item])
        
        add_calls = api_spy.get_calls('add')
        assert len(add_calls) == 1
        
        # Verify it was called with kwargs, not positional args
        call = add_calls[0]
        assert len(call['args']) == 0, "add() should be called with keyword arguments only"
        assert 'content' in call['kwargs']
        assert 'role' in call['kwargs']
        assert 'episode_type' in call['kwargs']
    
    async def test_search_uses_limit_not_top_k(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy,
        spied_memory_instance: Mock
    ):
        """Verify that search() uses 'limit' parameter (SDK name), not 'top_k'."""
        spied_memory_instance.search.return_value = {
            "episodic_memory": [],
            "semantic_memory": [],
            "episode_summary": []
        }
        
        await editor_with_spy.search(
            query="test",
            top_k=5,  # NAT uses top_k
            user_id="user123"
        )
        
        # Verify search was called with 'limit' (SDK parameter name)
        search_calls = api_spy.get_calls('search')
        assert len(search_calls) == 1
        assert 'limit' in search_calls[0]['kwargs']
        assert search_calls[0]['kwargs']['limit'] == 5
        assert 'top_k' not in search_calls[0]['kwargs'], "SDK uses 'limit', not 'top_k'"
    
    async def test_metadata_is_dict_or_none_not_empty_dict(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that metadata is passed as dict or None, never empty dict."""
        item = MemoryItem(
            conversation=[{"role": "user", "content": "Test"}],
            user_id="user123",
            memory="Test",
            metadata={"session_id": "session1", "agent_id": "agent1"},
            tags=[]  # No tags
        )
        
        await editor_with_spy.add_items([item])
        
        add_calls = api_spy.get_calls('add')
        assert len(add_calls) == 1
        
        metadata = add_calls[0]['kwargs'].get('metadata')
        # Should be None if empty, or a dict with content
        assert metadata is None or isinstance(metadata, dict)
        if metadata is not None:
            assert len(metadata) > 0, "Metadata should not be empty dict, use None instead"


class TestDataTransformation:
    """Test that data is correctly transformed between NAT and MemMachine formats."""
    
    async def test_conversation_messages_preserved_in_order(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that conversation messages are added in the correct order."""
        item = MemoryItem(
            conversation=[
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Second message"},
                {"role": "user", "content": "Third message"}
            ],
            user_id="user123",
            memory="Test",
            metadata={"session_id": "session1", "agent_id": "agent1"}
        )
        
        await editor_with_spy.add_items([item])
        
        add_calls = api_spy.get_calls('add')
        assert len(add_calls) == 3
        
        # Verify order and content
        assert add_calls[0]['kwargs']['content'] == "First message"
        assert add_calls[0]['kwargs']['role'] == "user"
        assert add_calls[1]['kwargs']['content'] == "Second message"
        assert add_calls[1]['kwargs']['role'] == "assistant"
        assert add_calls[2]['kwargs']['content'] == "Third message"
        assert add_calls[2]['kwargs']['role'] == "user"
    
    async def test_tags_included_in_metadata(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that tags are included in the metadata dict."""
        item = MemoryItem(
            conversation=[{"role": "user", "content": "Test"}],
            user_id="user123",
            memory="Test",
            metadata={"session_id": "session1", "agent_id": "agent1"},
            tags=["tag1", "tag2", "tag3"]
        )
        
        await editor_with_spy.add_items([item])
        
        add_calls = api_spy.get_calls('add')
        assert len(add_calls) == 1
        
        metadata = add_calls[0]['kwargs'].get('metadata', {})
        assert 'tags' in metadata
        # MemMachine SDK expects tags as a comma-separated string
        assert metadata['tags'] == "tag1, tag2, tag3"
    
    async def test_empty_conversation_uses_memory_text(
        self,
        editor_with_spy: MemMachineEditor,
        api_spy: APICallSpy
    ):
        """Verify that when conversation is None, memory text is used."""
        item = MemoryItem(
            conversation=None,
            user_id="user123",
            memory="This is the memory text",
            metadata={"session_id": "session1", "agent_id": "agent1"}
        )
        
        await editor_with_spy.add_items([item])
        
        add_calls = api_spy.get_calls('add')
        assert len(add_calls) == 1
        assert add_calls[0]['kwargs']['content'] == "This is the memory text"
        assert add_calls[0]['kwargs']['role'] == "user"  # Default role

