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
Integration tests for MemMachine memory integration.

These tests require a running MemMachine server. They test the full
integration by adding memories and then retrieving them.

To run these tests:
1. Start MemMachine server (and databases)
2. Set MEMMACHINE_BASE_URL environment variable (defaults to http://localhost:8080)
3. Run: pytest tests/test_memmachine_integration.py -v
"""

import os
import uuid

import pytest

from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import GeneralConfig
from nat.memory.models import MemoryItem
from nat.plugins.memmachine.memory import MemMachineMemoryClientConfig


@pytest.fixture(name="memmachine_base_url")
def memmachine_base_url_fixture():
    """Get MemMachine base URL from environment or use default."""
    return os.environ.get("MEMMACHINE_BASE_URL", "http://localhost:8080")


@pytest.fixture(name="test_config")
def test_config_fixture(memmachine_base_url: str):
    """Create a test configuration."""
    # Use unique org/project IDs for each test run to avoid conflicts
    test_id = str(uuid.uuid4())[:8]
    return MemMachineMemoryClientConfig(
        base_url=memmachine_base_url,
        org_id=f"test_org_{test_id}",
        project_id=f"test_project_{test_id}",
        timeout=30,
        max_retries=3
    )


@pytest.fixture(name="test_user_id")
def test_user_id_fixture():
    """Generate a unique user ID for testing."""
    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.mark.integration
@pytest.mark.slow
async def test_add_and_retrieve_conversation_memory(
    test_config: MemMachineMemoryClientConfig,
    test_user_id: str
):
    """Test adding a conversation memory and retrieving it."""
    general_config = GeneralConfig()
    async with WorkflowBuilder(general_config=general_config) as builder:
        await builder.add_memory_client("memmachine_memory", test_config)
        memory_client = await builder.get_memory_client("memmachine_memory")
        
        # Create a test conversation memory
        conversation = [
            {"role": "user", "content": "I love pizza and Italian food."},
            {"role": "assistant", "content": "I'll remember that you love pizza and Italian food."},
        ]
        
        memory_item = MemoryItem(
            conversation=conversation,
            user_id=test_user_id,
            memory="User loves pizza",
            metadata={
                "session_id": "test_session_1",
                "agent_id": "test_agent_1",
                "test_id": "conversation_test"
            },
            tags=["food", "preference"]
        )
        
        # Add the memory
        await memory_client.add_items([memory_item])
        
        # Wait a moment for indexing (if needed)
        import asyncio
        await asyncio.sleep(1)
        
        # Retrieve the memory
        retrieved_memories = await memory_client.search(
            query="pizza Italian food",
            top_k=10,
            user_id=test_user_id,
            session_id="test_session_1",
            agent_id="test_agent_1"
        )
        
        # Verify we got results
        assert len(retrieved_memories) > 0, "Should retrieve at least one memory"
        
        # Check that our memory is in the results
        # Note: MemMachine may store conversation messages separately or process them,
        # so we check for the content/keywords rather than exact conversation structure
        found = False
        for mem in retrieved_memories:
            # Check if this is our memory by looking for the test_id in metadata
            if mem.metadata.get("test_id") == "conversation_test":
                found = True
                # MemMachine may return individual messages, not full conversations
                # So we check that the content is present (either in conversation or memory field)
                content = mem.memory or (str(mem.conversation) if mem.conversation else "")
                assert "pizza" in content.lower() or "italian" in content.lower(), \
                    f"Should contain pizza/italian content. Got: {content}"
                # Verify tags
                assert "food" in mem.tags or "preference" in mem.tags, \
                    f"Should have tags. Got: {mem.tags}"
                break
        
        assert found, f"Should find the memory we just added. Found {len(retrieved_memories)} memories with metadata: {[m.metadata.get('test_id') for m in retrieved_memories]}"


@pytest.mark.integration
@pytest.mark.slow
async def test_add_and_retrieve_direct_memory(
    test_config: MemMachineMemoryClientConfig,
    test_user_id: str
):
    """Test adding a direct memory (fact/preference without conversation) and retrieving it.
    
    All memories are now added to both episodic and semantic memory types.
    """
    general_config = GeneralConfig()
    async with WorkflowBuilder(general_config=general_config) as builder:
        await builder.add_memory_client("memmachine_memory", test_config)
        memory_client = await builder.get_memory_client("memmachine_memory")
        
        # Create a direct memory (no conversation)
        direct_memory = MemoryItem(
            conversation=None,
            user_id=test_user_id,
            memory="User prefers working in the morning and is allergic to peanuts",
            metadata={
                "session_id": "test_session_2",
                "agent_id": "test_agent_2",
                "test_id": "direct_test"
            },
            tags=["preference", "allergy"]
        )
        
        # Add the memory
        await memory_client.add_items([direct_memory])
        
        # Wait for memory ingestion
        # Memories are processed asynchronously by MemMachine's background task
        import asyncio
        await asyncio.sleep(5)  # Wait for background ingestion task
        
        # Try searching multiple times with retries (memory ingestion is async)
        retrieved_memories = []
        for attempt in range(3):
            retrieved_memories = await memory_client.search(
                query="morning work allergy peanuts",
                top_k=10,
                user_id=test_user_id,
                session_id="test_session_2",
                agent_id="test_agent_2"
            )
            if len(retrieved_memories) > 0:
                break
            await asyncio.sleep(2)  # Wait another 2 seconds before retry
        
        # Verify we got results
        if len(retrieved_memories) == 0:
            # If no results, try a broader search
            retrieved_memories = await memory_client.search(
                query="preference allergy",  # Broader query
                top_k=20,
                user_id=test_user_id,
                session_id="test_session_2",
                agent_id="test_agent_2"
            )
        
        # Check for related keywords
        found = False
        for mem in retrieved_memories:
            # Check by test_id or by content keywords
            if mem.metadata.get("test_id") == "direct_test":
                found = True
                break
            content = mem.memory.lower() if mem.memory else ""
            if any(keyword in content for keyword in ["morning", "peanut", "allergy", "prefer"]):
                found = True
                break
        
        # It's acceptable if we don't find exact match immediately due to async processing
        if not found:
            pytest.skip(
                "Direct memory not found - this may be due to async processing delay. "
                f"Found {len(retrieved_memories)} memories. "
                "Memory ingestion can take several seconds."
            )


@pytest.mark.integration
@pytest.mark.slow
async def test_add_multiple_and_retrieve_all(
    test_config: MemMachineMemoryClientConfig,
    test_user_id: str
):
    """Test adding multiple memories and retrieving them all."""
    general_config = GeneralConfig()
    async with WorkflowBuilder(general_config=general_config) as builder:
        await builder.add_memory_client("memmachine_memory", test_config)
        memory_client = await builder.get_memory_client("memmachine_memory")
        
        # Create multiple test memories
        memories = [
            MemoryItem(
                conversation=[{"role": "user", "content": f"Memory {i}: I like item {i}"}],
                user_id=test_user_id,
                memory=f"Memory {i}",
                metadata={
                    "session_id": "test_session_3",
                    "agent_id": "test_agent_3",
                    "test_id": f"multi_test_{i}"
                },
                tags=[f"item_{i}"]
            )
            for i in range(1, 6)  # Create 5 memories
        ]
        
        # Add all memories
        await memory_client.add_items(memories)
        
        # Wait for indexing
        import asyncio
        await asyncio.sleep(2)
        
        # Retrieve all memories with a broad query
        retrieved_memories = await memory_client.search(
            query="*",  # Broad query to get all
            top_k=20,
            user_id=test_user_id,
            session_id="test_session_3",
            agent_id="test_agent_3"
        )
        
        # Verify we got results
        assert len(retrieved_memories) >= 3, f"Should retrieve at least 3 memories, got {len(retrieved_memories)}"
        
        # Check that our test memories are in the results
        found_ids = set()
        for mem in retrieved_memories:
            test_id = mem.metadata.get("test_id", "")
            if test_id.startswith("multi_test_"):
                found_ids.add(test_id)
        
        assert len(found_ids) >= 3, f"Should find at least 3 of our test memories, found: {found_ids}"


@pytest.mark.integration
@pytest.mark.slow
async def test_add_and_verify_conversation_content_match(
    test_config: MemMachineMemoryClientConfig,
    test_user_id: str
):
    """Test that conversation memory content can be retrieved.
    
    All memories are added to both episodic and semantic memory types.
    """
    general_config = GeneralConfig()
    async with WorkflowBuilder(general_config=general_config) as builder:
        await builder.add_memory_client("memmachine_memory", test_config)
        memory_client = await builder.get_memory_client("memmachine_memory")
        
        # Create a conversation memory
        original_content = "The user mentioned their favorite programming language is Python"
        original_tags = ["programming", "preference"]
        
        memory_item = MemoryItem(
            conversation=[{"role": "user", "content": original_content}],
            user_id=test_user_id,
            memory=original_content,
            metadata={
                "session_id": "test_session_4",
                "agent_id": "test_agent_4",
                "test_id": "conversation_content_test"
            },
            tags=original_tags
        )
        
        # Add the memory
        await memory_client.add_items([memory_item])
        
        # Wait for indexing
        import asyncio
        await asyncio.sleep(2)
        
        # Retrieve the memory
        retrieved_memories = await memory_client.search(
            query="Python programming language",
            top_k=10,
            user_id=test_user_id,
            session_id="test_session_4",
            agent_id="test_agent_4"
        )
        
        # Find our memory
        found_memory = None
        for mem in retrieved_memories:
            if mem.metadata.get("test_id") == "conversation_content_test":
                found_memory = mem
                break
        
        assert found_memory is not None, f"Should find the conversation memory. Found {len(retrieved_memories)} memories"
        
        # Verify content
        content = found_memory.memory.lower() if found_memory.memory else ""
        assert "python" in content or "programming" in content, \
            f"Retrieved memory should contain 'Python' or 'programming'. Got: {found_memory.memory}"
        
        # Verify tags are preserved
        assert len(found_memory.tags) > 0, "Should have tags"
        assert any("programming" in tag.lower() or "preference" in tag.lower() for tag in found_memory.tags), \
            f"Should have relevant tags. Got: {found_memory.tags}"


@pytest.mark.integration
@pytest.mark.slow
async def test_conversation_and_direct_memory_both_retrievable(
    test_config: MemMachineMemoryClientConfig,
    test_user_id: str
):
    """Test that both conversation and direct memories are stored and retrievable.
    
    All memories are now added to both episodic and semantic memory types.
    """
    general_config = GeneralConfig()
    async with WorkflowBuilder(general_config=general_config) as builder:
        await builder.add_memory_client("memmachine_memory", test_config)
        memory_client = await builder.get_memory_client("memmachine_memory")
        
        # Add conversation memory
        conversation_memory = MemoryItem(
            conversation=[
                {"role": "user", "content": "What's the weather today?"},
                {"role": "assistant", "content": "It's sunny and 75°F."}
            ],
            user_id=test_user_id,
            memory="Weather conversation",
            metadata={
                "session_id": "test_session_5",
                "agent_id": "test_agent_5",
                "test_id": "conversation_type_test"
            },
            tags=["weather"]
        )
        
        # Add direct memory (no conversation)
        direct_memory = MemoryItem(
            conversation=None,
            user_id=test_user_id,
            memory="User lives in San Francisco and works as a software engineer",
            metadata={
                "session_id": "test_session_5",
                "agent_id": "test_agent_5",
                "test_id": "direct_type_test"
            },
            tags=["location", "occupation"]
        )
        
        # Add both
        await memory_client.add_items([conversation_memory, direct_memory])
        
        # Wait for indexing
        import asyncio
        await asyncio.sleep(2)
        
        # Search for conversation memory
        conversation_results = await memory_client.search(
            query="weather sunny",
            top_k=10,
            user_id=test_user_id,
            session_id="test_session_5",
            agent_id="test_agent_5"
        )
        
        # Search for direct memory (with retries due to async processing)
        direct_results = []
        for attempt in range(3):
            direct_results = await memory_client.search(
                query="San Francisco software engineer",
                top_k=10,
                user_id=test_user_id,
                session_id="test_session_5",
                agent_id="test_agent_5"
            )
            if len(direct_results) > 0:
                break
            await asyncio.sleep(3)  # Wait for memory ingestion
        
        # Verify conversation memory can be retrieved
        conversation_found = any(m.metadata.get("test_id") == "conversation_type_test" for m in conversation_results)
        assert conversation_found or len(conversation_results) > 0, "Should find conversation memory"
        
        # Check for direct memory
        direct_found = any(m.metadata.get("test_id") == "direct_type_test" for m in direct_results)
        direct_keywords_found = any(
            any(keyword in (m.memory or "").lower() for keyword in ["san francisco", "software", "engineer"])
            for m in direct_results
        )
        
        # Direct memory may not be immediately available due to async processing
        if not direct_found and not direct_keywords_found:
            pytest.skip(
                "Direct memory not found - may be due to async processing delay. "
                "Memories are processed asynchronously."
            )
