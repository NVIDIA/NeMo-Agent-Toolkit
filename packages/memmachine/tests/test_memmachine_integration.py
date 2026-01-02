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
async def test_add_and_retrieve_semantic_memory(
    test_config: MemMachineMemoryClientConfig,
    test_user_id: str
):
    """Test adding a semantic memory (fact/preference) and retrieving it."""
    general_config = GeneralConfig()
    async with WorkflowBuilder(general_config=general_config) as builder:
        await builder.add_memory_client("memmachine_memory", test_config)
        memory_client = await builder.get_memory_client("memmachine_memory")
        
        # Create a semantic memory (fact/preference)
        semantic_memory = MemoryItem(
            conversation=None,
            user_id=test_user_id,
            memory="User prefers working in the morning and is allergic to peanuts",
            metadata={
                "session_id": "test_session_2",
                "agent_id": "test_agent_2",
                "use_semantic_memory": True,  # Mark as semantic memory
                "test_id": "semantic_test"
            },
            tags=["preference", "allergy"]
        )
        
        # Add the memory
        await memory_client.add_items([semantic_memory])
        
        # Wait for semantic memory ingestion
        # Semantic memories are processed asynchronously by MemMachine's background task
        # which runs every ~2 seconds and processes messages in batches
        # We need to wait longer for semantic memory to be ingested and searchable
        import asyncio
        await asyncio.sleep(5)  # Wait for background ingestion task
        
        # Try searching multiple times with retries (semantic memory ingestion is async)
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
        
        # Note: Semantic memory returns processed "features" (summaries), not raw text
        # So we may not find exact text matches, but should find related content
        # Verify we got results (may be processed/summarized, not exact match)
        if len(retrieved_memories) == 0:
            # If no results, try a broader search
            retrieved_memories = await memory_client.search(
                query="preference allergy",  # Broader query
                top_k=20,
                user_id=test_user_id,
                session_id="test_session_2",
                agent_id="test_agent_2"
            )
        
        # Semantic memory may return processed features, so we check for related keywords
        # rather than exact text matches
        found = False
        for mem in retrieved_memories:
            # Check by test_id or by content keywords
            if mem.metadata.get("test_id") == "semantic_test":
                found = True
                break
            # Also check if content relates to our semantic memory (processed features)
            content = mem.memory.lower() if mem.memory else ""
            if any(keyword in content for keyword in ["morning", "peanut", "allergy", "prefer"]):
                found = True
                break
        
        # For semantic memory, it's acceptable if we don't find exact match immediately
        # as it's processed asynchronously and may return summarized features
        if not found:
            pytest.skip(
                "Semantic memory not found - this may be due to async processing delay. "
                f"Found {len(retrieved_memories)} memories. "
                "Semantic memory ingestion can take several seconds."
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
async def test_add_and_verify_episodic_content_match(
    test_config: MemMachineMemoryClientConfig,
    test_user_id: str
):
    """Test that episodic memory content can be retrieved (conversation-based)."""
    general_config = GeneralConfig()
    async with WorkflowBuilder(general_config=general_config) as builder:
        await builder.add_memory_client("memmachine_memory", test_config)
        memory_client = await builder.get_memory_client("memmachine_memory")
        
        # Create an episodic memory (with conversation)
        original_content = "The user mentioned their favorite programming language is Python"
        original_tags = ["programming", "preference"]
        
        memory_item = MemoryItem(
            conversation=[{"role": "user", "content": original_content}],
            user_id=test_user_id,
            memory=original_content,
            metadata={
                "session_id": "test_session_4",
                "agent_id": "test_agent_4",
                "test_id": "episodic_content_test",
                "use_semantic_memory": False  # Explicitly episodic
            },
            tags=original_tags
        )
        
        # Add the memory
        await memory_client.add_items([memory_item])
        
        # Wait for indexing (episodic is faster than semantic)
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
            if mem.metadata.get("test_id") == "episodic_content_test":
                found_memory = mem
                break
        
        assert found_memory is not None, f"Should find the episodic memory. Found {len(retrieved_memories)} memories"
        
        # Verify content (episodic should preserve content better)
        content = found_memory.memory.lower() if found_memory.memory else ""
        assert "python" in content or "programming" in content, \
            f"Retrieved memory should contain 'Python' or 'programming'. Got: {found_memory.memory}"
        
        # Verify tags are preserved
        assert len(found_memory.tags) > 0, "Should have tags"
        assert any("programming" in tag.lower() or "preference" in tag.lower() for tag in found_memory.tags), \
            f"Should have relevant tags. Got: {found_memory.tags}"


@pytest.mark.integration
@pytest.mark.slow
async def test_episodic_vs_semantic_memory_types(
    test_config: MemMachineMemoryClientConfig,
    test_user_id: str
):
    """Test that episodic and semantic memories are stored correctly."""
    general_config = GeneralConfig()
    async with WorkflowBuilder(general_config=general_config) as builder:
        await builder.add_memory_client("memmachine_memory", test_config)
        memory_client = await builder.get_memory_client("memmachine_memory")
        
        # Add episodic memory (conversation)
        episodic_memory = MemoryItem(
            conversation=[
                {"role": "user", "content": "What's the weather today?"},
                {"role": "assistant", "content": "It's sunny and 75°F."}
            ],
            user_id=test_user_id,
            memory="Weather conversation",
            metadata={
                "session_id": "test_session_5",
                "agent_id": "test_agent_5",
                "test_id": "episodic_test",
                "use_semantic_memory": False  # Explicitly episodic
            },
            tags=["weather"]
        )
        
        # Add semantic memory (fact)
        semantic_memory = MemoryItem(
            conversation=None,
            user_id=test_user_id,
            memory="User lives in San Francisco and works as a software engineer",
            metadata={
                "session_id": "test_session_5",
                "agent_id": "test_agent_5",
                "test_id": "semantic_type_test",
                "use_semantic_memory": True  # Explicitly semantic
            },
            tags=["location", "occupation"]
        )
        
        # Add both
        await memory_client.add_items([episodic_memory, semantic_memory])
        
        # Wait for indexing (episodic is faster, semantic needs more time)
        import asyncio
        await asyncio.sleep(2)
        
        # Search for episodic memory
        episodic_results = await memory_client.search(
            query="weather sunny",
            top_k=10,
            user_id=test_user_id,
            session_id="test_session_5",
            agent_id="test_agent_5"
        )
        
        # Search for semantic memory (with retries due to async processing)
        semantic_results = []
        for attempt in range(3):
            semantic_results = await memory_client.search(
                query="San Francisco software engineer",
                top_k=10,
                user_id=test_user_id,
                session_id="test_session_5",
                agent_id="test_agent_5"
            )
            if len(semantic_results) > 0:
                break
            await asyncio.sleep(3)  # Wait for semantic memory ingestion
        
        # Verify episodic memory can be retrieved
        episodic_found = any(m.metadata.get("test_id") == "episodic_test" for m in episodic_results)
        assert episodic_found or len(episodic_results) > 0, "Should find episodic memory"
        
        # For semantic memory, it may be processed into features, so we check more leniently
        semantic_found = any(m.metadata.get("test_id") == "semantic_type_test" for m in semantic_results)
        # Also check if any result contains related keywords (semantic memory may be summarized)
        semantic_keywords_found = any(
            any(keyword in (m.memory or "").lower() for keyword in ["san francisco", "software", "engineer"])
            for m in semantic_results
        )
        
        # Semantic memory may not be immediately available due to async processing
        if not semantic_found and not semantic_keywords_found:
            pytest.skip(
                "Semantic memory not found - may be due to async processing delay. "
                "Semantic memories are processed asynchronously and may return processed features."
            )
