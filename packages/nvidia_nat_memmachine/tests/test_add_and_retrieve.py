# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Simple script to test adding memories and retrieving them.

This script demonstrates the full integration:
1. Adds memories using the NAT integration
2. Retrieves them back
3. Prints retrieved results to confirm API calls succeed

Usage:
    python tests/test_add_and_retrieve.py
    or
    pytest tests/test_add_and_retrieve.py
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime

import pytest
import httpx

from nat.builder.builder import Builder
from nat.memory.models import MemoryItem
from nat.plugins.memmachine.memory import MemMachineMemoryClientConfig

logger = logging.getLogger(__name__)


async def _memmachine_available(base_url: str) -> bool:
    """Return True if MemMachine server is reachable."""
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/api/v2/health")
            response.raise_for_status()
            return True
    except httpx.RequestError:
        return False


async def _run_add_and_retrieve():
    """Run add-and-retrieve flow. Skips if MemMachine server is not running. Not collected by pytest."""
    # Configuration
    base_url = os.environ.get("MEMMACHINE_BASE_URL", "http://localhost:8095")
    if not await _memmachine_available(base_url):
        pytest.skip(
            f"MemMachine server not available at {base_url}. "
            "Start the server or set MEMMACHINE_BASE_URL to run this test."
        )

    test_id = str(uuid.uuid4())[:8]

    config = MemMachineMemoryClientConfig(
        base_url=base_url,
        org_id=f"test_org_{test_id}",
        project_id=f"test_project_{test_id}",
    )
    
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    session_id = "test_session"
    agent_id = "test_agent"
    
    print("=" * 80)
    print("MemMachine Integration Test: Add and Retrieve Memories")
    print("=" * 80)
    print(f"Base URL: {base_url}")
    print(f"User ID: {user_id}")
    print(f"Org ID: {config.org_id}")
    print(f"Project ID: {config.project_id}")
    print()
    
    builder = Builder()
    
    try:
        async with builder:
            async with builder.get_memory_client("memmachine_memory", config) as memory_client:
                print("✓ Memory client initialized\n")
                
                # Test 1: Add conversation memory
                print("Test 1: Adding conversation memory...")
                conversation_memory = MemoryItem(
                    conversation=[
                        {"role": "user", "content": "I love pizza and Italian food."},
                        {"role": "assistant", "content": "I'll remember that you love pizza and Italian food."},
                    ],
                    user_id=user_id,
                    memory="User loves pizza",
                    metadata={
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "test_timestamp": datetime.now().isoformat()
                    },
                    tags=["food", "preference", "italian"]
                )
                
                await memory_client.add_items([conversation_memory])
                print("✓ Conversation memory added")
                
                # Wait a moment for indexing
                await asyncio.sleep(2)
                
                # Retrieve it
                print("\nRetrieving conversation memory...")
                retrieved = await memory_client.search(
                    query="pizza Italian food",
                    top_k=10,
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id
                )
                
                print(f"✓ Retrieved {len(retrieved)} memories")
                if retrieved:
                    print(f"  First memory: {retrieved[0].memory or str(retrieved[0].conversation)}")
                    print(f"  Tags: {retrieved[0].tags}")
                
                # Test 2: Add direct memory (no conversation)
                # All memories are added to both episodic and semantic memory types
                print("\n" + "-" * 80)
                print("Test 2: Adding direct memory...")
                direct_memory = MemoryItem(
                    conversation=None,
                    user_id=user_id,
                    memory="User prefers working in the morning and is allergic to peanuts",
                    metadata={
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "test_timestamp": datetime.now().isoformat()
                    },
                    tags=["preference", "allergy", "schedule"]
                )
                
                await memory_client.add_items([direct_memory])
                print("✓ Direct memory added")
                
                # Wait for indexing
                await asyncio.sleep(2)
                
                # Retrieve it
                print("\nRetrieving direct memory...")
                retrieved = await memory_client.search(
                    query="morning work allergy peanuts",
                    top_k=10,
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id
                )
                
                print(f"✓ Retrieved {len(retrieved)} memories")
                if retrieved:
                    for i, mem in enumerate(retrieved[:3], 1):
                        print(f"  Memory {i}: {mem.memory}")
                        print(f"    Tags: {mem.tags}")
                
                # Test 3: Add multiple memories and retrieve all
                print("\n" + "-" * 80)
                print("Test 3: Adding multiple memories...")
                multiple_memories = [
                    MemoryItem(
                        conversation=[{"role": "user", "content": f"Fact {i}: I like item {i}"}],
                        user_id=user_id,
                        memory=f"Fact {i}",
                        metadata={
                            "session_id": session_id,
                            "agent_id": agent_id,
                            "fact_number": i
                        },
                        tags=[f"fact_{i}"]
                    )
                    for i in range(1, 4)  # Add 3 memories
                ]
                
                await memory_client.add_items(multiple_memories)
                print("✓ Added 3 memories")
                
                # Wait for indexing
                await asyncio.sleep(2)
                
                # Retrieve all with broad query
                print("\nRetrieving all memories (broad search)...")
                all_memories = await memory_client.search(
                    query="*",  # Broad query
                    top_k=20,
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id
                )
                
                print(f"✓ Retrieved {len(all_memories)} total memories")
                print("\nAll memories:")
                for i, mem in enumerate(all_memories, 1):
                    content = mem.memory or (str(mem.conversation) if mem.conversation else "N/A")
                    print(f"  {i}. {content[:60]}...")
                    print(f"     Tags: {mem.tags}")
                
                print("\n" + "=" * 80)
                print("✓ All tests completed successfully!")
                print("=" * 80)
                
    except Exception as e:
        print(f"\n✗ Error: {e}")
        logger.error("Error during test execution", exc_info=True)
        raise


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_add_and_retrieve_integration():
    """Integration test for adding and retrieving memories."""
    await _run_add_and_retrieve()


if __name__ == "__main__":
    asyncio.run(_run_add_and_retrieve())
