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

import asyncio
import warnings

from pydantic.warnings import PydanticDeprecatedSince20

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
    from mem0 import AsyncMemoryClient

from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem


class Mem0Editor(MemoryEditor):
    """
    Wrapper class that implements NAT interfaces for Mem0 Integrations Async.
    """

    def __init__(self, mem0_client: AsyncMemoryClient):
        """
        Initialize class with Predefined Mem0 Client.

        Args:
        mem0_client (AsyncMemoryClient): Preinstantiated
        AsyncMemoryClient object for Mem0.
        """
        self._client = mem0_client

    async def add_items(self, items: list[MemoryItem], user_id: str, **kwargs) -> None:
        """
        Insert Multiple MemoryItems into the memory.
        Each MemoryItem is translated and uploaded.

        Args:
            items (list[MemoryItem]): The items to be added.
            user_id (str): The user ID for which to add memories.
            kwargs (dict): Provider-specific keyword arguments.
        """

        coroutines = []

        # Iteratively insert memories into Mem0
        for memory_item in items:
            item_meta = memory_item.metadata
            content = memory_item.conversation

            run_id = item_meta.pop("run_id", None)
            tags = memory_item.tags

            coroutines.append(
                self._client.add(content,
                                 user_id=user_id,
                                 run_id=run_id,
                                 tags=tags,
                                 metadata=item_meta,
                                 output_format="v1.1"))

        await asyncio.gather(*coroutines)

    async def retrieve_memory(self, query: str, user_id: str, **kwargs) -> str:
        """
        Retrieve formatted memory from Mem0 relevant to the given query.

        Formats search results into structured memory with memory content
        and optional metadata like categories.

        Args:
            query (str): The query string to match.
            user_id (str): The user ID for which to retrieve memory.
            kwargs: Mem0-specific keyword arguments.
                - top_k (int, optional): Maximum number of memories to include. Defaults to 5.
                - Other Mem0 search parameters

        Returns:
            str: Formatted memory string with relevant memories, or empty string if no results.
        """
        top_k = kwargs.pop("top_k", 5)

        search_result = await self._client.search(query, user_id=user_id, top_k=top_k, output_format="v1.1", **kwargs)

        # Return empty string if no results
        if not search_result.get("results"):
            return ""

        # Format results into context string
        context_parts = ["Relevant memories:"]

        for i, res in enumerate(search_result["results"], 1):
            memory_text = res.get("memory", "")
            categories = res.get("categories", [])

            if not memory_text:
                continue

            # Format each memory with number
            context_parts.append(f"\n{i}. {memory_text}")

            # Add categories if present
            if categories:
                context_parts.append(f"   (Categories: {', '.join(categories)})")

        # If only header remains (no actual memories), return empty string
        if len(context_parts) == 1:
            return ""

        return "\n".join(context_parts)

    async def remove_items(self, user_id: str, **kwargs):
        """
        Remove items for a specific user.

        Args:
            user_id (str): The user ID for which to remove memories.
            kwargs: Additional parameters.
                - memory_id (str): Optional specific memory ID to delete.
        """
        if "memory_id" in kwargs:
            memory_id = kwargs.pop("memory_id")
            await self._client.delete(memory_id)
        else:
            # Delete all memories for the user
            await self._client.delete_all(user_id=user_id)
