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
#
# This file is a part of the Nvidia AIQ Toolkit project, Mem0 Plugin, and has been modified to support the Mem0 v2 API.

import asyncio

from mem0 import AsyncMemory
from mem0 import AsyncMemoryClient

from aiq.memory.interfaces import MemoryEditor
from aiq.memory.models import MemoryItem


class Mem0Editor(MemoryEditor):
    """
    Wrapper class that implements AIQ Toolkit Interfaces for Mem0 Integrations Async.
    """

    def __init__(self, mem0_client: AsyncMemoryClient | AsyncMemory):
        """
        Initialize class with Predefined Mem0 Client.

        Args:
        mem0_client (AsyncMemoryClient | AsyncMemory): Preinstantiated
        AsyncMemoryClient or AsyncMemory object for Mem0.
        """
        self._client = mem0_client

        # Ensure the client is properly initialized
        if self._client is None:
            raise ValueError("Mem0 client cannot be None")

    async def add_items(self, items: list[MemoryItem]) -> None:
        """
        Insert Multiple MemoryItems into the memory.
        Each MemoryItem is translated and uploaded.
        """

        coroutines = []

        # Iteratively insert memories into Mem0
        for memory_item in items:
            item_meta = memory_item.metadata
            content = memory_item.conversation

            user_id = memory_item.user_id  # This must be specified
            run_id = item_meta.pop("run_id", None)

            # UPDATED: In mem0 v2 API, tags are now part of metadata
            # Moving tags into metadata dictionary
            tags = memory_item.tags
            if tags:
                item_meta["categories"] = tags

            # UPDATED: In mem0 v2 API, content is passed as messages array
            # Handle different types of content
            if isinstance(content, str):
                messages = [{"role": "user", "content": content}]
            elif isinstance(content, list) and all(isinstance(msg, dict) for msg in content):
                # If content is already in the correct format (list of message dicts)
                messages = content
            else:
                # Try to convert to string as a fallback
                try:
                    messages = [{"role": "user", "content": str(content)}]
                except Exception:
                    raise ValueError(f"Unable to convert content to a valid message format: {type(content)}")

            # UPDATED: Removed output_format parameter as it's deprecated in v2 API
            coroutines.append(self._client.add(messages, user_id=user_id, run_id=run_id, metadata=item_meta))

        await asyncio.gather(*coroutines)

    async def search(self, query: str, top_k: int = 5, **kwargs) \
            -> list[MemoryItem]:
        """
        Retrieve items relevant to the given query.

        Args:
            query (str): The query string to match.
            top_k (int): Maximum number of items to return.
            **kwargs: Other keyword arguments for search.

        Returns:
            list[MemoryItem]: The most relevant
            MemoryItems for the given query.
        """

        user_id = kwargs.pop("user_id")  # Ensure user ID is in keyword arguments

        # UPDATED: Removed output_format parameter as it's deprecated in v2 API
        search_result = await self._client.search(query, user_id=user_id, limit=top_k, **kwargs)

        memories = []

        # UPDATED: Processing search results according to v2 API structure
        # Handle both v1 and v2 API formats
        # In v1, search_result is a dict with a "results" key
        # In v2, search_result is directly a list of results
        results_to_process = search_result

        if isinstance(search_result, dict) and "results" in search_result:
            results_to_process = search_result["results"]

        for res in results_to_process:
            # Handle different result formats
            if isinstance(res, dict):
                item_meta = res.get("metadata", {})
                if isinstance(item_meta, dict):
                    # Make a copy to avoid modifying the original
                    item_meta = dict(item_meta)
                else:
                    item_meta = {}

                # UPDATED: In v2 API, tags/categories are in metadata
                tags = []
                if "categories" in item_meta:
                    tags = item_meta.pop("categories", [])
                    if not isinstance(tags, list):
                        tags = []

                memory_content = res.get("memory", "")

            elif isinstance(res, str):
                # If the result is a string, use it as the memory content
                memory_content = res
                item_meta = {}
                tags = []
            else:
                # Skip invalid results
                continue

            # Try to get the conversation from the 'input' field first (as in older versions)
            # Only try to get 'input' if res is a dictionary
            if isinstance(res, dict):
                conversation = res.get("input", [])
                # If not found or not in the expected format, construct it from memory_content
                if not conversation or not isinstance(conversation, list):
                    conversation = [{
                        "role": "user", "content": memory_content
                    }] if isinstance(memory_content, str) else memory_content
            else:
                # If res is not a dictionary, construct conversation from memory_content
                conversation = [{
                    "role": "user", "content": memory_content
                }] if isinstance(memory_content, str) else memory_content

            memories.append(
                MemoryItem(conversation=conversation,
                           user_id=user_id,
                           memory=memory_content,
                           tags=tags,
                           metadata=item_meta))

        return memories

    async def remove_items(self, **kwargs):
        if "memory_id" in kwargs:
            memory_id = kwargs.pop("memory_id")
            await self._client.delete(memory_id)

        elif "user_id" in kwargs:
            user_id = kwargs.pop("user_id")
            await self._client.delete_all(user_id=user_id)

        return
