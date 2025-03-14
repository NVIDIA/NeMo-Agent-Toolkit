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

from __future__ import annotations

import asyncio

from zep_cloud.client import AsyncZep
from zep_cloud.types import Message

from aiq.memory.interfaces import MemoryEditor
from aiq.memory.models import MemoryItem


class ZepEditor(MemoryEditor):
    """
    Wrapper class that implements AgentIQ Interfaces for Zep Integrations Async.
    """

    def __init__(self, zep_client: AsyncZep):
        """
        Initialize class with Predefined Mem0 Client.

        Args:
        zep_client (AsyncZep): Async client instance.
        """
        self._client = zep_client

    async def add_items(self, items: list[MemoryItem]) -> None:
        """
        Insert Multiple MemoryItems into the memory. Each MemoryItem is translated and uploaded.
        """

        coroutines = []

        # Iteratively insert memories into Mem0
        for memory_item in items:
            conversation = memory_item.conversation
            session_id = memory_item.user_id
            messages = []
            for msg in conversation:
                messages.append(Message(content=msg["content"], role_type=msg["role"]))

            coroutines.append(self._client.memory.add(session_id=session_id, messages=messages))

        await asyncio.gather(*coroutines)

    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
        """
        Retrieve items relevant to the given query.

        Args:
            query (str): The query string to match.
            top_k (int): Maximum number of items to return.
            **kwargs: Other keyword arguments for search.

        Returns:
            list[MemoryItem]: The most relevant MemoryItems for the given query.
        """

        session_id = kwargs.pop("user_id")  # Ensure user ID is in keyword arguments
        limit = top_k

        search_result = await self._client.memory.search_sessions(session_ids=[session_id],
                                                                  text=query,
                                                                  limit=limit,
                                                                  search_scope="messages",
                                                                  **kwargs)

        # Construct MemoryItem instances
        memories = []

        for res in search_result.results:
            memories.append(
                MemoryItem(conversation=[],
                           user_id=session_id,
                           memory=res.message.content,
                           metadata={
                               "relevance_score": res.score,
                               "created_at": res.message.created_at,
                               "updated_at": res.message.updated_at
                           }))

        return memories

    async def remove_items(self, **kwargs):

        if "session_id" in kwargs:
            session_id = kwargs.pop("session_id")
            await self._client.memory.delete(session_id)

        else:
            raise ValueError("session_id not provided as part of the tool call. ")
