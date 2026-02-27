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

import logging

from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem

logger = logging.getLogger(__name__)


def _memory_item_to_text(item: MemoryItem) -> str:
    """Derive storage text from MemoryItem: prefer memory field, else from conversation."""
    if item.memory:
        return item.memory
    if item.conversation:
        parts = [m.get("content", "") for m in item.conversation if isinstance(m, dict)]
        return " ".join(parts).strip() or ""
    return ""


class AgentMemoryServerEditor(MemoryEditor):
    """
    MemoryEditor implementation that uses the Redis Agent Memory Server
    (agent-memory-client) for long-term memory.
    """

    def __init__(self, client):
        """
        Initialize with an agent-memory-client instance (MemoryAPIClient or similar).

        Args:
            client: Client from create_memory_client() or MemoryAPIClient.
        """
        self._client = client

    async def add_items(self, items: list[MemoryItem]) -> None:
        """Insert MemoryItems into long-term memory via the Agent Memory Server."""
        if not items:
            return
        from agent_memory_client.models import ClientMemoryRecord
        from agent_memory_client.models import MemoryTypeEnum

        records = []
        for item in items:
            text = _memory_item_to_text(item)
            if not text:
                logger.warning("Skipping MemoryItem with no memory text or conversation content")
                continue
            record = ClientMemoryRecord(
                text=text,
                memory_type=MemoryTypeEnum.SEMANTIC,
                topics=item.tags or [],
                user_id=item.user_id,
            )
            records.append(record)
        if records:
            await self._client.create_long_term_memory(records)

    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
        """Search long-term memory; user_id should be passed in kwargs."""
        from agent_memory_client.filters import UserId

        user_id = kwargs.get("user_id")
        if user_id is None:
            raise ValueError("search() requires user_id in kwargs for Agent Memory Server")
        user_filter = UserId(eq=user_id)
        results = await self._client.search_long_term_memory(
            text=query,
            limit=top_k,
            user_id=user_filter,
            **{k: v for k, v in kwargs.items() if k != "user_id"},
        )
        out = []
        for m in getattr(results, "memories", []) or []:
            dist = getattr(m, "dist", None)
            text = getattr(m, "text", "") or ""
            meta = getattr(m, "metadata", None) or {}
            tags = getattr(m, "topics", None) or []
            out.append(
                MemoryItem(
                    user_id=user_id,
                    memory=text,
                    conversation=[{"role": "user", "content": text}],
                    tags=tags if isinstance(tags, list) else list(tags),
                    metadata=meta if isinstance(meta, dict) else {},
                    similarity_score=float(dist) if dist is not None else None,
                )
            )
        return out

    async def remove_items(self, **kwargs) -> None:
        """Remove memories by user_id or memory_id if the client supports it."""
        memory_id = kwargs.get("memory_id")
        user_id = kwargs.get("user_id")
        if memory_id is not None and hasattr(self._client, "delete_long_term_memory"):
            await self._client.delete_long_term_memory(memory_id)
            return
        if user_id is not None and hasattr(self._client, "forget"):
            await self._client.forget(user_id=user_id)
            return
        if user_id is not None:
            logger.warning(
                "Agent Memory Server client does not expose forget/delete by user_id; remove_items no-op"
            )
