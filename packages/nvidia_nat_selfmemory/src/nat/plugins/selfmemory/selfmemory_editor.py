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

import asyncio

from selfmemory import SelfMemory

from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem

from .translator import memory_item_to_add_kwargs, search_results_to_memory_items


class SelfMemoryEditor(MemoryEditor):
    """
    Wrapper class that implements NAT MemoryEditor for SelfMemory.

    Bridges SelfMemory's synchronous API to NeMo Agent Toolkit's
    asynchronous MemoryEditor interface using asyncio.to_thread().
    """

    def __init__(self, backend: SelfMemory):
        self._backend = backend

    async def add_items(self, items: list[MemoryItem]) -> None:
        """Insert multiple MemoryItems into SelfMemory."""
        coroutines = [
            asyncio.to_thread(self._backend.add, **memory_item_to_add_kwargs(item))
            for item in items
        ]
        await asyncio.gather(*coroutines)

    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
        """Retrieve items relevant to the given query."""
        user_id = kwargs.pop("user_id")

        result = await asyncio.to_thread(
            self._backend.search, query, user_id=user_id, limit=top_k
        )

        return search_results_to_memory_items(result, user_id)

    async def remove_items(self, **kwargs) -> None:
        """Remove items by memory_id or user_id."""
        if "memory_id" in kwargs:
            memory_id = kwargs.pop("memory_id")
            await asyncio.to_thread(self._backend.delete, memory_id)

        elif "user_id" in kwargs:
            user_id = kwargs.pop("user_id")
            await asyncio.to_thread(self._backend.delete_all, user_id=user_id)
