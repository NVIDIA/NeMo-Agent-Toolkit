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

from typing import Any

from nat.memory.models import MemoryItem


def memory_item_to_add_kwargs(item: MemoryItem) -> dict[str, Any]:
    """Convert a NeMo MemoryItem to SelfMemory add() keyword arguments."""
    messages = item.conversation if item.conversation else item.memory or ""
    tags = ",".join(item.tags) if item.tags else None

    metadata = dict(item.metadata) if item.metadata else {}
    people_mentioned = metadata.pop("people_mentioned", None)
    topic_category = metadata.pop("topic_category", None)
    project_id = metadata.pop("project_id", None)
    organization_id = metadata.pop("organization_id", None)

    kwargs = {
        "messages": messages,
        "user_id": item.user_id or "default",
        "tags": tags,
        "people_mentioned": people_mentioned,
        "topic_category": topic_category,
        "project_id": project_id,
        "organization_id": organization_id,
        "metadata": metadata if metadata else None,
    }

    return {k: v for k, v in kwargs.items() if v is not None}


def search_results_to_memory_items(
    results: dict[str, Any], user_id: str
) -> list[MemoryItem]:
    """Convert SelfMemory search results to a list of NeMo MemoryItems."""
    items = []

    for result in results.get("results", []):
        metadata = result.get("metadata", {})
        tags_str = metadata.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

        items.append(
            MemoryItem(
                conversation=[],
                user_id=user_id,
                memory=result.get("content", ""),
                tags=tags,
                metadata=metadata,
            )
        )

    return items
