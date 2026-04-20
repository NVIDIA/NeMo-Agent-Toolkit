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

from nat.memory.models import MemoryItem
from nat.plugins.selfmemory.translator import (
    memory_item_to_add_kwargs,
    search_results_to_memory_items,
)


class TestMemoryItemToAddKwargs:

    def test_basic_conversion(self):
        """Test basic MemoryItem to add kwargs conversion."""
        item = MemoryItem(
            conversation=[{"role": "user", "content": "I love pizza"}],
            user_id="alice",
            memory="Loves pizza",
            tags=["food"],
            metadata={},
        )

        kwargs = memory_item_to_add_kwargs(item)

        assert kwargs["messages"] == [{"role": "user", "content": "I love pizza"}]
        assert kwargs["user_id"] == "alice"
        assert kwargs["tags"] == "food"

    def test_multiple_tags(self):
        """Test that multiple tags are joined with commas."""
        item = MemoryItem(
            conversation=[],
            user_id="bob",
            memory="Note",
            tags=["work", "meeting", "important"],
            metadata={},
        )

        kwargs = memory_item_to_add_kwargs(item)

        assert kwargs["tags"] == "work,meeting,important"

    def test_empty_tags(self):
        """Test that empty tags are omitted."""
        item = MemoryItem(
            conversation=[],
            user_id="charlie",
            memory="Note",
            tags=[],
            metadata={},
        )

        kwargs = memory_item_to_add_kwargs(item)

        assert "tags" not in kwargs

    def test_extracts_metadata_fields(self):
        """Test that known metadata fields are extracted to top-level kwargs."""
        item = MemoryItem(
            conversation=[],
            user_id="alice",
            memory="Meeting note",
            tags=[],
            metadata={
                "people_mentioned": "Sarah,Mike",
                "topic_category": "work",
                "project_id": "proj_1",
                "organization_id": "org_1",
                "custom_field": "value",
            },
        )

        kwargs = memory_item_to_add_kwargs(item)

        assert kwargs["people_mentioned"] == "Sarah,Mike"
        assert kwargs["topic_category"] == "work"
        assert kwargs["project_id"] == "proj_1"
        assert kwargs["organization_id"] == "org_1"
        assert kwargs["metadata"] == {"custom_field": "value"}

    def test_fallback_to_memory_string(self):
        """Test that memory string is used when conversation is empty."""
        item = MemoryItem(
            conversation=[],
            user_id="alice",
            memory="Direct memory text",
            tags=[],
            metadata={},
        )

        kwargs = memory_item_to_add_kwargs(item)

        assert kwargs["messages"] == "Direct memory text"

    def test_default_user_id(self):
        """Test that empty user_id defaults to 'default'."""
        item = MemoryItem(
            conversation=[],
            user_id="",
            memory="Note",
            tags=[],
            metadata={},
        )

        kwargs = memory_item_to_add_kwargs(item)

        assert kwargs["user_id"] == "default"


class TestSearchResultsToMemoryItems:

    def test_basic_conversion(self):
        """Test basic search result to MemoryItem conversion."""
        results = {
            "results": [
                {
                    "id": "mem_1",
                    "content": "Loves pizza",
                    "score": 0.95,
                    "metadata": {
                        "data": "Loves pizza",
                        "user_id": "alice",
                        "tags": "food,preferences",
                    },
                }
            ]
        }

        items = search_results_to_memory_items(results, "alice")

        assert len(items) == 1
        assert items[0].memory == "Loves pizza"
        assert items[0].user_id == "alice"
        assert items[0].tags == ["food", "preferences"]

    def test_empty_results(self):
        """Test empty search results."""
        results = {"results": []}

        items = search_results_to_memory_items(results, "alice")

        assert len(items) == 0

    def test_empty_tags(self):
        """Test result with empty tags string."""
        results = {
            "results": [
                {
                    "id": "mem_1",
                    "content": "Note",
                    "metadata": {"tags": ""},
                }
            ]
        }

        items = search_results_to_memory_items(results, "alice")

        assert items[0].tags == []

    def test_missing_metadata(self):
        """Test result with missing metadata."""
        results = {
            "results": [
                {
                    "id": "mem_1",
                    "content": "Note",
                }
            ]
        }

        items = search_results_to_memory_items(results, "alice")

        assert items[0].memory == "Note"
        assert items[0].tags == []

    def test_multiple_results(self):
        """Test multiple search results."""
        results = {
            "results": [
                {"id": "mem_1", "content": "First", "metadata": {"tags": "a"}},
                {"id": "mem_2", "content": "Second", "metadata": {"tags": "b"}},
                {"id": "mem_3", "content": "Third", "metadata": {"tags": "c"}},
            ]
        }

        items = search_results_to_memory_items(results, "alice")

        assert len(items) == 3
        assert items[0].memory == "First"
        assert items[2].memory == "Third"
