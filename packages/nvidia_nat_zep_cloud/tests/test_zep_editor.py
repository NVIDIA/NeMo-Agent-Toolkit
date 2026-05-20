# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from nat.plugins.zep_cloud.zep_editor import ZepEditor


def test_get_thread_id_uses_context_conversation_id():
    """Explicit conversation IDs should map directly to Zep threads."""
    editor = ZepEditor(zep_client=Mock())
    context = Mock()
    context.conversation_id = "conversation-123"

    with patch("nat.plugins.zep_cloud.zep_editor.Context.get", return_value=context):
        assert editor._get_thread_id("user-123") == "conversation-123"


def test_get_thread_id_defaults_to_default_zep_thread():
    """Missing conversation IDs should preserve the default Zep thread."""
    editor = ZepEditor(zep_client=Mock())
    context = Mock()
    context.conversation_id = None

    with patch("nat.plugins.zep_cloud.zep_editor.Context.get", return_value=context):
        thread_a = editor._get_thread_id("user-a")
        thread_b = editor._get_thread_id("user-b")

    assert thread_a == "default_zep_thread"
    assert thread_b == "default_zep_thread"


async def test_search_uses_graph_search_with_query():
    """Search should use the query against the user's graph before falling back to thread context."""
    zep_client = Mock()
    zep_client.graph.search = AsyncMock(
        return_value=Mock(edges=[Mock(fact="User name is Sam.")], episodes=[Mock(content="User enjoys cooking.")]))
    zep_client.thread.get_user_context = AsyncMock()
    editor = ZepEditor(zep_client=zep_client)
    context = Mock()
    context.conversation_id = "conversation-123"

    with patch("nat.plugins.zep_cloud.zep_editor.Context.get", return_value=context):
        results = await editor.search(query="What is my name and hobby?", user_id="user-123", top_k=3)

    zep_client.graph.search.assert_awaited_once_with(query="What is my name and hobby?", user_id="user-123", limit=3)
    zep_client.thread.get_user_context.assert_not_awaited()
    assert len(results) == 1
    assert "User name is Sam." in results[0].memory
    assert "User enjoys cooking." in results[0].memory


async def test_search_falls_back_to_thread_context_without_graph_results():
    """Empty graph search results should still use Zep's formatted thread context."""
    zep_client = Mock()
    zep_client.graph.search = AsyncMock(return_value=Mock(edges=[], episodes=[]))
    zep_client.thread.get_user_context = AsyncMock(return_value=Mock(context="Formatted Zep context"))
    editor = ZepEditor(zep_client=zep_client)
    context = Mock()
    context.conversation_id = "conversation-123"

    with patch("nat.plugins.zep_cloud.zep_editor.Context.get", return_value=context):
        results = await editor.search(query="What do you know?", user_id="user-123", mode="summary")

    zep_client.thread.get_user_context.assert_awaited_once_with(thread_id="conversation-123", mode="summary")
    assert len(results) == 1
    assert results[0].memory == "Formatted Zep context"
