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

from contextlib import AsyncExitStack

import pytest
from pydantic import ValidationError

from nat.builder.context import Context
from nat.tool.memory_tools.common import AddMemoryInput
from nat.tool.memory_tools.common import DeleteMemoryInput
from nat.tool.memory_tools.common import GetMemoryInput
from nat.tool.memory_tools.common import resolve_memory_user_id


@pytest.mark.parametrize("input_type, input_data", [
    (AddMemoryInput, {"memory": "User prefers strawberry", "user_id": "another-user"}),
    (GetMemoryInput, {"query": "preferences", "top_k": 1, "user_id": "another-user"}),
    (DeleteMemoryInput, {"user_id": "another-user"}),
])
def test_memory_tool_inputs_reject_llm_supplied_user_id(input_type: type, input_data: dict):
    with pytest.raises(ValidationError, match="user_id"):
        input_type.model_validate(input_data)


def test_memory_tool_inputs_do_not_publish_user_id_in_schema():
    for input_type in (AddMemoryInput, GetMemoryInput, DeleteMemoryInput):
        assert "user_id" not in input_type.model_json_schema().get("properties", {})


def test_resolve_memory_user_id_prefers_configured_identity():
    with Context.scope(user_id="request-user"):
        assert resolve_memory_user_id("configured-user") == "configured-user"


def test_memory_tool_config_rejects_empty_user_id():
    from nat.tool.memory_tools.add_memory_tool import AddToolConfig

    with pytest.raises(ValidationError, match="user_id"):
        AddToolConfig(user_id="   ")


def test_resolve_memory_user_id_uses_context_identity():
    with Context.scope(user_id="request-user"):
        assert resolve_memory_user_id(None) == "request-user"


def test_resolve_memory_user_id_requires_config_or_context_identity():
    with Context.scope(user_id=None):
        with pytest.raises(ValueError, match="No user identity"):
            resolve_memory_user_id(None)


class _MemoryEditor:

    def __init__(self):
        self.items = []
        self.search_user_id = None
        self.delete_user_id = None

    async def add_items(self, items):
        self.items.extend(items)

    async def search(self, *, query, top_k, user_id):
        self.search_user_id = user_id
        return []

    async def remove_items(self, *, user_id):
        self.delete_user_id = user_id


class _Builder:

    def __init__(self, memory_editor):
        self.memory_editor = memory_editor

    async def get_memory_client(self, _memory):
        return self.memory_editor


async def test_memory_tools_use_context_identity_for_every_operation():
    from nat.tool.memory_tools.add_memory_tool import AddToolConfig
    from nat.tool.memory_tools.add_memory_tool import add_memory_tool
    from nat.tool.memory_tools.delete_memory_tool import DeleteToolConfig
    from nat.tool.memory_tools.delete_memory_tool import delete_memory_tool
    from nat.tool.memory_tools.get_memory_tool import GetToolConfig
    from nat.tool.memory_tools.get_memory_tool import get_memory_tool

    memory_editor = _MemoryEditor()
    builder = _Builder(memory_editor)

    async with AsyncExitStack() as stack:
        add_tool = await stack.enter_async_context(add_memory_tool(AddToolConfig(), builder))
        get_tool = await stack.enter_async_context(get_memory_tool(GetToolConfig(), builder))
        delete_tool = await stack.enter_async_context(delete_memory_tool(DeleteToolConfig(), builder))

        with Context.scope(user_id="request-user"):
            await add_tool.single_fn(AddMemoryInput(memory="User prefers strawberry"))
            await get_tool.single_fn(GetMemoryInput(query="preferences", top_k=1))
            await delete_tool.single_fn(DeleteMemoryInput())

    assert memory_editor.items[0].user_id == "request-user"
    assert memory_editor.search_user_id == "request-user"
    assert memory_editor.delete_user_id == "request-user"
