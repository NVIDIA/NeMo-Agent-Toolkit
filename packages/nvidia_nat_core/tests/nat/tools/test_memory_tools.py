# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import AsyncExitStack

import pytest
from pydantic import ValidationError

from nat.tool.memory_tools.common import AddMemoryInput
from nat.tool.memory_tools.common import DeleteMemoryInput
from nat.tool.memory_tools.common import GetMemoryInput
from nat.tool.memory_tools.common import resolve_memory_user_id


class _MemoryEditor:

    def __init__(self):
        self.added_user_ids: list[str] = []
        self.searched_user_ids: list[str] = []
        self.deleted_user_ids: list[str] = []

    async def add_items(self, items):
        self.added_user_ids.extend(item.user_id for item in items)

    async def search(self, *, query, top_k, user_id):
        del query, top_k
        self.searched_user_ids.append(user_id)
        return []

    async def remove_items(self, *, user_id):
        self.deleted_user_ids.append(user_id)


class _Builder:

    def __init__(self, memory_editor):
        self.memory_editor = memory_editor

    async def get_memory_client(self, memory):
        del memory
        return self.memory_editor


@pytest.mark.parametrize(
    "input_type, input_value",
    [
        (AddMemoryInput, {"memory": "hello", "user_id": "attacker"}),
        (GetMemoryInput, {"query": "hello", "top_k": 1, "user_id": "attacker"}),
        (DeleteMemoryInput, {"user_id": "attacker"}),
    ],
)
def test_memory_tool_inputs_reject_llm_supplied_user_id(input_type, input_value):
    with pytest.raises(ValidationError):
        input_type.model_validate(input_value)


@pytest.mark.parametrize("input_type", [AddMemoryInput, GetMemoryInput, DeleteMemoryInput])
def test_memory_tool_schemas_do_not_publish_user_id(input_type):
    assert "user_id" not in input_type.model_json_schema().get("properties", {})


@pytest.mark.asyncio
async def test_resolve_memory_user_id_uses_fixed_identity():
    from nat.tool.memory_tools.add_memory_tool import AddToolConfig

    assert await resolve_memory_user_id(AddToolConfig(user_id=" fixed-user ")) == "fixed-user"


@pytest.mark.asyncio
async def test_resolve_memory_user_id_uses_synchronous_resolver():
    from nat.tool.memory_tools.add_memory_tool import AddToolConfig

    config = AddToolConfig(user_id_resolver=lambda: " resolved-user ")
    assert await resolve_memory_user_id(config) == "resolved-user"


@pytest.mark.asyncio
async def test_resolve_memory_user_id_uses_asynchronous_resolver():
    from nat.tool.memory_tools.add_memory_tool import AddToolConfig

    async def resolver():
        return "resolved-user"

    assert await resolve_memory_user_id(AddToolConfig(user_id_resolver=resolver)) == "resolved-user"


@pytest.mark.parametrize("resolved_value", [None, "", "   ", 123])
@pytest.mark.asyncio
async def test_resolve_memory_user_id_rejects_invalid_resolver_result(resolved_value):
    from nat.tool.memory_tools.add_memory_tool import AddToolConfig

    config = AddToolConfig(user_id_resolver=lambda: resolved_value)
    with pytest.raises(ValueError, match="non-empty string"):
        await resolve_memory_user_id(config)


@pytest.mark.asyncio
async def test_resolve_memory_user_id_requires_an_explicit_source():
    from nat.tool.memory_tools.add_memory_tool import AddToolConfig

    with pytest.raises(ValueError, match="user_id or user_id_resolver"):
        await resolve_memory_user_id(AddToolConfig())


def test_memory_tool_config_rejects_blank_fixed_user_id():
    from nat.tool.memory_tools.add_memory_tool import AddToolConfig

    with pytest.raises(ValidationError, match="user_id must not be empty"):
        AddToolConfig(user_id="  ")


def test_memory_tool_config_rejects_multiple_identity_sources():
    from nat.tool.memory_tools.add_memory_tool import AddToolConfig

    with pytest.raises(ValidationError, match="only one"):
        AddToolConfig(user_id="fixed-user", user_id_resolver=lambda: "resolved-user")


@pytest.mark.asyncio
async def test_memory_tools_use_resolved_identity_for_every_operation():
    from nat.tool.memory_tools.add_memory_tool import AddToolConfig
    from nat.tool.memory_tools.add_memory_tool import add_memory_tool
    from nat.tool.memory_tools.delete_memory_tool import DeleteToolConfig
    from nat.tool.memory_tools.delete_memory_tool import delete_memory_tool
    from nat.tool.memory_tools.get_memory_tool import GetToolConfig
    from nat.tool.memory_tools.get_memory_tool import get_memory_tool

    memory_editor = _MemoryEditor()
    builder = _Builder(memory_editor)
    resolver_calls = 0

    def resolver():
        nonlocal resolver_calls
        resolver_calls += 1
        return "authenticated-user"

    async with AsyncExitStack() as stack:
        add_tool = await stack.enter_async_context(add_memory_tool(AddToolConfig(user_id_resolver=resolver), builder))
        get_tool = await stack.enter_async_context(get_memory_tool(GetToolConfig(user_id_resolver=resolver), builder))
        delete_tool = await stack.enter_async_context(
            delete_memory_tool(DeleteToolConfig(user_id_resolver=resolver), builder))

        await add_tool.single_fn(AddMemoryInput(memory="strawberry"))
        await get_tool.single_fn(GetMemoryInput(query="favorite flavor", top_k=5))
        await delete_tool.single_fn(DeleteMemoryInput())

    assert resolver_calls == 3
    assert memory_editor.added_user_ids == ["authenticated-user"]
    assert memory_editor.searched_user_ids == ["authenticated-user"]
    assert memory_editor.deleted_user_ids == ["authenticated-user"]
