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
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from nat.data_models.workspace import ActionContext
from nat.workspace_actions.workspace.todo.todo_read_action import TodoReadAction
from nat.workspace_actions.workspace.todo.todo_write_action import TodoWriteAction
from nat.workspace_actions.workspace.utils.todo_store import clear_todos


@pytest.fixture(name="fixture_action_context")
def fixture_action_context(tmp_path: Path) -> ActionContext:
    # Use a unique session id per test to avoid cross-test state pollution.
    session_id = f"test-session-todo-{uuid.uuid4().hex[:8]}"
    return ActionContext(session_id=session_id, root_path=tmp_path.resolve())


@pytest.fixture(name="fixture_todo_write")
def fixture_todo_write() -> TodoWriteAction:
    return TodoWriteAction()


@pytest.fixture(name="fixture_todo_read")
def fixture_todo_read() -> TodoReadAction:
    return TodoReadAction()


@pytest.fixture(autouse=True)
def _cleanup_todos(fixture_action_context: ActionContext):
    """Ensure todo store is clean before and after each test."""
    clear_todos(fixture_action_context.session_id)
    yield
    clear_todos(fixture_action_context.session_id)


async def test_todo_create_and_read(
    fixture_action_context: ActionContext,
    fixture_todo_write: TodoWriteAction,
    fixture_todo_read: TodoReadAction,
) -> None:
    """Write todos, then read them back and verify."""
    write_result = await fixture_todo_write.execute(
        fixture_action_context,
        {
            "todos": [
                {"id": "task-1", "content": "First task", "status": "pending"},
                {"id": "task-2", "content": "Second task", "status": "in_progress"},
            ],
        },
    )

    assert "is_error" not in write_result
    assert write_result["created"] == 2
    assert write_result["updated"] == 0

    read_result = await fixture_todo_read.execute(
        fixture_action_context,
        {},
    )

    assert "is_error" not in read_result
    assert read_result["count"] == 2
    ids = {item["id"] for item in read_result["todos"]}
    assert ids == {"task-1", "task-2"}


async def test_todo_merge_mode(
    fixture_action_context: ActionContext,
    fixture_todo_write: TodoWriteAction,
    fixture_todo_read: TodoReadAction,
) -> None:
    """Write an initial set, then merge new items preserving existing ones."""
    # Initial write
    await fixture_todo_write.execute(
        fixture_action_context,
        {
            "todos": [
                {"id": "t1", "content": "Original", "status": "pending"},
            ],
        },
    )

    # Merge a new item
    merge_result = await fixture_todo_write.execute(
        fixture_action_context,
        {
            "todos": [
                {"id": "t2", "content": "Merged", "status": "pending"},
            ],
            "merge": True,
        },
    )

    assert "is_error" not in merge_result
    assert merge_result["created"] == 1
    assert merge_result["updated"] == 0

    read_result = await fixture_todo_read.execute(fixture_action_context, {})
    assert read_result["count"] == 2
    ids = {item["id"] for item in read_result["todos"]}
    assert ids == {"t1", "t2"}


async def test_todo_replace_mode(
    fixture_action_context: ActionContext,
    fixture_todo_write: TodoWriteAction,
    fixture_todo_read: TodoReadAction,
) -> None:
    """Write an initial set, then replace entirely with a new set."""
    # Initial write
    await fixture_todo_write.execute(
        fixture_action_context,
        {
            "todos": [
                {"id": "old-1", "content": "Old task", "status": "pending"},
                {"id": "old-2", "content": "Another old", "status": "completed"},
            ],
        },
    )

    # Replace (merge=False is default)
    replace_result = await fixture_todo_write.execute(
        fixture_action_context,
        {
            "todos": [
                {"id": "new-1", "content": "New task", "status": "pending"},
            ],
            "merge": False,
        },
    )

    assert "is_error" not in replace_result
    assert replace_result["created"] == 1
    assert replace_result["removed"] == 2

    read_result = await fixture_todo_read.execute(fixture_action_context, {})
    assert read_result["count"] == 1
    assert read_result["todos"][0]["id"] == "new-1"


async def test_todo_filter_by_status(
    fixture_action_context: ActionContext,
    fixture_todo_write: TodoWriteAction,
    fixture_todo_read: TodoReadAction,
) -> None:
    """Write todos with mixed statuses and filter by pending."""
    await fixture_todo_write.execute(
        fixture_action_context,
        {
            "todos": [
                {"id": "a", "content": "Pending task", "status": "pending"},
                {"id": "b", "content": "Done task", "status": "completed"},
                {"id": "c", "content": "Another pending", "status": "pending"},
            ],
        },
    )

    read_result = await fixture_todo_read.execute(
        fixture_action_context,
        {"status": "pending"},
    )

    assert "is_error" not in read_result
    assert read_result["count"] == 2
    statuses = {item["status"] for item in read_result["todos"]}
    assert statuses == {"pending"}


async def test_todo_read_by_id(
    fixture_action_context: ActionContext,
    fixture_todo_write: TodoWriteAction,
    fixture_todo_read: TodoReadAction,
) -> None:
    """Read a specific todo by id."""
    await fixture_todo_write.execute(
        fixture_action_context,
        {
            "todos": [
                {"id": "target", "content": "Target task", "status": "pending"},
                {"id": "other", "content": "Other task", "status": "completed"},
            ],
        },
    )

    read_result = await fixture_todo_read.execute(
        fixture_action_context,
        {"id": "target"},
    )

    assert "is_error" not in read_result
    assert read_result["count"] == 1
    assert read_result["todos"][0]["id"] == "target"
    assert read_result["todos"][0]["content"] == "Target task"


async def test_todo_read_not_found(
    fixture_action_context: ActionContext,
    fixture_todo_read: TodoReadAction,
) -> None:
    """Read a nonexistent todo id and verify is_error."""
    read_result = await fixture_todo_read.execute(
        fixture_action_context,
        {"id": "nonexistent"},
    )

    assert read_result["is_error"] is True
    assert "not found" in read_result["error"].lower()
    assert read_result["count"] == 0


async def test_todo_dependency_validation(
    fixture_action_context: ActionContext,
    fixture_todo_write: TodoWriteAction,
) -> None:
    """A blocked_by reference to a nonexistent task should produce is_error."""
    write_result = await fixture_todo_write.execute(
        fixture_action_context,
        {
            "todos": [
                {
                    "id": "depends",
                    "content": "Depends on ghost",
                    "status": "pending",
                    "blocked_by": ["ghost-task"],
                },
            ],
        },
    )

    assert write_result["is_error"] is True
    assert "does not exist" in write_result["error"].lower() or "dependency" in write_result["error"].lower()


async def test_todo_cycle_detection(
    fixture_action_context: ActionContext,
    fixture_todo_write: TodoWriteAction,
) -> None:
    """Create a circular dependency (A blocked by B, B blocked by A) and verify is_error."""
    write_result = await fixture_todo_write.execute(
        fixture_action_context,
        {
            "todos": [
                {
                    "id": "cycle-a",
                    "content": "Task A",
                    "status": "pending",
                    "blocked_by": ["cycle-b"],
                },
                {
                    "id": "cycle-b",
                    "content": "Task B",
                    "status": "pending",
                    "blocked_by": ["cycle-a"],
                },
            ],
        },
    )

    assert write_result["is_error"] is True
    assert "circular" in write_result["error"].lower() or "cycle" in write_result["error"].lower()
