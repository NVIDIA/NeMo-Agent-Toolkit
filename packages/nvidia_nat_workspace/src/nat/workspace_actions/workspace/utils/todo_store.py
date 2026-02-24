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

"""Session-scoped in-memory todo store for TodoWrite and TodoRead tools."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TodoStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "cancelled",
    "error",
    "skipped",
    "paused",
]


class TodoItem(BaseModel):
    """Canonical todo item representation in store."""

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    status: TodoStatus
    active_form: str | None = None
    metadata: dict[str, Any] | None = None
    emoji: str | None = None
    blocks: list[str] | None = None
    blocked_by: list[str] | None = None


_session_stores: dict[str, dict[str, TodoItem]] = {}


def clear_todos(session_id: str) -> None:
    """Clear all todos from the session's in-memory store."""
    store = _session_stores.get(session_id)
    if store is not None:
        store.clear()


def set_todos(session_id: str, todos: Iterable[TodoItem]) -> None:
    """Replace session's todo store with provided items."""
    store = _session_stores.setdefault(session_id, {})
    store.clear()
    for todo in todos:
        store[todo.id] = todo


def get_todos(session_id: str) -> list[TodoItem]:
    """Return all todos in insertion order."""
    store = _session_stores.get(session_id, {})
    return list(store.values())


def get_todo_by_id(session_id: str, task_id: str) -> TodoItem | None:
    """Return one todo by id."""
    store = _session_stores.get(session_id, {})
    return store.get(task_id)


def is_task_blocked(task_id: str, store: dict[str, TodoItem] | None = None) -> bool:
    """Return whether a task is blocked by unfinished dependencies."""
    active_store = store or {}
    task = active_store.get(task_id)
    if task is None or not task.blocked_by:
        return False
    for blocker_id in task.blocked_by:
        blocker = active_store.get(blocker_id)
        if blocker is not None and blocker.status != "completed":
            return True
    return False


def get_blocked_tasks(task_id: str, store: dict[str, TodoItem] | None = None) -> list[TodoItem]:
    """Return tasks that are blocked by the provided task."""
    active_store = store or {}
    task = active_store.get(task_id)
    if task is None or not task.blocks:
        return []
    return [active_store[item_id] for item_id in task.blocks if item_id in active_store]


def get_blocking_tasks(task_id: str, store: dict[str, TodoItem] | None = None) -> list[TodoItem]:
    """Return tasks currently blocking the provided task."""
    active_store = store or {}
    task = active_store.get(task_id)
    if task is None or not task.blocked_by:
        return []
    return [active_store[item_id] for item_id in task.blocked_by if item_id in active_store]


def get_available_tasks(store: dict[str, TodoItem] | None = None) -> list[TodoItem]:
    """Return pending tasks that are not blocked."""
    active_store = store or {}
    return [
        task
        for task in active_store.values()
        if task.status == "pending" and not is_task_blocked(task.id, active_store)
    ]


def calculate_summary(todos: list[TodoItem]) -> dict[str, int]:
    """Build per-status counts and total."""
    summary = {
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
        "cancelled": 0,
        "error": 0,
        "skipped": 0,
        "paused": 0,
        "total": len(todos),
    }
    for todo in todos:
        summary[todo.status] += 1
    return summary


def validate_dependencies(
    session_id: str,
    task_id: str,
    blocked_by: list[str] | None,
    pending_task_ids: set[str] | None = None,
    store: dict[str, TodoItem] | None = None,
) -> str | None:
    """Validate one task dependency list against an existing todo store."""
    active_store = dict(store or _session_stores.get(session_id, {}))
    dependencies = blocked_by or []
    if not dependencies:
        return None

    for dependency_id in dependencies:
        exists_in_store = dependency_id in active_store
        exists_in_batch = dependency_id in (pending_task_ids or set())
        if not exists_in_store and not exists_in_batch:
            return f"Task {dependency_id} does not exist"
        if dependency_id == task_id:
            return "Task cannot be blocked by itself"

    original = active_store.get(task_id)
    if original is not None:
        active_store[task_id] = original.model_copy(update={"blocked_by": dependencies})

    if _has_cycle(active_store):
        return "Circular dependency detected"
    return None


def validate_todo_graph(store: dict[str, TodoItem]) -> str | None:
    """Validate all dependency links and detect cycles for a todo graph."""
    for task in store.values():
        dependencies = task.blocked_by or []
        for dependency_id in dependencies:
            if dependency_id == task.id:
                return "Task cannot be blocked by itself"
            if dependency_id not in store:
                return f"Task {dependency_id} does not exist"
    if _has_cycle(store):
        return "Circular dependency detected"
    return None


def _has_cycle(store: dict[str, TodoItem]) -> bool:
    visited: dict[str, int] = {}

    def dfs(task_id: str) -> bool:
        state = visited.get(task_id, 0)
        if state == 1:
            return True
        if state == 2:
            return False
        visited[task_id] = 1
        task = store.get(task_id)
        for dependency_id in (task.blocked_by or []) if task else []:
            if dependency_id in store and dfs(dependency_id):
                return True
        visited[task_id] = 2
        return False

    for task_id in store:
        if visited.get(task_id, 0) == 0 and dfs(task_id):
            return True
    return False
