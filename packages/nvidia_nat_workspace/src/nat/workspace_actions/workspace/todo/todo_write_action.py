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

"""TodoWrite workspace action for creating and updating session todos."""

from __future__ import annotations

import typing

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction
from nat.workspace_actions.workspace.utils.todo_store import (
    TodoItem,
    calculate_summary,
    get_todos,
    set_todos,
    validate_dependencies,
    validate_todo_graph,
)


@register_workspace_action
class TodoWriteAction(WorkspaceAction):
    """Create or update todo items in the session task store."""

    name = "todo_write"
    description = (
        "Create or update todo items in the session task store.\n"
        "Helps track progress, organize complex tasks, and demonstrate thoroughness.\n"
        "\n"
        "Use proactively for:\n"
        "1. Complex multi-step tasks (3+ distinct steps)\n"
        "2. Non-trivial tasks requiring careful planning\n"
        "3. After receiving new instructions: capture requirements as todos (merge=false)\n"
        "4. After completing tasks: mark complete with merge=true and add follow-ups\n"
        "\n"
        "Task states: pending, in_progress, completed, cancelled, error, skipped, paused\n"
        "\n"
        "Task management:\n"
        "- Update status in real-time\n"
        "- Mark complete IMMEDIATELY after finishing\n"
        "- Only ONE task in_progress at a time\n"
        "\n"
        "Task dependencies:\n"
        "- Use blocks/blocked_by to create task dependency chains\n"
        "- Automatic circular dependency detection\n"
        "- Validation ensures all referenced tasks exist"
    )
    parameters = [
        TypeSchema(
            type="array",
            description=(
                "todos: List of todo item objects. Each item requires: "
                "id (string), content (string), status (string). "
                "Optional fields: active_form (string), metadata (object), "
                "emoji (string), blocks (array of ids), blocked_by (array of ids)."
            ),
        ),
        TypeSchema(
            type="boolean",
            description="merge: If true, merge into existing todos; if false, replace entire list.",
        ),
    ]
    result = TypeSchema(
        type="object",
        description="Result payload with created, updated, removed counts, todos list, and summary.",
    )

    def __init__(self) -> None:
        pass

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        raw_todos = args.get("todos")
        if not raw_todos or not isinstance(raw_todos, list):
            return self._error(context.session_id, "Action requires a non-empty 'todos' array.")

        merge = args.get("merge", False)

        # Parse and validate each incoming todo item.
        incoming: list[TodoItem] = []
        for raw in raw_todos:
            if not isinstance(raw, dict):
                return self._error(context.session_id, "Each todo must be a dictionary.")
            for required_field in ("id", "content", "status"):
                if required_field not in raw:
                    return self._error(
                        context.session_id,
                        f"Todo item missing required field '{required_field}'.",
                    )
            try:
                item = TodoItem(
                    id=raw["id"],
                    content=raw["content"],
                    status=raw["status"],
                    active_form=raw.get("active_form"),
                    metadata=raw.get("metadata"),
                    emoji=raw.get("emoji"),
                    blocks=raw.get("blocks"),
                    blocked_by=raw.get("blocked_by"),
                )
            except Exception as exc:
                return self._error(context.session_id, f"Invalid todo item: {exc}")
            incoming.append(item)

        current_store = {todo.id: todo for todo in get_todos(context.session_id)}

        # Build candidate store.
        if merge:
            candidate = dict(current_store)
            for item in incoming:
                candidate[item.id] = item
        else:
            candidate = {item.id: item for item in incoming}

        # Validate per-item dependencies with batch cross-reference support.
        pending_ids = {item.id for item in incoming}
        for item in incoming:
            dependency_error = validate_dependencies(
                session_id=context.session_id,
                task_id=item.id,
                blocked_by=item.blocked_by,
                pending_task_ids=pending_ids,
                store=candidate,
            )
            if dependency_error:
                return self._error(
                    context.session_id,
                    f"Dependency validation failed for task {item.id}: {dependency_error}",
                )

        # Validate full graph for cycles.
        graph_error = validate_todo_graph(candidate)
        if graph_error:
            return self._error(context.session_id, f"Dependency validation failed: {graph_error}")

        # Count changes and persist.
        if merge:
            created = 0
            updated = 0
            removed = 0
            for item in incoming:
                if item.id in current_store:
                    updated += 1
                else:
                    created += 1
                current_store[item.id] = item
            set_todos(context.session_id, current_store.values())
        else:
            previous_ids = set(current_store.keys())
            new_ids = {item.id for item in incoming}
            created = len(new_ids - previous_ids)
            updated = len(new_ids & previous_ids)
            removed = len(previous_ids - new_ids)
            set_todos(context.session_id, incoming)

        todos = get_todos(context.session_id)
        summary = calculate_summary(todos)

        return {
            "created": created,
            "updated": updated,
            "removed": removed,
            "todos": [item.model_dump() for item in todos],
            "summary": summary,
        }

    @staticmethod
    def _error(session_id: str, message: str) -> dict[str, typing.Any]:
        current_todos = get_todos(session_id)
        summary = calculate_summary(current_todos)
        return {
            "is_error": True,
            "error": message,
            "created": 0,
            "updated": 0,
            "removed": 0,
            "todos": [item.model_dump() for item in current_todos],
            "summary": summary,
        }
