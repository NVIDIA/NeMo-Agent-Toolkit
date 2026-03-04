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

"""TodoRead workspace action for inspecting the session todo store."""

from __future__ import annotations

import typing

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction
from nat.workspace_actions.workspace.utils.todo_store import (
    calculate_summary,
    get_todo_by_id,
    get_todos,
    is_task_blocked,
)


@register_workspace_action
class TodoReadAction(WorkspaceAction):
    """Read todo items from the session task store."""

    name = "todo_read"
    description = (
        "Read the current todo list from the session task store.\n"
        "\n"
        "Basic usage: call with no parameters to get all todos.\n"
        "\n"
        "Filtering:\n"
        "- By ID: provide id to get a specific todo\n"
        "- By status: provide status to filter (pending/in_progress/completed/cancelled/error/skipped/paused)\n"
        "\n"
        "Output includes: todos, count, summary, available (unblocked pending), and blocked lists."
    )
    parameters = [
        TypeSchema(type="string", description="id: Optional specific task id to read."),
        TypeSchema(type="string", description="status: Optional status filter (pending/in_progress/completed/cancelled/error/skipped/paused)."),
        TypeSchema(type="boolean", description="include_blocked: Include blocked and available pending task views. Default true."),
    ]
    result = TypeSchema(
        type="object",
        description="Result payload with todos, count, summary, available, and blocked lists.",
    )

    def __init__(self) -> None:
        pass

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        task_id = args.get("id")
        status = args.get("status")
        include_blocked = args.get("include_blocked", True)

        all_todos = get_todos(context.session_id)

        if task_id:
            found = get_todo_by_id(context.session_id, task_id)
            if found is None:
                summary = calculate_summary(all_todos)
                return {
                    "is_error": True,
                    "error": f"Todo with ID '{task_id}' not found",
                    "todos": [],
                    "count": 0,
                    "summary": summary,
                    "available": [],
                    "blocked": [],
                }
            todos = [found]
        else:
            todos = list(all_todos)

        if status is not None:
            todos = [todo for todo in todos if todo.status == status]

        store = {todo.id: todo for todo in all_todos}
        pending = [todo for todo in all_todos if todo.status == "pending"]

        if include_blocked:
            available = [todo for todo in pending if not is_task_blocked(todo.id, store)]
            blocked = [todo for todo in pending if is_task_blocked(todo.id, store)]
        else:
            available = []
            blocked = []

        summary = calculate_summary(all_todos)

        return {
            "todos": [item.model_dump() for item in todos],
            "count": len(todos),
            "summary": summary,
            "available": [item.model_dump() for item in available],
            "blocked": [item.model_dump() for item in blocked],
        }
