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
"""Multi-edit action implementation for applying sequential edits to a single file."""

from __future__ import annotations

import typing

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction
from nat.workspace_actions.workspace.file.edit_action import perform_edit
from nat.workspace_actions.workspace.utils.path_utils import resolve_workspace_path


@register_workspace_action
class MultiEditAction(WorkspaceAction):
    """Apply multiple sequential edit operations to a single file."""

    name = "multi_edit"
    description = (
        "Apply multiple sequential edit operations to a single file.\n"
        "\n"
        "When to use:\n"
        "- When you need to make several related changes to one file\n"
        "- When changes should be applied together atomically\n"
        "- More efficient than multiple separate edit calls\n"
        "\n"
        "How it works:\n"
        "- Edits are applied sequentially in order\n"
        "- If any edit fails, remaining edits are skipped\n"
        "- Each edit uses the same validation as the edit action\n"
        "\n"
        "Important:\n"
        "- Order matters: edits are applied sequentially\n"
        "- Each old_string must be unique in the file (unless using replace_all)"
    )
    parameters = [
        TypeSchema(type="string", description="file_path: Absolute or workspace-relative path to modify."),
        TypeSchema(
            type="array",
            description=(
                "edits: Ordered list of edit operations. "
                "Each element is an object with old_string (string), new_string (string), "
                "and optional replace_all (boolean, default false)."
            ),
        ),
    ]
    result = TypeSchema(
        type="object",
        description=(
            "Result payload with path, total_edits, successful_edits, failed_edits, "
            "total_replacements, modified, and results."
        ),
    )

    def __init__(self) -> None:
        pass

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        file_path_str = args.get("file_path")
        if not isinstance(file_path_str, str) or not file_path_str.strip():
            return {"is_error": True, "error": "Action requires a non-empty 'file_path' string."}

        edits = args.get("edits")
        if not isinstance(edits, list) or not edits:
            return {"is_error": True, "error": "Action requires a non-empty 'edits' array."}

        path = resolve_workspace_path(file_path_str, context.root_path)

        if not path.exists():
            return {
                "is_error": True,
                "error": f"File does not exist: {path}",
                "path": str(path),
                "total_edits": len(edits),
                "successful_edits": 0,
                "failed_edits": len(edits),
                "total_replacements": 0,
                "modified": False,
                "results": [],
            }

        results: list[dict[str, typing.Any]] = []
        total_replacements = 0
        modified = False

        for index, edit in enumerate(edits):
            if not isinstance(edit, dict):
                results.append({
                    "index": index,
                    "success": False,
                    "replacements": 0,
                    "error": "Each edit must be an object with old_string and new_string.",
                })
                break

            old_string = edit.get("old_string", "")
            new_string = edit.get("new_string", "")
            replace_all = bool(edit.get("replace_all", False))

            if not isinstance(old_string, str) or not isinstance(new_string, str):
                results.append({
                    "index": index,
                    "success": False,
                    "replacements": 0,
                    "error": "old_string and new_string must be strings.",
                })
                break

            try:
                result = perform_edit(
                    file_path=str(path),
                    old_string=old_string,
                    new_string=new_string,
                    replace_all=replace_all,
                    root_path=context.root_path,
                    session_id=context.session_id,
                )
                replacements = int(result["replacements"])
                results.append({
                    "index": index,
                    "success": True,
                    "replacements": replacements,
                    "error": None,
                })
                total_replacements += replacements
                modified = modified or bool(result["modified"])
            except Exception as exc:
                results.append({
                    "index": index,
                    "success": False,
                    "replacements": 0,
                    "error": str(exc),
                })
                break

        successful_edits = sum(1 for item in results if item["success"])
        failed_edits = len(edits) - successful_edits

        return {
            "path": str(path),
            "total_edits": len(edits),
            "successful_edits": successful_edits,
            "failed_edits": failed_edits,
            "total_replacements": total_replacements,
            "modified": modified,
            "results": results,
            **({"is_error": True} if failed_edits > 0 else {}),
        }
