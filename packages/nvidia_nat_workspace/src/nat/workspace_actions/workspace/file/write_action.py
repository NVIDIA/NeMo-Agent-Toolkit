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
"""Write action implementation for creating or overwriting workspace files."""

from __future__ import annotations

import errno
import typing

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction
from nat.workspace_actions.workspace.utils.file_state import clear_file_read_state
from nat.workspace_actions.workspace.utils.path_utils import resolve_workspace_path


@register_workspace_action
class WriteAction(WorkspaceAction):
    """Create or overwrite a file in the workspace."""

    name = "write"
    description = (
        "Create or overwrite a file in the workspace.\n"
        "Creates the file and any necessary parent directories if they do not exist.\n"
        "Overwrites the file if it already exists.\n"
        "\n"
        "Usage:\n"
        "- Provide file_path (absolute or workspace-relative) and content string.\n"
        "- For new files in nested directories, parent directories are created automatically.\n"
        "- Prefer the edit action for modifying existing files (not write)."
    )
    parameters = [
        TypeSchema(type="string", description="file_path: Absolute or workspace-relative file path to write."),
        TypeSchema(type="string", description="content: File contents to write."),
    ]
    result = TypeSchema(
        type="object",
        description="Result payload with path, bytes_written, and created flags.",
    )

    def __init__(self) -> None:
        pass

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        file_path_str = args.get("file_path")
        if not isinstance(file_path_str, str) or not file_path_str.strip():
            return {"is_error": True, "error": "Action requires a non-empty 'file_path' string."}

        content = args.get("content")
        if not isinstance(content, str):
            return {"is_error": True, "error": "Action requires a 'content' string."}

        file_path = resolve_workspace_path(file_path_str, context.root_path)
        parent = file_path.parent
        created = not file_path.exists()

        try:
            parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            bytes_written = len(content.encode("utf-8"))
            clear_file_read_state(context.session_id, str(file_path))
            return {
                "path": str(file_path),
                "bytes_written": bytes_written,
                "created": created,
            }
        except OSError as exc:
            if exc.errno == errno.EACCES:
                message = f"Permission denied: {file_path}"
            elif exc.errno == errno.EISDIR:
                message = f"Is a directory: {file_path}"
            else:
                message = f"Write failed: {exc}"
            return {
                "is_error": True,
                "error": message,
                "path": str(file_path),
                "bytes_written": 0,
                "created": created,
            }
