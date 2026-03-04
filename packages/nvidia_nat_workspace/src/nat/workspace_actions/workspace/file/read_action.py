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
"""Read action implementation for workspace files."""

from __future__ import annotations

import errno
import json
import typing

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction
from nat.workspace_actions.workspace.utils.file_state import register_file_read
from nat.workspace_actions.workspace.utils.levenshtein import find_similar_file
from nat.workspace_actions.workspace.utils.path_utils import resolve_workspace_path

try:
    import nbformat
except Exception:  # pragma: no cover - optional fallback when dependency missing
    nbformat = None


MAX_TEXT_FILE_SIZE = 10 * 1024 * 1024
MAX_LINE_LENGTH = 2000
DEFAULT_MAX_LINES = 2000
MAX_CONTENT_CHARS = 100000

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_BINARY_SUFFIXES = (
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".dat",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".rar",
    ".7z",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".pyc",
    ".pyo",
    ".class",
    ".o",
    ".obj",
    ".a",
    ".lib",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".wav",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
)


def _is_image_file(file_path: typing.Any) -> bool:
    from pathlib import Path

    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def _is_binary_file(file_path: typing.Any) -> bool:
    from pathlib import Path

    lowered = Path(file_path).name.lower()
    return lowered.endswith(_BINARY_SUFFIXES)


def _format_file_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _parse_notebook(raw_content: str) -> dict[str, typing.Any]:
    """Parse a notebook from raw JSON, trying nbformat first."""
    if nbformat is not None:
        try:
            notebook = nbformat.reads(raw_content, as_version=4)
            if hasattr(notebook, "dict"):
                return notebook.dict()
            if isinstance(notebook, dict):
                return notebook
        except Exception:
            # Fall back to plain JSON parsing for loosely-structured notebooks.
            pass
    parsed = json.loads(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError("Notebook root must be an object")
    return parsed


@register_workspace_action
class ReadAction(WorkspaceAction):
    """Read a file from the workspace with numbered lines and pagination."""

    name = "read"
    description = (
        "Read a file from the workspace. Returns file contents with line numbers.\n"
        "Lines are numbered starting at 1, using format: LINE_NUMBER|LINE_CONTENT\n"
        "\n"
        "Supports multiple file types:\n"
        "- Text files: Returns content with line numbers\n"
        "- Jupyter Notebooks (.ipynb): Returns formatted cells with outputs\n"
        "- Images (PNG, JPEG, GIF, WebP): Returns image file metadata\n"
        "\n"
        "Features:\n"
        "- Tracks file reads for edit validation\n"
        "- Enforces 10MB size limit for text files\n"
        "- Truncates long lines (>2000 chars)\n"
        "- Suggests similar filenames if file not found\n"
        "- Use offset/limit for pagination on large files"
    )
    parameters = [
        TypeSchema(type="string", description="file_path: Absolute or workspace-relative path to read."),
        TypeSchema(type="number", description="offset: Line number to start from (1-indexed). Defaults to 1."),
        TypeSchema(type="number", description="limit: Maximum lines to return. Defaults to 2000."),
    ]
    result = TypeSchema(
        type="object",
        description="Result payload with content, total_lines, lines_returned, start_line, and end_line.",
    )

    def __init__(self) -> None:
        pass

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        file_path_str = args.get("file_path")
        if not isinstance(file_path_str, str) or not file_path_str.strip():
            return {"is_error": True, "error": "Action requires a non-empty 'file_path' string."}

        offset = args.get("offset")
        limit = args.get("limit")

        file_path = resolve_workspace_path(file_path_str, context.root_path)

        # --- stat the file ---
        try:
            stats = file_path.stat()
        except OSError as exc:
            return self._file_stat_error(file_path, exc)

        if not file_path.is_file():
            return {
                "is_error": True,
                "error": f"Path is not a file: {file_path}",
                "content": "",
                "total_lines": 0,
                "lines_returned": 0,
                "start_line": 0,
                "end_line": 0,
            }

        # --- image detection ---
        if _is_image_file(file_path):
            content = f"[Image file: {_format_file_size(stats.st_size)}]"
            register_file_read(context.session_id, str(file_path), content)
            return {
                "content": content,
                "total_lines": 1,
                "lines_returned": 1,
                "start_line": 1,
                "end_line": 1,
            }

        # --- notebook reading ---
        if file_path.suffix.lower() == ".ipynb":
            return self._read_notebook(file_path, context.session_id)

        # --- binary detection ---
        if _is_binary_file(file_path):
            return {
                "is_error": True,
                "error": "This tool cannot read binary files.",
                "content": "",
                "total_lines": 0,
                "lines_returned": 0,
                "start_line": 0,
                "end_line": 0,
            }

        # --- file size check ---
        if stats.st_size > MAX_TEXT_FILE_SIZE:
            return {
                "is_error": True,
                "error": (
                    f"File size ({_format_file_size(stats.st_size)}) exceeds the "
                    f"maximum allowed size ({_format_file_size(MAX_TEXT_FILE_SIZE)}). "
                    "Use offset/limit to read portions of the file."
                ),
                "content": "",
                "total_lines": 0,
                "lines_returned": 0,
                "start_line": 0,
                "end_line": 0,
            }

        # --- read text content ---
        try:
            raw_content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {
                "is_error": True,
                "error": "This tool cannot read binary files.",
                "content": "",
                "total_lines": 0,
                "lines_returned": 0,
                "start_line": 0,
                "end_line": 0,
            }
        except OSError as exc:
            return {
                "is_error": True,
                "error": str(exc),
                "content": "",
                "total_lines": 0,
                "lines_returned": 0,
                "start_line": 0,
                "end_line": 0,
            }

        normalized = raw_content.replace("\r\n", "\n")
        register_file_read(context.session_id, str(file_path), normalized)

        if normalized == "":
            return {
                "content": "File is empty.",
                "total_lines": 0,
                "lines_returned": 0,
                "start_line": 0,
                "end_line": 0,
            }

        lines = normalized.split("\n")
        total_lines = len(lines)
        start_line = int(offset) if offset is not None else 1
        line_limit = int(limit) if limit is not None else DEFAULT_MAX_LINES

        if start_line > total_lines:
            return {
                "is_error": True,
                "error": f"Start line {start_line} exceeds total lines {total_lines}.",
                "content": "",
                "total_lines": total_lines,
                "lines_returned": 0,
                "start_line": start_line,
                "end_line": start_line,
            }

        end_line = min(total_lines, start_line + line_limit - 1)
        selected_lines = lines[start_line - 1 : end_line]

        formatted_lines: list[str] = []
        total_chars = 0
        actual_end_line = start_line

        for index, line in enumerate(selected_lines):
            line_number = start_line + index
            truncated_line = (
                f"{line[:MAX_LINE_LENGTH]}... [truncated]"
                if len(line) > MAX_LINE_LENGTH
                else line
            )
            formatted = f"{line_number:>6}|{truncated_line}"
            projected_size = total_chars + len(formatted) + 1
            if projected_size > MAX_CONTENT_CHARS and formatted_lines:
                break
            formatted_lines.append(formatted)
            total_chars = projected_size
            actual_end_line = line_number

        content = "\n".join(formatted_lines)
        has_more = actual_end_line < total_lines
        was_truncated = actual_end_line < end_line
        if has_more or was_truncated:
            remaining = total_lines - actual_end_line
            content += (
                f"\n\n... [{remaining} more lines. "
                f"Use offset={actual_end_line + 1} to continue.] ..."
            )

        return {
            "content": content,
            "total_lines": total_lines,
            "lines_returned": len(formatted_lines),
            "start_line": start_line,
            "end_line": actual_end_line,
        }

    def _read_notebook(
        self, file_path: typing.Any, session_id: str
    ) -> dict[str, typing.Any]:
        """Read and format a Jupyter notebook."""
        try:
            raw_content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return {
                "is_error": True,
                "error": f"Failed to read notebook: {exc}",
                "content": "",
                "total_lines": 0,
                "lines_returned": 0,
                "start_line": 0,
                "end_line": 0,
            }

        try:
            notebook = _parse_notebook(raw_content)
            cells: list[dict[str, typing.Any]] = notebook.get("cells", [])
            formatted_cells: list[str] = []
            for idx, cell in enumerate(cells, start=1):
                cell_type = str(cell.get("cell_type", "unknown"))
                source = cell.get("source", "")
                if isinstance(source, list):
                    source_text = "".join(str(item) for item in source)
                else:
                    source_text = str(source)

                output_block = ""
                if cell_type == "code":
                    outputs = cell.get("outputs", [])
                    output_texts: list[str] = []
                    for output in outputs if isinstance(outputs, list) else []:
                        text_value = output.get("text")
                        if text_value:
                            if isinstance(text_value, list):
                                output_texts.append("".join(str(v) for v in text_value))
                            else:
                                output_texts.append(str(text_value))
                        data = output.get("data")
                        if isinstance(data, dict) and "text/plain" in data:
                            plain = data["text/plain"]
                            if isinstance(plain, list):
                                output_texts.append("".join(str(v) for v in plain))
                            else:
                                output_texts.append(str(plain))
                    if output_texts:
                        joined_output = "\n".join(output_texts)
                        output_block = f"\n\n[Output]\n{joined_output}"

                formatted_cells.append(
                    (
                        f"--- Cell {idx} ({cell_type}) ---\n"
                        f"{source_text}{output_block}"
                    ).rstrip()
                )

            content = (
                "\n\n".join(formatted_cells)
                if formatted_cells
                else "Notebook has no cells."
            )
            register_file_read(session_id, str(file_path), raw_content.replace("\r\n", "\n"))
            total_cells = len(formatted_cells)
            return {
                "content": content,
                "total_lines": total_cells,
                "lines_returned": total_cells,
                "start_line": 1 if total_cells else 0,
                "end_line": total_cells,
            }
        except Exception as exc:
            return {
                "is_error": True,
                "error": f"Failed to parse notebook: {exc}",
                "content": "",
                "total_lines": 0,
                "lines_returned": 0,
                "start_line": 0,
                "end_line": 0,
            }

    def _file_stat_error(
        self, file_path: typing.Any, exc: OSError
    ) -> dict[str, typing.Any]:
        """Build an error result for file stat failures."""
        if exc.errno == errno.ENOENT:
            suggestion = find_similar_file(str(file_path))
            message = f"File not found: {file_path}."
            if suggestion:
                message = f"{message} Did you mean: {suggestion}?"
        elif exc.errno == errno.EACCES:
            message = f"Permission denied: {file_path}"
        else:
            message = str(exc)
        return {
            "is_error": True,
            "error": message,
            "content": "",
            "total_lines": 0,
            "lines_returned": 0,
            "start_line": 0,
            "end_line": 0,
        }
