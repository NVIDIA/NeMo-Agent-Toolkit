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
"""Edit action implementation using advanced replacement strategies with fuzzy matching fallbacks."""

from __future__ import annotations

import errno
import typing
from pathlib import Path

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction
from nat.workspace_actions.workspace.utils.file_state import (
    clear_file_read_state,
    get_file_read_state,
    register_file_read,
)
from nat.workspace_actions.workspace.utils.levenshtein import find_similar_file
from nat.workspace_actions.workspace.utils.path_utils import resolve_workspace_path
from nat.workspace_actions.workspace.utils.replacers import (
    has_multiple_matches,
    perform_advanced_replacement,
)


SMART_QUOTE_REPLACEMENTS: dict[str, str] = {
    "\u2019": "'",
    "\u2018": "'",
    "\u201C": '"',
    "\u201D": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
}

_REPLACER_STRATEGIES = [
    "SimpleReplacer",
    "LineTrimmedReplacer",
    "BlockAnchorReplacer",
    "WhitespaceNormalizedReplacer",
    "IndentationFlexibleReplacer",
    "EscapeNormalizedReplacer",
    "TrimmedBoundaryReplacer",
    "ContextAwareReplacer",
    "MultiOccurrenceReplacer",
]


def _normalize_line_endings(text: str) -> str:
    """Normalize CRLF line endings to LF."""
    return text.replace("\r\n", "\n")


def _normalize_smart_quotes(text: str) -> str:
    """Replace smart/curly quotes and dashes with ASCII equivalents."""
    normalized = text
    for source, target in SMART_QUOTE_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized


def _generate_edit_snippet(old_content: str, new_content: str) -> str | None:
    """Generate a context diff snippet highlighting changed lines."""
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    max_length = max(len(old_lines), len(new_lines))

    first_changed = -1
    last_changed = -1
    for index in range(max_length):
        old_line = old_lines[index] if index < len(old_lines) else None
        new_line = new_lines[index] if index < len(new_lines) else None
        if old_line != new_line:
            if first_changed == -1:
                first_changed = index
            last_changed = index

    if first_changed == -1:
        return None

    context = 3
    start = max(0, first_changed - context)
    end = min(max(len(new_lines) - 1, 0), last_changed + context)

    snippet_lines: list[str] = []
    for index in range(start, end + 1):
        line_number = index + 1
        line_text = new_lines[index] if index < len(new_lines) else ""
        marker = "\u2192" if first_changed <= index <= last_changed else " "
        snippet_lines.append(f"{line_number:>4}{marker} {line_text}")
    return "\n".join(snippet_lines)


def _get_file_mtime(file_path: Path) -> float | None:
    """Get file modification time, or None on error."""
    try:
        return file_path.stat().st_mtime
    except OSError:
        return None


def _format_not_found(path: Path) -> str:
    """Format a file-not-found message with optional similar-file suggestion."""
    suggestion = find_similar_file(str(path))
    if suggestion:
        return f"File not found: {path}. Did you mean: {suggestion}?"
    return f"File not found: {path}."


def perform_edit(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
    root_path: Path,
    session_id: str,
) -> dict[str, typing.Any]:
    """Perform a single edit operation on a file.

    This is the core edit logic, extracted so it can be reused by multi_edit_action.

    Returns a dict with keys: path, replacements, modified, snippet, replacer_used.
    Raises ValueError or FileNotFoundError on failure.
    """
    path = resolve_workspace_path(file_path, root_path)

    if old_string != "" and old_string == new_string:
        raise ValueError("old_string and new_string are identical")

    # --- create mode ---
    if old_string == "":
        if path.exists():
            try:
                existing_content = path.read_text(encoding="utf-8")
                if existing_content.strip():
                    raise ValueError(
                        "Cannot create - file already exists and is not empty."
                    )
            except OSError:
                # If file is unreadable we still attempt to overwrite in create mode.
                pass

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_string, encoding="utf-8")
        clear_file_read_state(session_id, str(path))
        return {
            "path": str(path),
            "replacements": 1,
            "modified": True,
            "snippet": _generate_edit_snippet("", new_string),
            "replacer_used": None,
        }

    # --- file must exist for replacement ---
    if not path.exists():
        raise FileNotFoundError(_format_not_found(path))

    # --- read-before-edit validation ---
    read_state = get_file_read_state(session_id, str(path))
    if read_state is None:
        content_for_state = _normalize_line_endings(path.read_text(encoding="utf-8"))
        register_file_read(session_id, str(path), content_for_state)
        read_state = get_file_read_state(session_id, str(path))

    current_mtime = _get_file_mtime(path)
    if read_state is not None and current_mtime is not None:
        if current_mtime > read_state.timestamp:
            current_content = _normalize_line_endings(path.read_text(encoding="utf-8"))
            if current_content != read_state.content:
                raise ValueError("File has been modified since you last read it.")

    content = _normalize_line_endings(path.read_text(encoding="utf-8"))
    original_content = content

    normalized_old = _normalize_smart_quotes(old_string)
    normalized_new = _normalize_smart_quotes(new_string)

    # --- try advanced replacement ---
    replacement = perform_advanced_replacement(
        content,
        old_string,
        new_string,
        replace_all,
    )
    if replacement is None and normalized_old != old_string:
        replacement = perform_advanced_replacement(
            content,
            normalized_old,
            normalized_new,
            replace_all,
        )

    if replacement is None:
        multiple_matches = has_multiple_matches(content, old_string) or (
            normalized_old != old_string
            and has_multiple_matches(content, normalized_old)
        )
        if multiple_matches and not replace_all:
            raise ValueError(
                "String matches multiple locations. Add more context or use "
                "replace_all=True."
            )
        strategies = ", ".join(_REPLACER_STRATEGIES)
        raise ValueError(
            "String not found in file. Matching strategies attempted: "
            f"{strategies}."
        )

    path.write_text(replacement.new_content, encoding="utf-8")
    clear_file_read_state(session_id, str(path))
    snippet = _generate_edit_snippet(original_content, replacement.new_content)

    replacer_used: str | None = None
    if replacement.replacer_used != "SimpleReplacer":
        replacer_used = replacement.replacer_used

    return {
        "path": str(path),
        "replacements": replacement.replaced_count,
        "modified": True,
        "snippet": snippet,
        "replacer_used": replacer_used,
    }


@register_workspace_action
class EditAction(WorkspaceAction):
    """Edit file content via string replacement with fuzzy matching fallbacks."""

    name = "edit"
    description = (
        "Edit file content via string replacement with context snippets showing changes.\n"
        "\n"
        "Uses 9 advanced matching strategies for robust replacements:\n"
        "1. Exact match\n"
        "2. Line-trimmed (ignores leading/trailing whitespace per line)\n"
        "3. Block anchor (first/last lines as anchors)\n"
        "4. Whitespace normalized\n"
        "5. Indentation flexible\n"
        "6. Escape normalized\n"
        "7. Trimmed boundary\n"
        "8. Context aware\n"
        "9. Multi-occurrence (handles replace_all mode)\n"
        "\n"
        "Guidelines:\n"
        "- The old_string should match the file content (tries multiple strategies if exact match fails)\n"
        "- Include enough context (3-5 lines before/after) to ensure uniqueness\n"
        "- If old_string appears multiple times, add more context or use replace_all=true\n"
        "\n"
        "File creation mode (old_string=''):\n"
        "- Creates a new file if it doesn't exist\n"
        "- Fails if file exists and is not empty (prevents accidental overwrites)\n"
        "- Creates parent directories automatically\n"
        "\n"
        "The tool will:\n"
        "- Show a context snippet with changed lines marked\n"
        "- Detect if the file was modified since you last read it\n"
        "- Suggest similar filenames if the file is not found\n"
        "- Normalize smart quotes and Windows line endings automatically"
    )
    parameters = [
        TypeSchema(type="string", description="file_path: Absolute or workspace-relative path to modify."),
        TypeSchema(type="string", description="old_string: Text to replace. Empty string enables create mode."),
        TypeSchema(type="string", description="new_string: Replacement text."),
        TypeSchema(type="boolean", description="replace_all: Replace all matches instead of requiring uniqueness. Default false."),
    ]
    result = TypeSchema(
        type="object",
        description="Result payload with path, replacements, modified, snippet, and replacer_used.",
    )

    def __init__(self) -> None:
        pass

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        file_path_str = args.get("file_path")
        if not isinstance(file_path_str, str) or not file_path_str.strip():
            return {"is_error": True, "error": "Action requires a non-empty 'file_path' string."}

        old_string = args.get("old_string", "")
        if not isinstance(old_string, str):
            return {"is_error": True, "error": "'old_string' must be a string."}

        new_string = args.get("new_string", "")
        if not isinstance(new_string, str):
            return {"is_error": True, "error": "'new_string' must be a string."}

        replace_all = bool(args.get("replace_all", False))

        try:
            result = perform_edit(
                file_path=file_path_str,
                old_string=old_string,
                new_string=new_string,
                replace_all=replace_all,
                root_path=context.root_path,
                session_id=context.session_id,
            )
            return result
        except FileNotFoundError as exc:
            path = resolve_workspace_path(file_path_str, context.root_path)
            return {
                "is_error": True,
                "error": str(exc),
                "path": str(path),
                "replacements": 0,
                "modified": False,
            }
        except OSError as exc:
            path = resolve_workspace_path(file_path_str, context.root_path)
            if exc.errno == errno.EACCES:
                message = f"Permission denied: {file_path_str}"
            else:
                message = str(exc)
            return {
                "is_error": True,
                "error": message,
                "path": str(path),
                "replacements": 0,
                "modified": False,
            }
        except Exception as exc:
            path = resolve_workspace_path(file_path_str, context.root_path)
            return {
                "is_error": True,
                "error": str(exc),
                "path": str(path),
                "replacements": 0,
                "modified": False,
            }
