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
"""Glob action implementation for file pattern matching."""

from __future__ import annotations

import asyncio
import glob as glob_module
import os
import typing
from pathlib import Path, PurePosixPath

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction
from nat.workspace_actions.workspace.utils.path_utils import resolve_workspace_path
from nat.workspace_actions.workspace.utils.ripgrep_utils import get_ripgrep_path

MAX_RESULTS = 1000
_FALLBACK_EXCLUDED_DIRS: set[str] = {".git", "node_modules", "dist"}


def _normalize_pattern(pattern: str) -> str:
    """Prepend ``**/`` when the pattern does not already start with it."""
    if not pattern.startswith("**/") and not pattern.startswith("/"):
        return f"**/{pattern}"
    return pattern


def _extract_search_path_from_pattern(pattern: str) -> tuple[Path | None, str]:
    """Split an absolute glob pattern into a concrete directory and a relative glob."""
    expanded = os.path.expanduser(pattern)
    if not os.path.isabs(expanded):
        return None, expanded

    path_obj = Path(expanded)
    parts = path_obj.parts

    search_parts: list[str] = []
    for part in parts:
        if glob_module.has_magic(part):
            break
        search_parts.append(part)

    if not search_parts:
        search_path = Path("/")
        adjusted_pattern = expanded.lstrip("/")
    else:
        search_path = Path(*search_parts)
        remaining_parts = parts[len(search_parts):]
        adjusted_pattern = str(Path(*remaining_parts)) if remaining_parts else "**/*"

    return search_path.resolve(), adjusted_pattern


def _resolve_search_target(
    glob_pattern: str,
    target_directory: str | None,
    root_path: Path,
) -> tuple[Path, str]:
    """Resolve the search directory and normalise the glob pattern."""
    pattern = glob_pattern

    if target_directory:
        search_directory = resolve_workspace_path(target_directory, root_path)
        if os.path.isabs(pattern):
            extracted_dir, extracted_pattern = _extract_search_path_from_pattern(pattern)
            if extracted_dir is not None:
                search_directory = extracted_dir
                pattern = extracted_pattern
    else:
        extracted_dir, extracted_pattern = _extract_search_path_from_pattern(pattern)
        search_directory = extracted_dir if extracted_dir is not None else root_path.resolve()
        pattern = extracted_pattern

    return search_directory, _normalize_pattern(pattern)


async def _glob_with_ripgrep(search_directory: Path, pattern: str) -> list[str]:
    """Run ripgrep in ``--files`` mode with a glob filter (async)."""
    rg_path = get_ripgrep_path()
    if rg_path is None:
        raise RuntimeError("ripgrep is not available")

    process = await asyncio.create_subprocess_exec(
        rg_path,
        "--files",
        "--glob",
        pattern,
        "--max-count",
        str(MAX_RESULTS),
        "--sort",
        "modified",
        str(search_directory),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=30)
    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

    if process.returncode not in (0, 1):
        raise RuntimeError(stderr.strip() or "ripgrep glob search failed")
    if process.returncode == 1:
        return []

    files: list[str] = []
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = (search_directory / candidate_path).resolve()
        if candidate_path.is_file():
            files.append(str(candidate_path))

    return files[:MAX_RESULTS]


def _glob_with_python(search_directory: Path, pattern: str) -> list[str]:
    """Fallback glob using ``os.walk`` and ``PurePosixPath.match``."""
    normalized_pattern = pattern.replace("\\", "/")
    candidate_patterns = [normalized_pattern]
    if normalized_pattern.startswith("**/"):
        candidate_patterns.append(normalized_pattern[3:])

    matches: list[Path] = []
    for root, dirs, files in os.walk(search_directory):
        dirs[:] = [d for d in dirs if d not in _FALLBACK_EXCLUDED_DIRS]
        for file_name in files:
            candidate = Path(root) / file_name
            relative = candidate.relative_to(search_directory).as_posix()
            if any(
                PurePosixPath(relative).match(pattern_candidate)
                for pattern_candidate in candidate_patterns
            ):
                matches.append(candidate)

    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path.resolve()) for path in matches[:MAX_RESULTS]]


@register_workspace_action
class GlobAction(WorkspaceAction):
    """Find files matching a glob pattern in the workspace."""

    name = "glob"
    description = (
        "Find files matching a glob pattern in the workspace.\n"
        "Returns matching file paths sorted by modification time (most recent first).\n"
        "\n"
        "Features:\n"
        "- High-performance matching using ripgrep with automatic fallback\n"
        "- Sorts results by modification time (most recent first)\n"
        "- Supports standard glob patterns (*, **, ?, etc.)\n"
        "- Automatically skips .git, node_modules, dist directories\n"
        "- Patterns not starting with '**/' are automatically prepended with '**/' for recursive matching\n"
        "\n"
        "Use this tool when you need to find files by name patterns."
    )
    parameters = [
        TypeSchema(type="string", description="glob_pattern: Glob pattern to match files."),
        TypeSchema(type="string", description="target_directory: Optional absolute directory path to search in."),
    ]
    result = TypeSchema(
        type="object",
        description="Result with files, count, pattern, and search_directory.",
    )

    def __init__(self) -> None:
        pass

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        glob_pattern = args.get("glob_pattern")
        if not isinstance(glob_pattern, str) or not glob_pattern.strip():
            return {
                "is_error": True,
                "error": "A non-empty 'glob_pattern' string is required.",
                "files": [],
                "count": 0,
                "pattern": str(glob_pattern),
                "search_directory": str(context.root_path),
            }

        target_directory = args.get("target_directory")

        try:
            search_directory, pattern = _resolve_search_target(
                glob_pattern, target_directory, context.root_path,
            )

            if not search_directory.is_dir():
                return {
                    "is_error": True,
                    "error": f"Path is not a directory: {search_directory}",
                    "files": [],
                    "count": 0,
                    "pattern": pattern,
                    "search_directory": str(search_directory),
                }

            files: list[str]
            if get_ripgrep_path() is not None:
                try:
                    files = await _glob_with_ripgrep(search_directory, pattern)
                except Exception:
                    files = _glob_with_python(search_directory, pattern)
            else:
                files = _glob_with_python(search_directory, pattern)

            return {
                "files": files,
                "count": len(files),
                "pattern": pattern,
                "search_directory": str(search_directory),
            }
        except Exception as exc:
            return {
                "is_error": True,
                "error": str(exc),
                "files": [],
                "count": 0,
                "pattern": glob_pattern,
                "search_directory": str(context.root_path),
            }
