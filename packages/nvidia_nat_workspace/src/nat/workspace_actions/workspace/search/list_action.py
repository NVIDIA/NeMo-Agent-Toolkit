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
"""List action implementation for directory tree views."""

from __future__ import annotations

import asyncio
import os
import typing
from pathlib import Path, PurePosixPath

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction
from nat.workspace_actions.workspace.utils.ripgrep_utils import get_ripgrep_path

MAX_FILES = 200

DEFAULT_IGNORE_PATTERNS = [
    "node_modules/",
    "__pycache__/",
    ".git/",
    "dist/",
    "build/",
    "target/",
    "vendor/",
    "bin/",
    "obj/",
    ".idea/",
    ".vscode/",
    ".zig-cache/",
    "zig-out/",
    ".coverage/",
    "coverage/",
    "tmp/",
    "temp/",
    ".cache/",
    "cache/",
    "logs/",
    ".venv/",
    "venv/",
    "env/",
    ".next/",
    ".nuxt/",
    ".output/",
    ".turbo/",
    ".parcel-cache/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".tox/",
    ".eggs/",
    "*.egg-info/",
    ".sass-cache/",
    ".gradle/",
    ".mvn/",
    ".cargo/",
    "Pods/",
    ".dart_tool/",
    ".pub-cache/",
]


def _resolve_root(path_value: str | None, root_path: Path) -> Path:
    """Resolve the listing root directory."""
    if path_value is None:
        return root_path.resolve()
    root = Path(path_value).expanduser()
    if not root.is_absolute():
        root = root_path / root
    return root.resolve()


async def _get_files_with_ripgrep(
    root: Path,
    ignore_patterns: list[str],
    show_hidden: bool,
    max_depth: int | None,
) -> tuple[list[str], bool]:
    """List files using ripgrep (async)."""
    rg_path = get_ripgrep_path()
    if rg_path is None:
        raise RuntimeError("ripgrep is not available")

    cmd = [rg_path, "--files", "--sort", "path"]
    for pattern in ignore_patterns:
        cmd.extend(["--glob", f"!{pattern}"])
    if show_hidden:
        cmd.append("--hidden")
    if max_depth is not None:
        cmd.extend(["--max-depth", str(max_depth)])
    cmd.append(str(root))

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=30)
    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

    if process.returncode not in (0, 1):
        raise RuntimeError(stderr.strip() or "ripgrep list search failed")

    files: list[str] = []
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = (root / candidate_path).resolve()
        if candidate_path.is_file():
            files.append(str(candidate_path))

    truncated = len(files) > MAX_FILES
    return files[:MAX_FILES], truncated


def _is_ignored(path_value: str, ignore_patterns: list[str], *, is_dir: bool) -> bool:
    """Check whether a relative path matches any ignore pattern."""
    for pattern in ignore_patterns:
        if pattern.endswith("/"):
            prefix = pattern[:-1]
            if path_value == prefix or path_value.startswith(f"{prefix}/"):
                return True
            continue

        if PurePosixPath(path_value).match(pattern):
            return True

        if is_dir and PurePosixPath(f"{path_value}/").match(pattern):
            return True

    return False


def _get_files_with_walk(
    root: Path,
    ignore_patterns: list[str],
    show_hidden: bool,
    max_depth: int | None,
) -> tuple[list[str], bool]:
    """Fallback file listing using ``os.walk``."""
    matches: list[str] = []
    truncated = False

    for current_root, dirs, files in os.walk(root, topdown=True):
        current_root_path = Path(current_root)
        relative_root = current_root_path.relative_to(root)
        depth = 0 if str(relative_root) == "." else len(relative_root.parts)

        if max_depth is not None and depth >= max_depth:
            dirs[:] = []

        filtered_dirs: list[str] = []
        for dir_name in dirs:
            if not show_hidden and dir_name.startswith("."):
                continue
            rel_dir = (current_root_path / dir_name).relative_to(root).as_posix()
            if _is_ignored(rel_dir, ignore_patterns, is_dir=True):
                continue
            filtered_dirs.append(dir_name)
        dirs[:] = filtered_dirs

        for file_name in files:
            if not show_hidden and file_name.startswith("."):
                continue
            full_path = current_root_path / file_name
            rel_path = full_path.relative_to(root).as_posix()
            if _is_ignored(rel_path, ignore_patterns, is_dir=False):
                continue

            matches.append(str(full_path.resolve()))
            if len(matches) >= MAX_FILES:
                truncated = True
                return matches, truncated

    return matches, truncated


async def _get_files(
    root: Path,
    ignore_patterns: list[str],
    show_hidden: bool,
    max_depth: int | None,
) -> tuple[list[str], bool]:
    """Retrieve file list, preferring ripgrep with os.walk fallback."""
    if get_ripgrep_path() is not None:
        try:
            return await _get_files_with_ripgrep(
                root=root,
                ignore_patterns=ignore_patterns,
                show_hidden=show_hidden,
                max_depth=max_depth,
            )
        except Exception:
            pass

    return _get_files_with_walk(
        root=root,
        ignore_patterns=ignore_patterns,
        show_hidden=show_hidden,
        max_depth=max_depth,
    )


def _build_tree(files: list[str], root: Path) -> tuple[str, int]:
    """Construct a tree string and return the directory count."""
    tree: dict[str, typing.Any] = {"dirs": {}, "files": []}
    directories: set[str] = set()

    for file_path in files:
        relative = Path(file_path).resolve().relative_to(root).parts
        node = tree
        path_accumulator: list[str] = []

        for part in relative[:-1]:
            path_accumulator.append(part)
            directories.add("/".join(path_accumulator))
            node = node["dirs"].setdefault(part, {"dirs": {}, "files": []})

        node["files"].append(relative[-1])

    lines = [f"{root.name or str(root)}/"]
    lines.extend(_render_node(tree, prefix=""))
    return "\n".join(lines), len(directories)


def _render_node(node: dict[str, typing.Any], prefix: str) -> list[str]:
    """Render a tree node with box-drawing connectors."""
    lines: list[str] = []

    dir_entries = sorted(node["dirs"].items(), key=lambda item: item[0])
    file_entries = sorted(node["files"])
    children: list[tuple[str, str, typing.Any]] = [
        *(("dir", name, child) for name, child in dir_entries),
        *(("file", name, None) for name in file_entries),
    ]

    for index, (entry_type, name, child) in enumerate(children):
        is_last = index == len(children) - 1
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        next_prefix = f"{prefix}{'    ' if is_last else '\u2502   '}"

        if entry_type == "dir":
            lines.append(f"{prefix}{connector}{name}/")
            lines.extend(_render_node(child, next_prefix))
        else:
            lines.append(f"{prefix}{connector}{name}")

    return lines


@register_workspace_action
class ListAction(WorkspaceAction):
    """List directory contents as a tree view."""

    name = "list"
    description = (
        "List directory contents as a tree view.\n"
        "\n"
        "Features:\n"
        "- Tree-style output showing directory hierarchy\n"
        "- Smart ignore patterns for common build artifacts\n"
        "- Supports hidden files and depth limiting\n"
        "\n"
        "Default ignored patterns include:\n"
        "- node_modules/, dist/, build/, target/\n"
        "- .git/, .vscode/, .idea/\n"
        "- __pycache__/, .venv/, venv/\n"
        "- coverage/, .cache/, logs/\n"
        "- And many more common artifacts\n"
        "\n"
        "Use the ignore parameter to add custom patterns.\n"
        "Use max_depth to limit traversal depth (1-10)."
    )
    parameters = [
        TypeSchema(type="string", description="path: Directory to list. Defaults to workspace root."),
        TypeSchema(type="array", description="ignore: Additional ignore glob patterns."),
        TypeSchema(type="boolean", description="show_hidden: Include hidden files/directories. Default false."),
        TypeSchema(type="number", description="max_depth: Maximum directory depth to traverse (1-10)."),
    ]
    result = TypeSchema(
        type="object",
        description="Result with root, tree, file_count, dir_count, and truncated.",
    )

    def __init__(self) -> None:
        pass

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        try:
            root = _resolve_root(args.get("path"), context.root_path)

            if not root.is_dir():
                return {
                    "is_error": True,
                    "error": f"Path is not a directory: {root}",
                    "root": str(root),
                    "tree": "",
                    "file_count": 0,
                    "dir_count": 0,
                    "truncated": False,
                }

            ignore_patterns = [*DEFAULT_IGNORE_PATTERNS, *(args.get("ignore") or [])]

            show_hidden = bool(args.get("show_hidden", False))

            max_depth_raw = args.get("max_depth")
            max_depth: int | None = None
            if max_depth_raw is not None:
                max_depth = max(1, min(10, int(max_depth_raw)))

            files, truncated = await _get_files(
                root=root,
                ignore_patterns=ignore_patterns,
                show_hidden=show_hidden,
                max_depth=max_depth,
            )

            if not files:
                tree = f"{root.name or str(root)}/\n  (empty or all files ignored)"
                return {
                    "root": str(root),
                    "tree": tree,
                    "file_count": 0,
                    "dir_count": 0,
                    "truncated": False,
                }

            tree, dir_count = _build_tree(files=files, root=root)
            if truncated:
                tree = f"{tree}\n... (truncated, showing first {MAX_FILES} files)"

            return {
                "root": str(root),
                "tree": tree,
                "file_count": len(files),
                "dir_count": dir_count,
                "truncated": truncated,
            }
        except Exception as exc:
            return {
                "is_error": True,
                "error": str(exc),
                "root": str(context.root_path),
                "tree": "",
                "file_count": 0,
                "dir_count": 0,
                "truncated": False,
            }
