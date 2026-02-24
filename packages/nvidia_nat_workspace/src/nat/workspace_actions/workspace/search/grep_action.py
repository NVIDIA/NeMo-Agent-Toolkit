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
"""Grep action implementation using ripgrep."""

from __future__ import annotations

import asyncio
import re
import typing
from pathlib import Path

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction
from nat.workspace_actions.workspace.utils.ripgrep_utils import get_ripgrep_path


def _get_ripgrep_timeout() -> int:
    """Use a longer timeout under WSL due to slower filesystem behaviour."""
    try:
        release = Path("/proc/version").read_text(encoding="utf-8").lower()
        if "microsoft" in release or "wsl" in release:
            return 60
    except OSError:
        pass
    return 10


def _is_eagain_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return (
        "os error 11" in lowered
        or "resource temporarily unavailable" in lowered
        or "eagain" in lowered
    )


def _resolve_search_path(path_value: str | None, root_path: Path) -> Path:
    """Resolve a search path relative to the workspace root."""
    if path_value is None:
        return root_path
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root_path / path
    return path.resolve()


def _build_args(args: dict[str, typing.Any], search_path: Path) -> list[str]:
    """Build the ripgrep argument list from action parameters."""
    rg_args: list[str] = []

    output_mode = args.get("output_mode", "content")

    if output_mode == "files_with_matches":
        rg_args.append("-l")
    elif output_mode == "count":
        rg_args.append("-c")
    else:
        rg_args.append("-n")

    if args.get("case_insensitive"):
        rg_args.append("-i")

    if output_mode == "content":
        context_after = args.get("context_after")
        if context_after is not None:
            rg_args.extend(["-A", str(int(context_after))])
        context_before = args.get("context_before")
        if context_before is not None:
            rg_args.extend(["-B", str(int(context_before))])
        context = args.get("context")
        if context is not None:
            rg_args.extend(["-C", str(int(context))])

    if args.get("multiline"):
        rg_args.extend(["-U", "--multiline-dotall"])

    rg_type = args.get("type")
    if rg_type:
        rg_args.extend(["--type", str(rg_type)])

    glob = args.get("glob")
    if glob:
        rg_args.extend(["--glob", str(glob)])

    rg_args.extend(["--max-columns", "500"])
    rg_args.extend(["--", str(args["pattern"]), str(search_path)])
    return rg_args


async def _run_ripgrep(
    args: list[str],
    cwd: str,
    timeout_seconds: int,
    *,
    is_retry: bool = False,
    use_single_threaded: bool = False,
) -> tuple[str, str, int, bool]:
    """Execute ripgrep asynchronously. Returns (stdout, stderr, exit_code, use_single_threaded)."""
    rg_path = get_ripgrep_path()
    if rg_path is None:
        return "", "ripgrep not found.", -1, use_single_threaded

    final_args = args
    if use_single_threaded and not is_retry:
        final_args = ["-j", "1", *args]

    process = await asyncio.create_subprocess_exec(
        rg_path,
        *final_args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds,
        )
        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        exit_code = process.returncode if process.returncode is not None else -1
    except asyncio.TimeoutError:
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        timeout_message = "Timeout: ripgrep exceeded time limit"
        stderr = (stderr + "\n" if stderr else "") + timeout_message
        return stdout, stderr, -1, use_single_threaded

    if not is_retry and exit_code not in (0, 1) and _is_eagain_error(stderr):
        return await _run_ripgrep(
            ["-j", "1", *args],
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            is_retry=True,
            use_single_threaded=True,
        )

    return stdout, stderr, exit_code, use_single_threaded


def _apply_pagination(
    stdout: str,
    offset: int,
    head_limit: int | None,
) -> tuple[str, bool, int]:
    """Slice output lines by offset and head_limit."""
    lines = stdout.splitlines()
    total_lines = len(lines)
    truncated = False

    processed = lines
    if offset > 0:
        processed = processed[offset:]

    if head_limit is not None and head_limit >= 0 and len(processed) > head_limit:
        processed = processed[:head_limit]
        truncated = True

    return "\n".join(processed), truncated, total_lines


def _count_results(
    output: str,
    output_mode: str,
) -> tuple[int, int]:
    """Parse ripgrep output to count matches and files."""
    lines = [line for line in output.splitlines() if line.strip()]

    if output_mode == "files_with_matches":
        count = len(lines)
        return count, count

    if output_mode == "count":
        match_count = 0
        files_matched = 0
        for line in lines:
            match = re.search(r":(\d+)$", line)
            if match:
                match_count += int(match.group(1))
                files_matched += 1
        return match_count, files_matched

    # content mode
    file_set: set[str] = set()
    match_count = 0
    for line in lines:
        match = re.match(r"^([^:]+):(\d+):", line)
        if match:
            file_set.add(match.group(1))
            match_count += 1
    return match_count, len(file_set)


@register_workspace_action
class GrepAction(WorkspaceAction):
    """Search file contents using ripgrep."""

    name = "grep"
    description = (
        "Search file contents using ripgrep.\n"
        "\n"
        "Usage:\n"
        "- ALWAYS use this grep action for search tasks. NEVER invoke grep or rg via bash.\n"
        "- Supports full regex syntax (e.g., 'log.*Error', 'function\\\\s+\\\\w+', 'import.*from')\n"
        "- Filter files with glob parameter (e.g., '*.js', '**/*.tsx') or type parameter (e.g., 'py', 'js', 'rust')\n"
        "- Output modes: 'content' (default), 'files_with_matches', 'count'\n"
        "\n"
        "Pattern syntax (ripgrep, not grep):\n"
        "- Literal braces need escaping: use 'interface\\\\{\\\\}' to find 'interface{}' in Go\n"
        "- Word boundaries: use '\\\\bword\\\\b' to match whole words only\n"
        "\n"
        "Multiline matching:\n"
        "- By default patterns match within single lines only\n"
        "- For cross-line patterns, set multiline=true\n"
        "\n"
        "Best practices:\n"
        "- Call multiple grep actions in parallel when searching for different patterns\n"
        "- Use type parameter for efficiency when searching specific file types\n"
        "- Use head_limit to cap results when you only need a few matches\n"
        "- Use context/context_before/context_after to see surrounding code"
    )
    parameters = [
        TypeSchema(type="string", description="pattern: Regex pattern to search for."),
        TypeSchema(type="string", description="path: File or directory to search in. Defaults to workspace root."),
        TypeSchema(type="string", description="glob: Glob file filter, e.g. '*.py'."),
        TypeSchema(type="string", description="type: Ripgrep file type, e.g. py, js, rust."),
        TypeSchema(type="boolean", description="case_insensitive: Case-insensitive search. Default false."),
        TypeSchema(type="number", description="context_after: Lines to show after each match."),
        TypeSchema(type="number", description="context_before: Lines to show before each match."),
        TypeSchema(type="number", description="context: Lines to show before and after each match."),
        TypeSchema(type="boolean", description="multiline: Enable multiline matching. Default false."),
        TypeSchema(type="string", description="output_mode: content, files_with_matches, or count. Default content."),
        TypeSchema(type="number", description="head_limit: Limit output entries after applying offset."),
        TypeSchema(type="number", description="offset: Skip first N output entries. Default 0."),
    ]
    result = TypeSchema(
        type="object",
        description="Result with output, match_count, files_matched, truncated, total_lines, offset.",
    )

    def __init__(self) -> None:
        self._use_single_threaded_mode = False

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return {
                "is_error": True,
                "error": "A non-empty 'pattern' string is required.",
                "output": "",
                "match_count": 0,
                "files_matched": 0,
                "truncated": False,
                "total_lines": None,
                "offset": None,
            }

        if get_ripgrep_path() is None:
            return {
                "is_error": True,
                "error": "ripgrep (rg) is not available in PATH.",
                "output": "",
                "match_count": 0,
                "files_matched": 0,
                "truncated": False,
                "total_lines": None,
                "offset": None,
            }

        search_path = _resolve_search_path(args.get("path"), context.root_path)
        rg_args = _build_args(args, search_path)
        timeout_seconds = _get_ripgrep_timeout()

        stdout, stderr, exit_code, self._use_single_threaded_mode = await _run_ripgrep(
            rg_args,
            cwd=str(context.root_path),
            timeout_seconds=timeout_seconds,
            use_single_threaded=self._use_single_threaded_mode,
        )

        if exit_code not in (0, 1):
            return {
                "is_error": True,
                "error": stderr.strip() or f"ripgrep exited with code {exit_code}",
                "output": stdout,
                "match_count": 0,
                "files_matched": 0,
                "truncated": False,
                "total_lines": None,
                "offset": None,
            }

        offset = int(args.get("offset", 0))
        head_limit_raw = args.get("head_limit")
        head_limit = int(head_limit_raw) if head_limit_raw is not None else None

        output, truncated, total_lines = _apply_pagination(
            stdout=stdout,
            offset=offset,
            head_limit=head_limit,
        )

        output_mode = args.get("output_mode", "content")
        match_count, files_matched = _count_results(output, output_mode)
        displayed_output = output if output else "No matches found."
        include_pagination = offset > 0 or head_limit is not None

        return {
            "output": displayed_output,
            "match_count": match_count,
            "files_matched": files_matched,
            "truncated": truncated,
            "total_lines": total_lines if include_pagination else None,
            "offset": offset if include_pagination else None,
        }
