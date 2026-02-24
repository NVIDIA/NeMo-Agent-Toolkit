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
"""Bash action implementation with separate streams, truncation, and diagnostics."""

from __future__ import annotations

import asyncio
import os
import signal
import time
import typing
from pathlib import Path

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction
from nat.workspace_actions.workspace.utils.bash_commands import (
    interpret_exit_code,
    is_image_output,
    is_long_running_command,
)

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600
MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB


@register_workspace_action
class BashAction(WorkspaceAction):
    """Execute shell commands within the workspace session root."""

    name = "bash"
    description = (
        "Execute a shell command in the terminal.\n"
        "\n"
        "Commands run synchronously with a configurable timeout (default 120s, max 600s).\n"
        "Output (stdout/stderr) is captured separately and returned.\n"
        "\n"
        "When NOT to use Bash:\n"
        "- For file reading: use the read action (not cat/head/tail)\n"
        "- For file editing: use the edit action (not sed/awk)\n"
        "- For file writing: use the write action (not echo/cat with heredoc)\n"
        "- For searching files: use the grep action (not grep/rg)\n"
        "- For finding files: use the glob action (not find/ls)\n"
        "\n"
        "Guidelines:\n"
        "- Always provide a description parameter explaining what the command does.\n"
        "- For commands that may take longer than 120s, set timeout_seconds appropriately.\n"
        "- Use absolute paths when possible to avoid working directory issues.\n"
        "- Chain related commands with && to ensure proper sequencing.\n"
        "\n"
        "Exit code interpretation:\n"
        "- 127: Command not found\n"
        "- 126: Command not executable\n"
        "- 137: Process killed (SIGKILL, possibly OOM)\n"
        "- 130: Process interrupted (Ctrl+C)\n"
        "- 143: Process terminated (SIGTERM)"
    )
    parameters = [
        TypeSchema(type="string", description="command: Shell command to execute."),
        TypeSchema(type="number", description="timeout_seconds: Optional timeout in seconds (default 120, max 600)."),
        TypeSchema(type="string", description="working_directory: Optional working directory relative to workspace root."),
        TypeSchema(type="object", description="env: Optional environment variable overrides."),
        TypeSchema(type="string", description="description: Optional short description of what the command does."),
    ]
    result = TypeSchema(
        type="object",
        description="Result with stdout, stderr, exit_code, timed_out, duration_ms, signal, and exit_code_interpretation.",
    )

    def __init__(self) -> None:
        pass

    async def __aenter__(self) -> BashAction:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_env(env_overrides: object | None) -> dict[str, str]:
        env = os.environ.copy()
        if env_overrides is None:
            return env
        if not isinstance(env_overrides, dict):
            raise ValueError("'env' must be a dictionary of string keys to string values.")
        for key, value in env_overrides.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("'env' must be a dictionary of string keys to string values.")
            env[key] = value
        return env

    @staticmethod
    def _truncate_output(output: str) -> str:
        """Keep the last *MAX_OUTPUT_BYTES* of output, prefixing with a truncation notice."""
        if is_image_output(output):
            return output
        encoded = output.encode("utf-8", errors="replace")
        if len(encoded) <= MAX_OUTPUT_BYTES:
            return output
        tail = encoded[-MAX_OUTPUT_BYTES:]
        tail_text = tail.decode("utf-8", errors="replace")
        removed_text = output[: max(len(output) - len(tail_text), 0)]
        removed_lines = max(removed_text.count("\n"), 1)
        return f"[...{removed_lines} lines truncated...]\n{tail_text}"

    def _resolve_cwd(self, context: ActionContext, working_directory: str | None) -> Path:
        root = context.root_path.resolve()
        if working_directory is None:
            return root
        path = Path(working_directory)
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve()
        # Ensure the resolved path is within the workspace root.
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"working_directory must be within the workspace root ({root})."
            ) from exc
        return resolved

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Action requires a non-empty 'command' string.")

        # Timeout
        timeout_seconds = args.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("'timeout_seconds' must be a number.") from exc
        timeout_seconds = min(timeout_seconds, MAX_TIMEOUT_SECONDS)

        # Working directory
        cwd = self._resolve_cwd(context, args.get("working_directory"))

        # Environment
        env = self._resolve_env(args.get("env"))

        # Long-running warning
        long_running_warning = ""
        if is_long_running_command(command):
            long_running_warning = (
                "Warning: command appears long-running. "
                "Consider running it with a larger timeout or in the background.\n"
            )

        start = time.perf_counter()
        signal_name: str | None = None

        process = await asyncio.create_subprocess_exec(
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            command,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            timed_out = True
            # Graceful shutdown: SIGTERM → wait 2s → SIGKILL
            process.terminate()
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=2,
                )
            except TimeoutError:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
            signal_name = "SIGTERM"

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

        exit_code: int | None = None if timed_out else process.returncode

        # Detect signal from negative return code
        if exit_code is not None and exit_code < 0 and signal_name is None:
            try:
                signal_name = signal.Signals(-exit_code).name
            except (ValueError, KeyError):
                signal_name = None

        # Truncate large outputs
        stdout = self._truncate_output(stdout)
        stderr = self._truncate_output(stderr)

        # Prepend long-running warning
        if long_running_warning:
            stderr = f"{long_running_warning}{stderr}"

        # Timeout annotation
        if timed_out:
            timeout_msg = f"Command timed out after {timeout_seconds}s."
            stderr = f"{stderr}\n{timeout_msg}" if stderr else timeout_msg

        interpretation = interpret_exit_code(exit_code, signal_name)
        duration_ms = int((time.perf_counter() - start) * 1000)

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": duration_ms,
            "signal": signal_name,
            "exit_code_interpretation": interpretation,
        }
