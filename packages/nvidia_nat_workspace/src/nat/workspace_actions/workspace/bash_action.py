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
"""Bash action implementation."""

from __future__ import annotations

import asyncio
import os
import signal
import typing
import uuid

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema
from nat.workspace.types import WorkspaceAction


@register_workspace_action
class BashAction(WorkspaceAction):
    """Execute shell commands within the workspace session root."""

    name = "bash"
    description = "Run a shell command inside the workspace session root."
    parameters = [
        TypeSchema(type="string", description="command: Shell command to execute."),
        TypeSchema(type="number", description="timeout_seconds: Optional timeout in seconds."),
        TypeSchema(type="object", description="env: Optional environment variable overrides."),
    ]
    result = TypeSchema(
        type="object",
        description="Result payload with stdout, stderr, exit_code, and timed_out flags.",
    )

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> BashAction:
        if self._context is None:
            raise RuntimeError("BashAction missing ActionContext.")
        if self._process is None or self._process.returncode is not None:
            self._process = await asyncio.create_subprocess_exec(
                "bash",
                "--noprofile",
                "--norc",
                cwd=str(self._context.root_path),
                env=os.environ.copy(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except TimeoutError:
            self._process.kill()
        self._process = None

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Execute a shell command in the session root."""
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("BashAction is not initialized.")
        stdout_stream = process.stdout
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Action requires a non-empty 'command' string.")

        timeout_seconds = args.get("timeout_seconds", 30)
        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("'timeout_seconds' must be a number.") from exc

        async with self._lock:
            sentinel = f"__NAT_DONE_{uuid.uuid4().hex}__"
            script = f"{command}\n__nat_status__=$?\nprintf '\\n{sentinel}%s\\n' $__nat_status__\n"
            process.stdin.write(script.encode("utf-8"))
            await process.stdin.drain()

            stdout_chunks: list[str] = []
            exit_code: int | None = None

            async def read_output() -> tuple[str, int | None]:
                while True:
                    line = await stdout_stream.readline()
                    if not line:
                        return "".join(stdout_chunks), exit_code
                    decoded = line.decode("utf-8", errors="replace")
                    if decoded.startswith(sentinel):
                        status_text = decoded[len(sentinel):].strip()
                        if status_text.isdigit():
                            return "".join(stdout_chunks), int(status_text)
                        return "".join(stdout_chunks), None
                    stdout_chunks.append(decoded)

            timed_out = False
            try:
                stdout, exit_code = await asyncio.wait_for(read_output(), timeout=timeout_seconds)
            except TimeoutError:
                timed_out = True
                process.send_signal(signal.SIGINT)
                stdout = "".join(stdout_chunks)

        return {
            "stdout": stdout,
            "stderr": "",
            "exit_code": exit_code,
            "timed_out": timed_out,
        }
