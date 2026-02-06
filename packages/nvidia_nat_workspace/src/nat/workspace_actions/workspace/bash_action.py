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
        pass

    async def __aenter__(self) -> BashAction:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def _resolve_env(self, env_overrides: object | None) -> dict[str, str]:
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

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Action requires a non-empty 'command' string.")

        timeout_seconds = args.get("timeout_seconds", 30)
        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("'timeout_seconds' must be a number.") from exc
        env = self._resolve_env(args.get("env"))

        process = await asyncio.create_subprocess_exec(
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            command,
            cwd=str(context.root_path),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        stdout_chunks: list[str] = []

        async def read_output() -> None:
            if process.stdout is None:
                return
            while True:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    return
                stdout_chunks.append(chunk.decode("utf-8", errors="replace"))

        timed_out = False
        output_task = asyncio.create_task(read_output())
        try:
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except TimeoutError:
                timed_out = True
                process.send_signal(signal.SIGINT)
                try:
                    await asyncio.wait_for(process.wait(), timeout=1)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        finally:
            if process.stdin is not None:
                process.stdin.close()

        await output_task
        stdout = "".join(stdout_chunks)
        exit_code = None if timed_out else process.returncode

        return {
            "stdout": stdout,
            "stderr": "",
            "exit_code": exit_code,
            "timed_out": timed_out,
        }
