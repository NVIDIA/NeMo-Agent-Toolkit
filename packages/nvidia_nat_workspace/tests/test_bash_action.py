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
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from nat.data_models.workspace import ActionContext
from nat.workspace_actions.workspace.terminal.bash_action import BashAction


@pytest.fixture(name="fixture_action_context")
def fixture_action_context(tmp_path: Path) -> ActionContext:
    return ActionContext(session_id="test-session", root_path=tmp_path.resolve())


@pytest.fixture(name="fixture_bash_action")
def fixture_bash_action() -> BashAction:
    return BashAction()


async def _wait_for_marker(marker_path: Path, timeout_seconds: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if marker_path.exists():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Marker file was not created in time.")


async def test_bash_action_executes_command(
    fixture_action_context: ActionContext,
    fixture_bash_action: BashAction,
) -> None:
    fixture_bash_action.set_context(fixture_action_context)
    async with fixture_bash_action as action:
        result = await action.execute(
            fixture_action_context,
            {"command": "pwd"},
        )

    assert result["timed_out"] is False
    assert result["exit_code"] == 0
    assert result["stderr"] == ""
    assert result["stdout"].strip() == str(fixture_action_context.root_path)
    assert isinstance(result["duration_ms"], int)
    assert result["duration_ms"] >= 0
    assert result["signal"] is None
    assert result["exit_code_interpretation"] is None


async def test_bash_action_timeout(
    fixture_action_context: ActionContext,
    fixture_bash_action: BashAction,
) -> None:
    fixture_bash_action.set_context(fixture_action_context)
    async with fixture_bash_action as action:
        result = await action.execute(
            fixture_action_context,
            {
                "command": "sleep 1", "timeout_seconds": 0.01
            },
        )

    assert result["timed_out"] is True
    assert result["exit_code"] is None
    assert result["signal"] == "SIGTERM"
    assert "timed out" in result["stderr"]


async def test_bash_action_parallel_commands(
    fixture_action_context: ActionContext,
    fixture_bash_action: BashAction,
) -> None:
    fixture_bash_action.set_context(fixture_action_context)
    marker_path = fixture_action_context.root_path / "marker.txt"
    if marker_path.exists():
        marker_path.unlink()

    async with fixture_bash_action as action:
        long_command = "echo started > marker.txt; sleep 0.4; rm -f marker.txt"
        long_task = asyncio.create_task(action.execute(fixture_action_context, {"command": long_command}))

        await _wait_for_marker(marker_path)

        short_command = "if [ -f marker.txt ]; then echo busy; else echo idle; fi"
        short_result = await action.execute(fixture_action_context, {"command": short_command})
        long_result = await long_task

    assert short_result["timed_out"] is False
    assert short_result["exit_code"] == 0
    assert short_result["stdout"].strip() == "busy"
    assert long_result["timed_out"] is False
    assert long_result["exit_code"] == 0


async def test_bash_separate_stderr(
    fixture_action_context: ActionContext,
    fixture_bash_action: BashAction,
) -> None:
    """Verify stdout and stderr are captured independently."""
    fixture_bash_action.set_context(fixture_action_context)
    async with fixture_bash_action as action:
        result = await action.execute(
            fixture_action_context,
            {"command": "echo out_msg; echo err_msg >&2"},
        )

    assert result["exit_code"] == 0
    assert "out_msg" in result["stdout"]
    assert "err_msg" in result["stderr"]
    # stderr should NOT leak into stdout
    assert "err_msg" not in result["stdout"]


async def test_bash_exit_code_interpretation(
    fixture_action_context: ActionContext,
    fixture_bash_action: BashAction,
) -> None:
    """Non-zero well-known exit codes get a human-readable interpretation."""
    fixture_bash_action.set_context(fixture_action_context)
    async with fixture_bash_action as action:
        result = await action.execute(
            fixture_action_context,
            {"command": "exit 127"},
        )

    assert result["exit_code"] == 127
    assert result["exit_code_interpretation"] == "Command not found"


async def test_bash_output_truncation(
    fixture_action_context: ActionContext,
    fixture_bash_action: BashAction,
) -> None:
    """Output exceeding 100 KB is truncated with a notice."""
    fixture_bash_action.set_context(fixture_action_context)
    # Generate ~150 KB of output (each line ~101 bytes with padding)
    async with fixture_bash_action as action:
        result = await action.execute(
            fixture_action_context,
            {"command": "python3 -c \"print('A' * 100 + '\\n') * 1500\""},
        )

    # Fallback: generate via seq + printf for reliability
    async with fixture_bash_action as action:
        result = await action.execute(
            fixture_action_context,
            {"command": "for i in $(seq 1 1500); do printf '%0100d\\n' $i; done"},
        )

    assert result["exit_code"] == 0
    assert "[..." in result["stdout"]
    assert "truncated...]" in result["stdout"]


async def test_bash_working_directory(
    fixture_action_context: ActionContext,
    fixture_bash_action: BashAction,
) -> None:
    """Custom working_directory resolves relative to workspace root."""
    fixture_bash_action.set_context(fixture_action_context)
    subdir = fixture_action_context.root_path / "subdir"
    subdir.mkdir()

    async with fixture_bash_action as action:
        result = await action.execute(
            fixture_action_context,
            {"command": "pwd", "working_directory": "subdir"},
        )

    assert result["exit_code"] == 0
    assert result["stdout"].strip() == str(subdir)


async def test_bash_working_directory_escape_rejected(
    fixture_action_context: ActionContext,
    fixture_bash_action: BashAction,
) -> None:
    """working_directory outside the workspace root is rejected."""
    fixture_bash_action.set_context(fixture_action_context)
    async with fixture_bash_action as action:
        with pytest.raises(ValueError, match="within the workspace root"):
            await action.execute(
                fixture_action_context,
                {"command": "pwd", "working_directory": "/tmp"},
            )
