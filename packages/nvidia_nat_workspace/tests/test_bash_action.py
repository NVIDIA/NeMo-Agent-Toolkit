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
from nat.workspace_actions.workspace.bash_action import BashAction


@pytest.fixture(name="fixture_action_context")
def fixture_action_context(tmp_path: Path) -> ActionContext:
    return ActionContext(session_id="test-session", root_path=tmp_path)


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
        long_task = asyncio.create_task(action.execute(fixture_action_context, {"command": long_command}), )

        await _wait_for_marker(marker_path)

        short_command = "if [ -f marker.txt ]; then echo busy; else echo idle; fi"
        short_result = await action.execute(fixture_action_context, {"command": short_command})
        long_result = await long_task

    assert short_result["timed_out"] is False
    assert short_result["exit_code"] == 0
    assert short_result["stdout"].strip() == "busy"
    assert long_result["timed_out"] is False
    assert long_result["exit_code"] == 0
