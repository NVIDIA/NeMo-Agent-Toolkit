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

from pathlib import Path

import pytest

from nat.data_models.workspace import ActionContext
from nat.workspace_actions.workspace.search.glob_action import GlobAction


@pytest.fixture(name="fixture_action_context")
def fixture_action_context(tmp_path: Path) -> ActionContext:
    return ActionContext(session_id="test-session-glob", root_path=tmp_path.resolve())


@pytest.fixture(name="fixture_glob_action")
def fixture_glob_action() -> GlobAction:
    return GlobAction()


async def test_glob_matches_pattern(
    fixture_action_context: ActionContext,
    fixture_glob_action: GlobAction,
) -> None:
    """Create .py and .txt files, glob for *.py, verify only .py files found."""
    root = fixture_action_context.root_path
    (root / "script.py").write_text("print('hello')", encoding="utf-8")
    (root / "notes.txt").write_text("some notes", encoding="utf-8")
    (root / "lib.py").write_text("import os", encoding="utf-8")

    result = await fixture_glob_action.execute(
        fixture_action_context,
        {"glob_pattern": "*.py"},
    )

    assert "is_error" not in result
    assert result["count"] == 2
    file_names = [Path(f).name for f in result["files"]]
    assert "script.py" in file_names
    assert "lib.py" in file_names
    assert "notes.txt" not in file_names


async def test_glob_no_matches(
    fixture_action_context: ActionContext,
    fixture_glob_action: GlobAction,
) -> None:
    """Glob for *.rs in a directory containing only .py files."""
    root = fixture_action_context.root_path
    (root / "main.py").write_text("pass", encoding="utf-8")

    result = await fixture_glob_action.execute(
        fixture_action_context,
        {"glob_pattern": "*.rs"},
    )

    assert "is_error" not in result
    assert result["count"] == 0
    assert result["files"] == []


async def test_glob_not_a_directory(
    fixture_action_context: ActionContext,
    fixture_glob_action: GlobAction,
) -> None:
    """Provide a file path as target_directory and verify error."""
    root = fixture_action_context.root_path
    file_path = root / "afile.txt"
    file_path.write_text("content", encoding="utf-8")

    result = await fixture_glob_action.execute(
        fixture_action_context,
        {"glob_pattern": "*.py", "target_directory": str(file_path)},
    )

    assert result["is_error"] is True
    assert "not a directory" in result["error"].lower()
