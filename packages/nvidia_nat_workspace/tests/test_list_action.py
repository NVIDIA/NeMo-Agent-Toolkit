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
from nat.workspace_actions.workspace.search.list_action import ListAction


@pytest.fixture(name="fixture_action_context")
def fixture_action_context(tmp_path: Path) -> ActionContext:
    return ActionContext(session_id="test-session-list", root_path=tmp_path.resolve())


@pytest.fixture(name="fixture_list_action")
def fixture_list_action() -> ListAction:
    return ListAction()


async def test_list_directory(
    fixture_action_context: ActionContext,
    fixture_list_action: ListAction,
) -> None:
    """Create several files, list the directory, and verify the tree output has correct file_count."""
    root = fixture_action_context.root_path
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "b.txt").write_text("b", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c", encoding="utf-8")

    result = await fixture_list_action.execute(
        fixture_action_context,
        {"path": str(root)},
    )

    assert "is_error" not in result
    assert result["file_count"] == 3
    # The tree output should mention the file names
    tree = result["tree"]
    assert "a.txt" in tree
    assert "b.txt" in tree
    assert "c.txt" in tree
    assert "sub" in tree


async def test_list_empty_directory(
    fixture_action_context: ActionContext,
    fixture_list_action: ListAction,
) -> None:
    """List an empty directory and verify file_count=0."""
    empty_dir = fixture_action_context.root_path / "empty"
    empty_dir.mkdir()

    result = await fixture_list_action.execute(
        fixture_action_context,
        {"path": str(empty_dir)},
    )

    assert "is_error" not in result
    assert result["file_count"] == 0


async def test_list_with_max_depth(
    fixture_action_context: ActionContext,
    fixture_list_action: ListAction,
) -> None:
    """Create nested directories and verify depth limiting works with max_depth=1."""
    root = fixture_action_context.root_path
    (root / "top.txt").write_text("top", encoding="utf-8")
    level1 = root / "level1"
    level1.mkdir()
    (level1 / "mid.txt").write_text("mid", encoding="utf-8")
    level2 = level1 / "level2"
    level2.mkdir()
    (level2 / "deep.txt").write_text("deep", encoding="utf-8")

    result = await fixture_list_action.execute(
        fixture_action_context,
        {"path": str(root), "max_depth": 1},
    )

    assert "is_error" not in result
    tree = result["tree"]
    # top.txt should be listed (depth 0)
    assert "top.txt" in tree
    # mid.txt is at depth 1, should be included
    assert "mid.txt" in tree
    # deep.txt is at depth 2, should NOT be included with max_depth=1
    assert "deep.txt" not in tree
