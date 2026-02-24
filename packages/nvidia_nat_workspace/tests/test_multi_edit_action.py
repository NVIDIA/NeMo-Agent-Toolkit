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
from nat.workspace_actions.workspace.file.multi_edit_action import MultiEditAction
from nat.workspace_actions.workspace.file.read_action import ReadAction


@pytest.fixture(name="fixture_action_context")
def fixture_action_context(tmp_path: Path) -> ActionContext:
    return ActionContext(session_id="test-session-multi-edit", root_path=tmp_path.resolve())


@pytest.fixture(name="fixture_multi_edit_action")
def fixture_multi_edit_action() -> MultiEditAction:
    return MultiEditAction()


@pytest.fixture(name="fixture_read_action")
def fixture_read_action() -> ReadAction:
    return ReadAction()


async def test_multi_edit_sequential(
    fixture_action_context: ActionContext,
    fixture_multi_edit_action: MultiEditAction,
    fixture_read_action: ReadAction,
) -> None:
    """Apply 2 sequential edits and verify both succeed."""
    file_path = fixture_action_context.root_path / "multi_edit.txt"
    file_path.write_text("aaa bbb ccc\n", encoding="utf-8")

    # Read the file first (required for edit validation)
    await fixture_read_action.execute(
        fixture_action_context,
        {"file_path": str(file_path)},
    )

    result = await fixture_multi_edit_action.execute(
        fixture_action_context,
        {
            "file_path": str(file_path),
            "edits": [
                {"old_string": "aaa", "new_string": "AAA"},
                {"old_string": "bbb", "new_string": "BBB"},
            ],
        },
    )

    assert result["total_edits"] == 2
    assert result["successful_edits"] == 2
    assert result["failed_edits"] == 0
    assert result["total_replacements"] == 2
    assert result["modified"] is True
    assert "is_error" not in result
    assert file_path.read_text(encoding="utf-8") == "AAA BBB ccc\n"


async def test_multi_edit_stops_on_failure(
    fixture_action_context: ActionContext,
    fixture_multi_edit_action: MultiEditAction,
    fixture_read_action: ReadAction,
) -> None:
    """First edit succeeds, second edit fails (string not found), verify counts."""
    file_path = fixture_action_context.root_path / "partial_edit.txt"
    file_path.write_text("apple banana\n", encoding="utf-8")

    # Read the file first
    await fixture_read_action.execute(
        fixture_action_context,
        {"file_path": str(file_path)},
    )

    result = await fixture_multi_edit_action.execute(
        fixture_action_context,
        {
            "file_path": str(file_path),
            "edits": [
                {"old_string": "apple", "new_string": "APPLE"},
                {"old_string": "cherry", "new_string": "CHERRY"},  # does not exist
            ],
        },
    )

    assert result["total_edits"] == 2
    assert result["successful_edits"] == 1
    assert result["failed_edits"] == 1
    assert result["is_error"] is True
    # The first edit should have been applied
    assert file_path.read_text(encoding="utf-8") == "APPLE banana\n"
    # Check that results list reflects success then failure
    assert len(result["results"]) == 2
    assert result["results"][0]["success"] is True
    assert result["results"][1]["success"] is False
