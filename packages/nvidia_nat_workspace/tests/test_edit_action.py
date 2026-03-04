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
from nat.workspace_actions.workspace.file.edit_action import EditAction
from nat.workspace_actions.workspace.file.read_action import ReadAction


@pytest.fixture(name="fixture_action_context")
def fixture_action_context(tmp_path: Path) -> ActionContext:
    return ActionContext(session_id="test-session-edit", root_path=tmp_path.resolve())


@pytest.fixture(name="fixture_edit_action")
def fixture_edit_action() -> EditAction:
    return EditAction()


@pytest.fixture(name="fixture_read_action")
def fixture_read_action() -> ReadAction:
    return ReadAction()


async def test_edit_exact_replace(
    fixture_action_context: ActionContext,
    fixture_edit_action: EditAction,
    fixture_read_action: ReadAction,
) -> None:
    """Replace a string in a file and verify replacements=1."""
    file_path = fixture_action_context.root_path / "edit_me.txt"
    file_path.write_text("Hello World\nGoodbye World\n", encoding="utf-8")

    # Read the file first (required for edit validation)
    await fixture_read_action.execute(
        fixture_action_context,
        {"file_path": str(file_path)},
    )

    result = await fixture_edit_action.execute(
        fixture_action_context,
        {
            "file_path": str(file_path),
            "old_string": "Hello World",
            "new_string": "Hi World",
        },
    )

    assert "is_error" not in result
    assert result["replacements"] == 1
    assert result["modified"] is True
    assert file_path.read_text(encoding="utf-8") == "Hi World\nGoodbye World\n"


async def test_edit_create_mode(
    fixture_action_context: ActionContext,
    fixture_edit_action: EditAction,
) -> None:
    """Use old_string='' (create mode) to create a new file."""
    file_path = fixture_action_context.root_path / "created_by_edit.txt"

    result = await fixture_edit_action.execute(
        fixture_action_context,
        {
            "file_path": str(file_path),
            "old_string": "",
            "new_string": "brand new content",
        },
    )

    assert "is_error" not in result
    assert result["replacements"] == 1
    assert result["modified"] is True
    assert file_path.read_text(encoding="utf-8") == "brand new content"


async def test_edit_file_not_found(
    fixture_action_context: ActionContext,
    fixture_edit_action: EditAction,
) -> None:
    """Attempt to edit a nonexistent file and verify is_error."""
    result = await fixture_edit_action.execute(
        fixture_action_context,
        {
            "file_path": str(fixture_action_context.root_path / "nope.txt"),
            "old_string": "foo",
            "new_string": "bar",
        },
    )

    assert result["is_error"] is True
    assert result["replacements"] == 0
    assert result["modified"] is False


async def test_edit_identical_strings(
    fixture_action_context: ActionContext,
    fixture_edit_action: EditAction,
) -> None:
    """Passing identical old_string and new_string should produce an error."""
    file_path = fixture_action_context.root_path / "same.txt"
    file_path.write_text("same content", encoding="utf-8")

    result = await fixture_edit_action.execute(
        fixture_action_context,
        {
            "file_path": str(file_path),
            "old_string": "same content",
            "new_string": "same content",
        },
    )

    assert result["is_error"] is True
    assert "identical" in result["error"].lower()


async def test_edit_replace_all(
    fixture_action_context: ActionContext,
    fixture_edit_action: EditAction,
    fixture_read_action: ReadAction,
) -> None:
    """Use replace_all=True to replace multiple occurrences of a string."""
    file_path = fixture_action_context.root_path / "multi.txt"
    file_path.write_text("foo bar foo baz foo\n", encoding="utf-8")

    # Read the file first
    await fixture_read_action.execute(
        fixture_action_context,
        {"file_path": str(file_path)},
    )

    result = await fixture_edit_action.execute(
        fixture_action_context,
        {
            "file_path": str(file_path),
            "old_string": "foo",
            "new_string": "qux",
            "replace_all": True,
        },
    )

    assert "is_error" not in result
    assert result["replacements"] == 3
    assert result["modified"] is True
    assert file_path.read_text(encoding="utf-8") == "qux bar qux baz qux\n"


async def test_edit_string_not_found(
    fixture_action_context: ActionContext,
    fixture_edit_action: EditAction,
    fixture_read_action: ReadAction,
) -> None:
    """Try to replace text that does not exist in the file and verify is_error."""
    file_path = fixture_action_context.root_path / "no_match.txt"
    file_path.write_text("alpha beta gamma\n", encoding="utf-8")

    # Read the file first
    await fixture_read_action.execute(
        fixture_action_context,
        {"file_path": str(file_path)},
    )

    result = await fixture_edit_action.execute(
        fixture_action_context,
        {
            "file_path": str(file_path),
            "old_string": "delta",
            "new_string": "epsilon",
        },
    )

    assert result["is_error"] is True
    assert "not found" in result["error"].lower() or "string" in result["error"].lower()
    assert result["replacements"] == 0
    assert result["modified"] is False
