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
from nat.workspace_actions.workspace.file.read_action import ReadAction


@pytest.fixture(name="fixture_action_context")
def fixture_action_context(tmp_path: Path) -> ActionContext:
    return ActionContext(session_id="test-session-read", root_path=tmp_path.resolve())


@pytest.fixture(name="fixture_read_action")
def fixture_read_action() -> ReadAction:
    return ReadAction()


async def test_read_normal_file(
    fixture_action_context: ActionContext,
    fixture_read_action: ReadAction,
) -> None:
    """Write a file and read it back, verifying that content includes line numbers."""
    file_path = fixture_action_context.root_path / "hello.txt"
    file_path.write_text("line one\nline two\nline three\n", encoding="utf-8")

    result = await fixture_read_action.execute(
        fixture_action_context,
        {"file_path": str(file_path)},
    )

    assert "is_error" not in result
    assert result["total_lines"] == 4  # trailing newline produces an empty 4th line
    assert result["lines_returned"] > 0
    # Content should contain numbered lines (e.g. "     1|line one")
    content = result["content"]
    assert "1|" in content
    assert "line one" in content
    assert "line two" in content


async def test_read_with_offset_and_limit(
    fixture_action_context: ActionContext,
    fixture_read_action: ReadAction,
) -> None:
    """Read lines 2-3 from a 5-line file using offset and limit."""
    file_path = fixture_action_context.root_path / "five_lines.txt"
    file_path.write_text("aaa\nbbb\nccc\nddd\neee", encoding="utf-8")

    result = await fixture_read_action.execute(
        fixture_action_context,
        {"file_path": str(file_path), "offset": 2, "limit": 2},
    )

    assert "is_error" not in result
    assert result["start_line"] == 2
    assert result["end_line"] == 3
    assert result["lines_returned"] == 2
    content = result["content"]
    assert "bbb" in content
    assert "ccc" in content
    # Line 1 (aaa) should not be present in the output
    assert "aaa" not in content


async def test_read_file_not_found(
    fixture_action_context: ActionContext,
    fixture_read_action: ReadAction,
) -> None:
    """Attempt to read a nonexistent file and verify is_error is True."""
    result = await fixture_read_action.execute(
        fixture_action_context,
        {"file_path": str(fixture_action_context.root_path / "does_not_exist.txt")},
    )

    assert result["is_error"] is True
    assert "not found" in result["error"].lower() or "no such file" in result["error"].lower()


async def test_read_binary_file(
    fixture_action_context: ActionContext,
    fixture_read_action: ReadAction,
) -> None:
    """Attempt to read a binary file (.exe) and verify an error about binary content."""
    file_path = fixture_action_context.root_path / "program.exe"
    file_path.write_bytes(b"\x00\x01\x02\x03")

    result = await fixture_read_action.execute(
        fixture_action_context,
        {"file_path": str(file_path)},
    )

    assert result["is_error"] is True
    assert "binary" in result["error"].lower()


async def test_read_empty_file(
    fixture_action_context: ActionContext,
    fixture_read_action: ReadAction,
) -> None:
    """Read an empty file and verify the content says 'File is empty.'."""
    file_path = fixture_action_context.root_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    result = await fixture_read_action.execute(
        fixture_action_context,
        {"file_path": str(file_path)},
    )

    assert "is_error" not in result
    assert result["content"] == "File is empty."
    assert result["total_lines"] == 0
    assert result["lines_returned"] == 0
