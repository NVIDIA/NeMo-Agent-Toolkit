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
from nat.workspace_actions.workspace.file.write_action import WriteAction


@pytest.fixture(name="fixture_action_context")
def fixture_action_context(tmp_path: Path) -> ActionContext:
    return ActionContext(session_id="test-session-write", root_path=tmp_path.resolve())


@pytest.fixture(name="fixture_write_action")
def fixture_write_action() -> WriteAction:
    return WriteAction()


async def test_write_creates_new_file(
    fixture_action_context: ActionContext,
    fixture_write_action: WriteAction,
) -> None:
    """Write a new file and verify created=True."""
    file_path = fixture_action_context.root_path / "new_file.txt"

    result = await fixture_write_action.execute(
        fixture_action_context,
        {"file_path": str(file_path), "content": "hello world"},
    )

    assert "is_error" not in result
    assert result["created"] is True
    assert result["bytes_written"] == len("hello world".encode("utf-8"))
    assert file_path.read_text(encoding="utf-8") == "hello world"


async def test_write_overwrites_existing(
    fixture_action_context: ActionContext,
    fixture_write_action: WriteAction,
) -> None:
    """Write a file twice; the second write should report created=False."""
    file_path = fixture_action_context.root_path / "overwrite.txt"

    result1 = await fixture_write_action.execute(
        fixture_action_context,
        {"file_path": str(file_path), "content": "first"},
    )
    assert result1["created"] is True

    result2 = await fixture_write_action.execute(
        fixture_action_context,
        {"file_path": str(file_path), "content": "second"},
    )
    assert result2["created"] is False
    assert "is_error" not in result2
    assert file_path.read_text(encoding="utf-8") == "second"


async def test_write_auto_creates_dirs(
    fixture_action_context: ActionContext,
    fixture_write_action: WriteAction,
) -> None:
    """Write to a nested path and verify parent directories are created."""
    file_path = fixture_action_context.root_path / "nested" / "path" / "file.txt"

    result = await fixture_write_action.execute(
        fixture_action_context,
        {"file_path": str(file_path), "content": "deep content"},
    )

    assert "is_error" not in result
    assert result["created"] is True
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == "deep content"


async def test_write_bytes_written(
    fixture_action_context: ActionContext,
    fixture_write_action: WriteAction,
) -> None:
    """Verify bytes_written matches the UTF-8 encoded length of the content."""
    # Use a string with multi-byte characters to ensure UTF-8 encoding is used.
    content = "hello \u00e9\u00e8\u00ea \u2603"  # accented chars + snowman
    file_path = fixture_action_context.root_path / "utf8_test.txt"

    result = await fixture_write_action.execute(
        fixture_action_context,
        {"file_path": str(file_path), "content": content},
    )

    assert "is_error" not in result
    expected_bytes = len(content.encode("utf-8"))
    assert result["bytes_written"] == expected_bytes
