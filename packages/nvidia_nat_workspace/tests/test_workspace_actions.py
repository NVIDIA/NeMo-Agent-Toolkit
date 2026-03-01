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
"""Tests for build_workspace_actions guardrail wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from nat.guardrails.workspace import WorkspaceGuardrail
from nat.guardrails.workspace import WorkspaceGuardrailViolation
from nat.plugins.workspace.workspace_actions import WorkspaceActionsConfig
from nat.plugins.workspace.workspace_actions import build_workspace_actions
from nat.workspace.types import TypeSchema
from nat.workspace.types import WorkspaceActionSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _BlockingGuardrail(WorkspaceGuardrail):
    """Guardrail that blocks every action."""

    name = "always-block"

    async def validate_action(self, action: Any) -> WorkspaceGuardrailViolation:
        return WorkspaceGuardrailViolation(
            guardrail_name=self.name,
            message="Blocked by test guardrail.",
        )


def _make_mock_workspace(actions: list[WorkspaceActionSchema] | None = None, ) -> MagicMock:
    """Return a MagicMock workspace with get_actions and guardrail tracking."""
    workspace = MagicMock()
    workspace.get_actions = AsyncMock(return_value=actions or [])
    workspace.add_workspace_guardrail = MagicMock()
    return workspace


def _make_mock_builder(
    workspace: MagicMock,
    guardrails: list[WorkspaceGuardrail] | None = None,
) -> MagicMock:
    """Return a MagicMock builder that yields the given workspace."""
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=workspace)
    manager.__aexit__ = AsyncMock(return_value=None)

    builder = MagicMock()
    builder.get_workspace_manager = AsyncMock(return_value=manager)
    builder.get_workspace_guardrails = AsyncMock(return_value=guardrails or [])
    return builder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_guardrails_added_to_workspace() -> None:
    """build_workspace_actions calls add_workspace_guardrail for each guardrail."""
    guardrail = _BlockingGuardrail()
    workspace = _make_mock_workspace()
    builder = _make_mock_builder(workspace=workspace, guardrails=[guardrail])

    async with build_workspace_actions(WorkspaceActionsConfig(), builder):
        pass

    workspace.add_workspace_guardrail.assert_called_once_with(guardrail)


async def test_no_guardrails_skips_add() -> None:
    """build_workspace_actions with empty guardrails list never calls add_workspace_guardrail."""
    workspace = _make_mock_workspace()
    builder = _make_mock_builder(workspace=workspace, guardrails=[])

    async with build_workspace_actions(WorkspaceActionsConfig(), builder):
        pass

    workspace.add_workspace_guardrail.assert_not_called()


async def test_multiple_guardrails_all_added() -> None:
    """All guardrails returned by get_workspace_guardrails are added."""

    class _GuardrailA(WorkspaceGuardrail):
        name = "guardrail-a"

    class _GuardrailB(WorkspaceGuardrail):
        name = "guardrail-b"

    g_a, g_b = _GuardrailA(), _GuardrailB()
    workspace = _make_mock_workspace()
    builder = _make_mock_builder(workspace=workspace, guardrails=[g_a, g_b])

    async with build_workspace_actions(WorkspaceActionsConfig(), builder):
        pass

    assert workspace.add_workspace_guardrail.call_count == 2
    calls = {c.args[0] for c in workspace.add_workspace_guardrail.call_args_list}
    assert g_a in calls
    assert g_b in calls


async def test_no_workspace_raises() -> None:
    """build_workspace_actions raises RuntimeError when no workspace is configured."""
    builder = MagicMock()
    builder.get_workspace_manager = AsyncMock(return_value=None)

    with pytest.raises(RuntimeError, match="No workspace configured"):
        async with build_workspace_actions(WorkspaceActionsConfig(), builder):
            pass


async def test_actions_exposed_as_functions() -> None:
    """Functions in the yielded group correspond to workspace actions."""
    actions = [
        WorkspaceActionSchema(
            name="ping",
            description="Ping the workspace",
            parameters=[],
            result=TypeSchema(type="string", description="Response"),
        ),
    ]
    workspace = _make_mock_workspace(actions=actions)
    builder = _make_mock_builder(workspace=workspace)

    async with build_workspace_actions(WorkspaceActionsConfig(), builder) as group:
        functions = await group.get_accessible_functions()

    # Functions are namespaced by the group config name, e.g. "workspace_actions__ping"
    assert any("ping" in key for key in functions)
