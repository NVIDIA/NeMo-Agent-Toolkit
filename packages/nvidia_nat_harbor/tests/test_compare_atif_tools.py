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
"""Tests for ATIF tool sequence smoke comparison."""

from __future__ import annotations

import json
from pathlib import Path

from nat_harbor.smoke.compare_atif_tools import MATCH_POORER
from nat_harbor.smoke.compare_atif_tools import MATCH_RICHER
from nat_harbor.smoke.compare_atif_tools import MATCH_SAME
from nat_harbor.smoke.compare_atif_tools import MISMATCH
from nat_harbor.smoke.compare_atif_tools import classify_tool_sequences
from nat_harbor.smoke.compare_atif_tools import compare_atif_tool_sequences
from nat_harbor.smoke.compare_atif_tools import extract_tool_sequence


def _trajectory(*steps):
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "test-session",
        "agent": {
            "name": "test-agent"
        },
        "steps": list(steps),
    }


def _tool_step(*tool_names: str):
    return {
        "source": "agent",
        "tool_calls": [{
            "function_name": tool_name,
            "arguments": {},
        } for tool_name in tool_names],
    }


def _write_trajectory(path: Path, *tool_groups: tuple[str, ...]) -> Path:
    steps = [_tool_step(*tool_names) for tool_names in tool_groups]
    path.write_text(json.dumps(_trajectory(*steps)), encoding="utf-8")
    return path


def test_extract_tool_sequence_preserves_step_and_call_order() -> None:
    trajectory = _trajectory(
        _tool_step("grep", "read"),
        {
            "source": "system", "message": "no tools"
        },
        _tool_step("edit"),
    )

    assert extract_tool_sequence(trajectory) == ["grep", "read", "edit"]


def test_classify_tool_sequences() -> None:
    assert classify_tool_sequences(["task", "read"], ["task", "read"]) == MATCH_SAME
    assert classify_tool_sequences(["task", "read"], ["grep", "task", "read"]) == MATCH_RICHER
    assert classify_tool_sequences(["grep", "task", "read"], ["task", "read"]) == MATCH_POORER
    assert classify_tool_sequences(["task", "read"], ["read", "task"]) == MISMATCH


def test_compare_atif_tool_sequences_reads_files(tmp_path: Path) -> None:
    native_path = _write_trajectory(tmp_path / "native.json", ("task", ), ("read", ))
    candidate_path = _write_trajectory(tmp_path / "candidate.json", ("grep", "task"), ("read", ))

    comparison = compare_atif_tool_sequences(native_path, candidate_path)

    assert comparison.classification == MATCH_RICHER
    assert comparison.native_tools == ["task", "read"]
    assert comparison.candidate_tools == ["grep", "task", "read"]
    assert comparison.native_counts == {
        "task": 1,
        "read": 1,
    }
