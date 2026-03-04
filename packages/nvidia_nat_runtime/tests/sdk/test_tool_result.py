# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for nat.sdk.tool.result — ToolResult."""

from nat.data_models.workspace import ActionResult
from nat.data_models.workspace import ActionStatus
from nat.sdk.tool.result import ToolResult


class TestToolResult:

    def test_success(self) -> None:
        r = ToolResult(output="hello")
        assert r.output == "hello"
        assert r.error is None
        assert not r.is_error
        assert r.metadata == {}

    def test_error(self) -> None:
        r = ToolResult(error="something broke")
        assert r.is_error
        assert r.output is None

    def test_with_metadata(self) -> None:
        r = ToolResult(output="ok", metadata={"duration": 1.5})
        assert r.metadata["duration"] == 1.5

    def test_str_success(self) -> None:
        r = ToolResult(output=42)
        assert str(r) == "42"

    def test_str_error(self) -> None:
        r = ToolResult(error="fail")
        assert str(r) == "Error: fail"

    def test_str_none_output(self) -> None:
        r = ToolResult()
        assert str(r) == ""

    def test_is_error_false_when_no_error(self) -> None:
        r = ToolResult(output="data")
        assert r.is_error is False

    def test_is_error_true_when_error_set(self) -> None:
        r = ToolResult(error="e")
        assert r.is_error is True

    def test_both_output_and_error(self) -> None:
        """Edge case: if both are set, is_error is True."""
        r = ToolResult(output="partial", error="warning")
        assert r.is_error
        assert r.output == "partial"


class TestFromActionResult:

    def test_success(self) -> None:
        r = ActionResult(status=ActionStatus.SUCCESS, output="hello")
        tr = ToolResult.from_action_result(r)
        assert tr.output == "hello"
        assert not tr.is_error

    def test_failure(self) -> None:
        r = ActionResult(status=ActionStatus.FAILURE, error_message="fail")
        tr = ToolResult.from_action_result(r)
        assert tr.is_error
        assert "fail" in tr.error

    def test_blocked(self) -> None:
        r = ActionResult(status=ActionStatus.BLOCKED_BY_GUARDRAIL, error_message="blocked")
        tr = ToolResult.from_action_result(r)
        assert tr.is_error
        assert "blocked" in tr.error
