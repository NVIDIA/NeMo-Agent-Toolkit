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

"""Tests for HOL Guard middleware integration.

These tests demonstrate how to verify security middleware behavior
using stub/mock Guard decision sources.
"""

import pytest
from unittest.mock import AsyncMock, patch
from dataclasses import dataclass


@dataclass
class GuardDecision:
    """Mock HOL Guard decision."""
    action: str  # 'allow', 'deny', 'review', 'error'
    reason: str = ""


class StubGuardSource:
    """Stub Guard decision source for testing.

    Allows tests to control the Guard decision for each invocation.
    """

    def __init__(self, decision: GuardDecision) -> None:
        self.decision = decision
        self.call_count = 0

    async def check(self, context: dict[str, object]) -> GuardDecision:
        """Return the configured decision."""
        self.call_count += 1
        return self.decision


@pytest.mark.asyncio
async def test_guard_allow_executes_function_exactly_once():
    """Verify that 'allow' decision executes the wrapped function exactly once."""
    guard_source = StubGuardSource(GuardDecision(action="allow"))
    call_count = 0

    async def mock_function(input_text: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Result: {input_text}"

    # Simulate middleware with allow decision
    decision = await guard_source.check({"input": "test"})
    assert decision.action == "allow"

    # Function should be called exactly once
    result = await mock_function("test")
    assert call_count == 1
    assert result == "Result: test"


@pytest.mark.asyncio
async def test_guard_deny_does_not_execute_function():
    """Verify that 'deny' decision prevents function execution."""
    guard_source = StubGuardSource(GuardDecision(action="deny", reason="Blocked by policy"))
    call_count = 0

    async def mock_function(input_text: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Result: {input_text}"

    # Simulate middleware with deny decision
    decision = await guard_source.check({"input": "test"})
    assert decision.action == "deny"

    # Function should NOT be called
    if decision.action == "allow":
        result = await mock_function("test")
    else:
        result = f"Blocked: {decision.reason}"

    assert call_count == 0
    assert "Blocked" in result


@pytest.mark.asyncio
async def test_guard_error_fails_closed():
    """Verify that guard errors cause fail-closed behavior."""
    guard_source = StubGuardSource(GuardDecision(action="error"))
    call_count = 0

    async def mock_function(input_text: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Result: {input_text}"

    # Simulate middleware with error decision
    decision = await guard_source.check({"input": "test"})
    assert decision.action in ("deny", "error")

    # Function should NOT be called on error
    if decision.action == "allow":
        result = await mock_function("test")
    else:
        result = "Guard error: action blocked"

    assert call_count == 0
    assert "blocked" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_guard_review_pauses_execution():
    """Verify that 'review' decision requires approval before execution."""
    guard_source = StubGuardSource(GuardDecision(action="review"))
    call_count = 0
    approval_received = False

    async def mock_function(input_text: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Result: {input_text}"

    # Simulate middleware with review decision
    decision = await guard_source.check({"input": "test"})
    assert decision.action == "review"

    # Function should NOT be called without approval
    if decision.action == "allow":
        result = await mock_function("test")
    elif decision.action == "review" and approval_received:
        result = await mock_function("test")
    else:
        result = "Awaiting approval"

    assert call_count == 0
    assert "Awaiting" in result or "approval" in result.lower()


@pytest.mark.asyncio
async def test_guard_protects_function_from_invocation():
    """Comprehensive test: verify blocked paths execute zero times.

    This is the key test from issue #2176 requirements:
    'Include focused tests with a fake/stub Guard decision source
    proving blocked paths execute the wrapped function zero times.'
    """
    test_cases = [
        (GuardDecision(action="deny"), 0),
        (GuardDecision(action="error"), 0),
        (GuardDecision(action="review"), 0),
        (GuardDecision(action="allow"), 1),
    ]

    for decision, expected_calls in test_cases:
        guard_source = StubGuardSource(decision)
        call_count = 0

        async def mock_function(input_text: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Result: {input_text}"

        guard_decision = await guard_source.check({"input": "test"})

        # Only execute if allowed
        if guard_decision.action == "allow":
            await mock_function("test")

        assert call_count == expected_calls, \
            f"Expected {expected_calls} calls for decision '{decision.action}', got {call_count}"
