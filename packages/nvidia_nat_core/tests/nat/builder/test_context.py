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

from nat.builder.context import Context
from nat.builder.context import ContextState


def test_has_manual_latency_sensitivity_false_by_default():
    """Default stack [2] means no manual decorator is active."""
    state = ContextState.get()
    # Reset to ensure fresh state
    state._latency_sensitivity_stack.set(None)

    ctx = Context.get()
    assert ctx.has_manual_latency_sensitivity is False


def test_has_manual_latency_sensitivity_true_when_pushed():
    """After push_latency_sensitivity, a manual decorator is active."""
    state = ContextState.get()
    state._latency_sensitivity_stack.set(None)

    ctx = Context.get()
    with ctx.push_latency_sensitivity(5):
        assert ctx.has_manual_latency_sensitivity is True


def test_has_manual_latency_sensitivity_false_after_pop():
    """After exiting push scope, manual flag reverts."""
    state = ContextState.get()
    state._latency_sensitivity_stack.set(None)

    ctx = Context.get()
    with ctx.push_latency_sensitivity(5):
        assert ctx.has_manual_latency_sensitivity is True
    assert ctx.has_manual_latency_sensitivity is False


def test_has_manual_latency_sensitivity_nested():
    """Nested pushes maintain manual flag."""
    state = ContextState.get()
    state._latency_sensitivity_stack.set(None)

    ctx = Context.get()
    with ctx.push_latency_sensitivity(3):
        assert ctx.has_manual_latency_sensitivity is True
        with ctx.push_latency_sensitivity(1):
            assert ctx.has_manual_latency_sensitivity is True
        assert ctx.has_manual_latency_sensitivity is True
    assert ctx.has_manual_latency_sensitivity is False
