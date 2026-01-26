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


def test_function_path_stack_default_empty():
    """Test that function_path_stack starts empty."""
    state = ContextState.get()
    # Reset to test fresh state
    state._function_path_stack.set(None)

    path = state.function_path_stack.get()
    assert path == []


def test_function_path_stack_can_be_set():
    """Test that function_path_stack can be set and retrieved."""
    state = ContextState.get()
    state.function_path_stack.set(["workflow", "agent"])

    path = state.function_path_stack.get()
    assert path == ["workflow", "agent"]


def test_push_active_function_updates_path_stack():
    """Test that push_active_function pushes/pops from path stack."""
    ctx = Context.get()
    state = ctx._context_state

    # Reset path stack
    state._function_path_stack.set(None)

    # Initially empty
    assert state.function_path_stack.get() == []

    with ctx.push_active_function("my_workflow", input_data=None):
        assert state.function_path_stack.get() == ["my_workflow"]

        with ctx.push_active_function("react_agent", input_data=None):
            assert state.function_path_stack.get() == ["my_workflow", "react_agent"]

            with ctx.push_active_function("tool_call", input_data=None):
                assert state.function_path_stack.get() == ["my_workflow", "react_agent", "tool_call"]

            # After tool_call exits
            assert state.function_path_stack.get() == ["my_workflow", "react_agent"]

        # After react_agent exits
        assert state.function_path_stack.get() == ["my_workflow"]

    # After workflow exits
    assert state.function_path_stack.get() == []


def test_context_function_path_property():
    """Test that Context.function_path returns a copy of the path stack."""
    ctx = Context.get()
    state = ctx._context_state

    # Reset path stack
    state._function_path_stack.set(None)

    with ctx.push_active_function("workflow", input_data=None):
        with ctx.push_active_function("agent", input_data=None):
            path = ctx.function_path
            assert path == ["workflow", "agent"]

            # Verify it's a copy (modifications don't affect original)
            path.append("modified")
            assert ctx.function_path == ["workflow", "agent"]
