# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


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
