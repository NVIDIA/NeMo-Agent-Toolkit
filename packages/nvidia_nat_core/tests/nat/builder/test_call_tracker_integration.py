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
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.llm.prediction_context import get_call_tracker


def test_llm_start_increments_call_tracker():
    """Test that pushing an LLM_START step increments the call tracker."""
    ctx = Context.get()
    step_manager = ctx.intermediate_step_manager

    with ctx.push_active_function("test_agent", input_data=None):
        active_fn = ctx.active_function
        tracker = get_call_tracker()

        # Initially no count for this function
        assert tracker.counts.get(active_fn.function_id, 0) == 0

        # Push LLM_START
        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID="llm-call-1",
                event_type=IntermediateStepType.LLM_START,
                name="test-model",
            ))

        # Call tracker should be incremented
        assert tracker.counts.get(active_fn.function_id) == 1

        # Push another LLM_START
        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID="llm-call-2",
                event_type=IntermediateStepType.LLM_START,
                name="test-model",
            ))

        # Should be 2 now
        assert tracker.counts.get(active_fn.function_id) == 2


def test_non_llm_start_does_not_increment_tracker():
    """Test that non-LLM_START events don't increment the tracker."""
    ctx = Context.get()
    step_manager = ctx.intermediate_step_manager

    with ctx.push_active_function("test_agent_2", input_data=None):
        active_fn = ctx.active_function
        tracker = get_call_tracker()

        initial_count = tracker.counts.get(active_fn.function_id, 0)

        # Push TOOL_START (should not increment)
        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID="tool-call-1",
                event_type=IntermediateStepType.TOOL_START,
                name="test-tool",
            ))

        # Count should be unchanged
        assert tracker.counts.get(active_fn.function_id, 0) == initial_count
