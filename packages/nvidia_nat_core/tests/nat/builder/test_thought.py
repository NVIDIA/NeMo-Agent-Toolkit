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

import uuid

import pytest

from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.thought import emit_thought
from nat.builder.thought import emit_thought_chunk
from nat.builder.thought import emit_thought_end
from nat.builder.thought import emit_thought_start
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.invocation_node import InvocationNode

# --------------------------------------------------------------------------- #
# Minimal stubs so the tests do not need the whole NAT code-base
# --------------------------------------------------------------------------- #


class _DummyFunction(InvocationNode):  # what active_function.get() returns

    def __init__(self, name="fn", fid=None, parent_id=None, parent_name=None):
        super().__init__(function_id=fid or str(uuid.uuid4()),
                         function_name=name,
                         parent_id=parent_id,
                         parent_name=parent_name)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(name="ctx_state")
def ctx_state_fixture():
    """Fresh context state for each test."""
    s = ContextState()

    s.active_function.set(_DummyFunction(parent_id="root", parent_name="root"))

    yield s

    assert len(s.active_span_id_stack.get()) == 1, "Active span id stack should be reset after a test"


@pytest.fixture(name="output_steps")
def output_steps_fixture():
    return []


@pytest.fixture(name="ctx")
def ctx_fixture(ctx_state: ContextState, output_steps):
    """Context with intermediate step manager subscribed to output_steps."""
    ctx = Context(ctx_state)

    def on_next(step: IntermediateStep):
        output_steps.append(step)

    ctx.intermediate_step_manager.subscribe(on_next)
    return ctx


# --------------------------------------------------------------------------- #
# Tests for emit_thought()
# --------------------------------------------------------------------------- #


def test_emit_thought_creates_start_and_end_events(ctx: Context, output_steps: list[IntermediateStep]):
    """Test that emit_thought() emits both START and END events with the same UUID."""
    thought_uuid = emit_thought(ctx, "Processing data...")

    # Should emit exactly 2 events (START + END)
    assert len(output_steps) == 2

    start_event = output_steps[0]
    end_event = output_steps[1]

    # Both should share the same UUID
    assert start_event.payload.UUID == thought_uuid
    assert end_event.payload.UUID == thought_uuid

    # Check event types
    assert start_event.payload.event_type == IntermediateStepType.SPAN_START
    assert end_event.payload.event_type == IntermediateStepType.SPAN_END

    # Check thought_text in data
    assert start_event.payload.data.input == "Processing data..."
    assert end_event.payload.data.output == "Processing data..."


def test_emit_thought_custom_name(ctx: Context, output_steps: list[IntermediateStep]):
    """Test that emit_thought() accepts a custom name."""
    emit_thought(ctx, "Validating input", name="validation")

    start_event = output_steps[0]
    assert start_event.payload.name == "validation"


# --------------------------------------------------------------------------- #
# Tests for emit_thought_start()
# --------------------------------------------------------------------------- #


def test_emit_thought_start_creates_start_event(ctx: Context, output_steps: list[IntermediateStep]):
    """Test that emit_thought_start() emits only a START event."""
    thought_uuid = emit_thought_start(ctx, "Starting process...")

    # Should emit exactly 1 event (START)
    assert len(output_steps) == 1

    start_event = output_steps[0]
    assert start_event.payload.UUID == thought_uuid
    assert start_event.payload.event_type == IntermediateStepType.SPAN_START
    assert start_event.payload.data.input == "Starting process..."

    # Balance span stack for fixture teardown
    emit_thought_end(ctx, thought_uuid)


# --------------------------------------------------------------------------- #
# Tests for emit_thought_chunk()
# --------------------------------------------------------------------------- #


def test_emit_thought_chunk_creates_chunk_event(ctx: Context, output_steps: list[IntermediateStep]):
    """Test that emit_thought_chunk() emits a CHUNK event with the same UUID."""
    thought_uuid = emit_thought_start(ctx, "Processing: 0%")
    output_steps.clear()  # Clear the START event

    emit_thought_chunk(ctx, thought_uuid, "Processing: 50%")

    # Should emit exactly 1 event (CHUNK)
    assert len(output_steps) == 1

    chunk_event = output_steps[0]
    assert chunk_event.payload.UUID == thought_uuid
    assert chunk_event.payload.event_type == IntermediateStepType.SPAN_CHUNK
    assert chunk_event.payload.data.chunk == "Processing: 50%"

    emit_thought_end(ctx, thought_uuid)


# --------------------------------------------------------------------------- #
# Tests for emit_thought_end()
# --------------------------------------------------------------------------- #


def test_emit_thought_end_creates_end_event(ctx: Context, output_steps: list[IntermediateStep]):
    """Test that emit_thought_end() emits an END event with the same UUID."""
    thought_uuid = emit_thought_start(ctx, "Starting...")
    output_steps.clear()

    emit_thought_end(ctx, thought_uuid, "Complete")

    # Should emit exactly 1 event (END)
    assert len(output_steps) == 1

    end_event = output_steps[0]
    assert end_event.payload.UUID == thought_uuid
    assert end_event.payload.event_type == IntermediateStepType.SPAN_END
    assert end_event.payload.data.output == "Complete"


def test_emit_thought_end_with_no_text(ctx: Context, output_steps: list[IntermediateStep]):
    """Test that emit_thought_end() can be called without thought_text."""
    thought_uuid = emit_thought_start(ctx, "Starting...")
    output_steps.clear()

    emit_thought_end(ctx, thought_uuid)

    end_event = output_steps[0]
    assert end_event.payload.UUID == thought_uuid
    assert end_event.payload.event_type == IntermediateStepType.SPAN_END
    # data.output should be None when no thought_text provided
    assert end_event.payload.data.output is None


# --------------------------------------------------------------------------- #
# Integration Tests (Full Lifecycle)
# --------------------------------------------------------------------------- #


def test_streaming_thought_lifecycle(ctx: Context, output_steps: list[IntermediateStep]):
    """Test the full lifecycle of a streaming thought: START -> CHUNK -> CHUNK -> END."""
    thought_uuid = emit_thought_start(ctx, "Processing dataset: 0%")
    emit_thought_chunk(ctx, thought_uuid, "Processing dataset: 50%")
    emit_thought_chunk(ctx, thought_uuid, "Processing dataset: 100%")
    emit_thought_end(ctx, thought_uuid, "Dataset processing complete")

    # Should have 4 events total
    assert len(output_steps) == 4

    # Verify event types and order
    assert output_steps[0].payload.event_type == IntermediateStepType.SPAN_START
    assert output_steps[1].payload.event_type == IntermediateStepType.SPAN_CHUNK
    assert output_steps[2].payload.event_type == IntermediateStepType.SPAN_CHUNK
    assert output_steps[3].payload.event_type == IntermediateStepType.SPAN_END

    # All should share the same UUID
    for event in output_steps:
        assert event.payload.UUID == thought_uuid


def test_multiple_concurrent_thoughts(ctx: Context, output_steps: list[IntermediateStep]):
    """Test that multiple thoughts can be emitted with different UUIDs."""
    uuid1 = emit_thought_start(ctx, "Task 1: Starting")
    uuid2 = emit_thought_start(ctx, "Task 2: Starting")

    # Should have 2 START events with different UUIDs
    assert len(output_steps) == 2
    assert uuid1 != uuid2
    assert output_steps[0].payload.UUID == uuid1
    assert output_steps[1].payload.UUID == uuid2

    output_steps.clear()

    emit_thought_chunk(ctx, uuid1, "Task 1: 50%")
    emit_thought_chunk(ctx, uuid2, "Task 2: 25%")

    # Should have 2 CHUNK events with correct UUIDs
    assert len(output_steps) == 2
    assert output_steps[0].payload.UUID == uuid1
    assert output_steps[1].payload.UUID == uuid2

    # Pop in reverse order (LIFO)
    emit_thought_end(ctx, uuid2)
    emit_thought_end(ctx, uuid1)
