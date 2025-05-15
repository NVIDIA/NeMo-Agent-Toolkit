# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextvars
import threading
import uuid

import pytest

from aiq.builder.context import AIQContext
from aiq.builder.context import AIQContextState
from aiq.builder.intermediate_step_manager import IntermediateStepManager
from aiq.builder.intermediate_step_manager import IntermediateStepPayload
from aiq.builder.intermediate_step_manager import IntermediateStepState

# --------------------------------------------------------------------------- #
# Minimal stubs so the tests do not need the whole aiq code-base
# --------------------------------------------------------------------------- #


class _DummyStream(list):
    """Bare-bones Observable / Subject replacement."""

    def on_next(self, value):  # reactive push
        self.append(value)

    # simple subscribe: just call back synchronously
    def subscribe(self, on_next, on_error=None, on_complete=None):  # pylint: disable=unused-argument
        for v in self:
            on_next(v)
        return lambda: None  # fake Subscription


class _DummyFunction:  # what active_function.get() returns

    def __init__(self, name="fn", fid=None, parent_name=None):
        self.function_name = name
        self.function_id = fid or str(uuid.uuid4())
        self.parent_name = parent_name


class _DummyContextState:
    """Only what IntermediateStepManager touches."""

    def __init__(self):
        self.active_function = contextvars.ContextVar("active_function")
        self.active_function.set(_DummyFunction())

        self.event_stream = contextvars.ContextVar("event_stream")
        self.event_stream.set(_DummyStream())

        self.active_span_id_stack = contextvars.ContextVar("active_span_id_stack")
        self.active_span_id_stack.set(["root"])


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(name="ctx_state")
def ctx_state_fixture():
    """Fresh manager + its stubbed context-state for each test."""
    s = AIQContextState()

    s.active_function.set(_DummyFunction())
    s.event_stream.set(_DummyStream())

    return s


@pytest.fixture(name="ctx")
def ctx_fixture(ctx_state: AIQContextState):
    return AIQContext(ctx_state)


@pytest.fixture(name="mgr")
def mgr_fixture(ctx_state: AIQContextState):
    """Fresh manager + its stubbed context-state for each test."""
    return IntermediateStepManager(context_state=ctx_state)


def _payload(step_id=None, state=IntermediateStepState.START, name="step", etype="LLM_START"):
    """Helper to create a payload with only the fields the manager uses."""
    return IntermediateStepPayload(
        UUID=step_id or str(uuid.uuid4()),
        name=name,
        event_state=state,
        event_type=etype,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_start_pushes_event_and_tracks_open_step(ctx_state: AIQContextState, mgr: IntermediateStepManager):
    pay = _payload()
    mgr.push_intermediate_step(pay)

    # one event captured
    stream = ctx_state.event_stream.get()
    assert len(stream) == 1
    # step now in outstanding dict
    assert pay.UUID in mgr._outstanding_start_steps


def test_chunk_preserves_parent_id(ctx: AIQContext, mgr: IntermediateStepManager):

    start = _payload()
    mgr.push_intermediate_step(start)  # START

    assert ctx.active_span_id == start.UUID

    chunk = _payload(step_id=start.UUID, state=IntermediateStepState.CHUNK)
    mgr.push_intermediate_step(chunk)

    # parent should still be the START id
    assert ctx.active_span_id == start.UUID


def test_end_same_context_restores_parent(ctx: AIQContext, mgr: IntermediateStepManager):
    start1 = _payload()
    mgr.push_intermediate_step(start1)

    assert ctx.active_span_id == start1.UUID

    start2 = _payload()
    mgr.push_intermediate_step(start2)

    assert ctx.active_span_id == start2.UUID

    # End the second start
    mgr.push_intermediate_step(_payload(step_id=start2.UUID, state=IntermediateStepState.END, etype="LLM_END"))

    # Verify that the parent is the first start
    assert ctx.active_span_id == start1.UUID

    # End the first start
    mgr.push_intermediate_step(_payload(step_id=start1.UUID, state=IntermediateStepState.END, etype="LLM_END"))

    # open-step removed, ContextVar back to parent (None)
    assert start1.UUID not in mgr._outstanding_start_steps


def _end_in_thread(manager, payload):
    """Helper for cross-thread END."""
    manager.push_intermediate_step(payload)


def test_end_other_thread_no_token_error(mgr: IntermediateStepManager):
    pay = _payload()
    mgr.push_intermediate_step(pay)

    end_pay = _payload(step_id=pay.UUID, state=IntermediateStepState.END, etype="LLM_END")
    t = threading.Thread(target=_end_in_thread, args=(mgr, end_pay))
    t.start()
    t.join()

    # still cleaned up
    assert pay.UUID not in mgr._outstanding_start_steps


def test_mismatched_chunk_logs_warning(mgr: IntermediateStepManager, caplog: pytest.LogCaptureFixture):
    # CHUNK without START
    chunk = _payload(state=IntermediateStepState.CHUNK, etype="LLM_NEW_TOKEN")
    mgr.push_intermediate_step(chunk)

    assert "no matching start step" in caplog.text.lower()
