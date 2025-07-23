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

import asyncio
import contextvars
import functools
import threading
import uuid

import pytest

from aiq.builder.context import AIQContext
from aiq.builder.context import AIQContextState
from aiq.builder.intermediate_step_manager import IntermediateStepManager
from aiq.builder.intermediate_step_manager import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.invocation_node import InvocationNode

# --------------------------------------------------------------------------- #
# Minimal stubs so the tests do not need the whole aiq code-base
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
    """Fresh manager + its stubbed context-state for each test."""
    s = AIQContextState()

    s.active_function.set(_DummyFunction(parent_id="root", parent_name="root"))

    yield s

    assert len(s.active_span_id_stack.get()) == 1, "Active span id stack should be reset after a test"


@pytest.fixture(name="output_steps")
def output_steps_fixture():
    return []


@pytest.fixture(name="ctx")
def ctx_fixture(ctx_state: AIQContextState):
    return AIQContext(ctx_state)


@pytest.fixture(name="mgr")
def mgr_fixture(ctx_state: AIQContextState, output_steps):
    """Fresh manager + its stubbed context-state for each test."""
    mgr = IntermediateStepManager(context_state=ctx_state)

    def on_next(step: IntermediateStep):
        output_steps.append(step)

    mgr.subscribe(on_next)
    return mgr


def _payload(step_id=None, name="step", etype: IntermediateStepType = IntermediateStepType.LLM_START):
    """Helper to create a payload with only the fields the manager uses."""
    return IntermediateStepPayload(
        UUID=step_id or str(uuid.uuid4()),
        name=name,
        event_type=IntermediateStepType(etype),
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_start_pushes_event_and_tracks_open_step(mgr: IntermediateStepManager, output_steps: list[IntermediateStep]):
    pay = _payload()
    mgr.push_intermediate_step(pay)

    # one event captured
    assert len(output_steps) == 1
    # step now in outstanding dict
    assert pay.UUID in mgr._outstanding_start_steps

    mgr.push_intermediate_step(_payload(step_id=pay.UUID, etype=IntermediateStepType.LLM_END))

    assert pay.UUID not in mgr._outstanding_start_steps


def test_chunk_preserves_parent_id(ctx: AIQContext, mgr: IntermediateStepManager):

    start = _payload()
    mgr.push_intermediate_step(start)  # START

    assert ctx.active_span_id == start.UUID

    chunk = _payload(step_id=start.UUID, etype=IntermediateStepType.LLM_NEW_TOKEN)
    mgr.push_intermediate_step(chunk)

    # parent should still be the START id
    assert ctx.active_span_id == start.UUID

    mgr.push_intermediate_step(_payload(step_id=start.UUID, etype=IntermediateStepType.LLM_END))


def test_end_same_context_restores_parent(ctx: AIQContext, mgr: IntermediateStepManager):
    start1 = _payload()
    mgr.push_intermediate_step(start1)

    assert ctx.active_span_id == start1.UUID

    start2 = _payload()
    mgr.push_intermediate_step(start2)

    assert ctx.active_span_id == start2.UUID

    # End the second start
    mgr.push_intermediate_step(_payload(step_id=start2.UUID, etype=IntermediateStepType.LLM_END))

    # Verify that the parent is the first start
    assert ctx.active_span_id == start1.UUID

    # End the first start
    mgr.push_intermediate_step(_payload(step_id=start1.UUID, etype=IntermediateStepType.LLM_END))

    # open-step removed, ContextVar back to parent (None)
    assert start1.UUID not in mgr._outstanding_start_steps


def _end_in_thread(manager, payload):
    """Helper for cross-thread END."""
    manager.push_intermediate_step(payload)


def test_end_other_thread_no_token_error(mgr: IntermediateStepManager):
    pay = _payload()
    mgr.push_intermediate_step(pay)

    end_pay = _payload(step_id=pay.UUID, etype=IntermediateStepType.LLM_END)
    t = threading.Thread(target=_end_in_thread, args=(mgr, end_pay))
    t.start()
    t.join()

    # still cleaned up
    assert pay.UUID not in mgr._outstanding_start_steps


def test_mismatched_chunk_logs_warning(mgr: IntermediateStepManager, caplog: pytest.LogCaptureFixture):
    # CHUNK without START
    chunk = _payload(etype=IntermediateStepType.LLM_NEW_TOKEN)
    mgr.push_intermediate_step(chunk)

    assert "no matching start step" in caplog.text.lower()


async def _nested_fn(mgr: IntermediateStepManager, to_call: list[str]):
    pay = _payload(step_id=to_call[0], name=to_call[0])
    mgr.push_intermediate_step(pay)

    await asyncio.sleep(0)

    if len(to_call) > 1:
        await _nested_fn(mgr, to_call[1:])

    mgr.push_intermediate_step(_payload(step_id=pay.UUID, name=to_call[0], etype=IntermediateStepType.LLM_END))


def _nested_fn_sync(mgr: IntermediateStepManager, to_call: list[str]):
    pay = _payload(step_id=to_call[0], name=to_call[0])
    mgr.push_intermediate_step(pay)

    if len(to_call) > 1:
        _nested_fn_sync(mgr, to_call[1:])

    mgr.push_intermediate_step(_payload(step_id=pay.UUID, name=to_call[0], etype=IntermediateStepType.LLM_END))


async def test_async_nested(mgr: IntermediateStepManager, output_steps: list[IntermediateStep]):

    await _nested_fn(mgr, ["fn1", "fn2", "fn3"])

    expected_output = [
        ("fn1", IntermediateStepType.LLM_START),
        ("fn2", IntermediateStepType.LLM_START),
        ("fn3", IntermediateStepType.LLM_START),
        ("fn3", IntermediateStepType.LLM_END),
        ("fn2", IntermediateStepType.LLM_END),
        ("fn1", IntermediateStepType.LLM_END),
    ]

    for (child, etype), actual in zip(expected_output, output_steps):
        assert child == actual.name
        assert etype == actual.event_type


async def test_async_nested_with_coroutine(mgr: IntermediateStepManager, output_steps: list[IntermediateStep]):

    pay = _payload(step_id="base", name="base")
    mgr.push_intermediate_step(pay)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_nested_fn(mgr, ["a1", "a2", "a3"]))
        tg.create_task(_nested_fn(mgr, ["b1"]))
        tg.create_task(_nested_fn(mgr, ["c1", "c2"]))

    mgr.push_intermediate_step(_payload(step_id=pay.UUID, name="base", etype=IntermediateStepType.LLM_END))

    expected_ancestry = [
        ("a1", "base"),
        ("a2", "a1"),
        ("a3", "a2"),
        ("b1", "base"),
        ("c1", "base"),
        ("c2", "c1"),
    ]

    for actual in output_steps:
        for child, parent in expected_ancestry:
            if actual.name == child:
                assert parent == actual.parent_id
                break


async def test_async_with_task_end(mgr: IntermediateStepManager, output_steps: list[IntermediateStep]):

    async def _main():

        pay = _payload(step_id="main", name="main")
        mgr.push_intermediate_step(pay)

        await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                contextvars.copy_context().run,
                _nested_fn_sync,
                mgr,
                ["fn1_sync"],
            ),
        )
        await _nested_fn(mgr, ["fn1", "fn2"])

        async def _end_event():
            mgr.push_intermediate_step(_payload(step_id=pay.UUID, name="main", etype=IntermediateStepType.LLM_END))

        await asyncio.shield(asyncio.create_task(_end_event()))

        await _nested_fn(mgr, ["fn3"])

    pay = _payload(step_id="base", name="base")
    mgr.push_intermediate_step(pay)

    await _main()

    mgr.push_intermediate_step(_payload(step_id=pay.UUID, name="base", etype=IntermediateStepType.LLM_END))

    expected_output = [
        ("base", None, IntermediateStepType.LLM_START),
        ("main", "base", IntermediateStepType.LLM_START),
        ("fn1_sync", "main", IntermediateStepType.LLM_START),
        ("fn1_sync", "main", IntermediateStepType.LLM_END),
        ("fn1", "main", IntermediateStepType.LLM_START),
        ("fn2", "fn1", IntermediateStepType.LLM_START),
        ("fn2", "fn1", IntermediateStepType.LLM_END),
        ("fn1", "main", IntermediateStepType.LLM_END),
        ("main", "base", IntermediateStepType.LLM_END),
        ("fn3", "base", IntermediateStepType.LLM_START),
        ("fn3", "base", IntermediateStepType.LLM_END),
        ("base", "root", IntermediateStepType.LLM_END),
    ]

    for (child, parent, etype), actual in zip(expected_output, output_steps):
        assert child == actual.name
        assert parent is None or parent == actual.parent_id
        assert etype == actual.event_type
