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
"""Exercise telemetry through the middleware chain, including early exits."""

import asyncio
from collections.abc import Iterator

import pytest

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepCategory
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepState
from nat.data_models.span import Span
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.function_middleware import FunctionMiddlewareChain
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.telemetry import trace_middleware
from nat.observability.exporter.span_exporter import SpanExporter


@pytest.fixture(name="recorded_steps")
def recorded_steps_fixture() -> Iterator[list[IntermediateStep]]:
    """Record intermediate steps and verify that the caller's span is restored."""
    context = Context.get()
    steps = []
    subscription = context.intermediate_step_manager.subscribe(steps.append)
    initial_span = context.active_span_id
    yield steps
    subscription.unsubscribe()
    assert context.active_span_id == initial_span


def chain(*middleware):
    """Build a middleware chain around a minimal echo-function context."""
    context = FunctionMiddlewareContext(name="echo",
                                        config=None,
                                        description=None,
                                        input_schema=None,
                                        single_output_schema=None,
                                        stream_output_schema=None)
    return FunctionMiddlewareChain(middleware=middleware, context=context)


def assert_pairs(steps, count):
    """Require matching start/end UUIDs and return both event lists."""
    starts = [step for step in steps if step.payload.event_state == IntermediateStepState.START]
    ends = [step for step in steps if step.payload.event_state == IntermediateStepState.END]
    assert len(starts) == len(ends) == count
    assert {step.UUID for step in starts} == {step.UUID for step in ends}
    return starts, ends


async def echo(value):
    """Return the input unchanged to isolate middleware behavior."""
    return value


async def test_nested_middleware_parentage_and_output(recorded_steps):
    """Nested middleware spans retain parentage, ancestry and the delegated output."""
    call = chain(FunctionMiddleware(), FunctionMiddleware()).build_single(echo)
    assert await call("hello") == "hello"
    starts, ends = assert_pairs(recorded_steps, 2)
    assert starts[1].parent_id == starts[0].UUID
    assert all(step.payload.data.output == "hello" for step in ends)
    assert starts[0].function_ancestry.function_id == starts[1].function_ancestry.function_id


async def test_disabled_middleware_emits_no_steps(recorded_steps):
    """Disabled middleware delegates successfully without recording spans."""

    class Disabled(FunctionMiddleware):

        @property
        def enabled(self):
            """Disable this test middleware."""
            return False

    assert await chain(Disabled()).build_single(echo)("hello") == "hello"
    assert recorded_steps == []


@pytest.mark.parametrize("stage", ["pre", "target", "post"])
async def test_exception_still_pairs_spans(stage, recorded_steps):
    """Failures in hooks or the target still close the enclosing middleware span."""

    class Failing(FunctionMiddleware):

        async def pre_invoke(self, context):
            """Raise only when the parametrized failure occurs before invocation."""
            if stage == "pre":
                raise ValueError("failed")

        async def post_invoke(self, context):
            """Raise only when the parametrized failure occurs after invocation."""
            if stage == "post":
                raise ValueError("failed")

    async def target(value):
        """Return the input unless the target is the selected failure point."""
        if stage == "target":
            raise ValueError("failed")
        return value

    with pytest.raises(ValueError, match="failed"):
        await chain(Failing()).build_single(target)("hello")
    _, ends = assert_pairs(recorded_steps, 1)
    assert ends[0].payload.metadata["status"] == "ValueError"


async def test_cancellation_pairs_spans(recorded_steps):
    """Cancelling an active target closes its middleware span with cancellation status."""
    entered = asyncio.Event()

    async def target(value):
        """Signal entry and wait until the test cancels this invocation."""
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(chain(FunctionMiddleware()).build_single(target)("hello"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    _, ends = assert_pairs(recorded_steps, 1)
    assert ends[0].payload.metadata["status"] == "CancelledError"


@pytest.mark.parametrize("close_early", [False, True])
async def test_stream_completion_and_close_pair_nested_spans(close_early, recorded_steps):
    """Exhaustion and explicit stream closure both pair nested middleware events."""

    async def target(value):
        """Yield two chunks to exercise exhaustion and early closure."""
        yield "first"
        yield "last"

    stream = chain(FunctionMiddleware(), FunctionMiddleware()).build_stream(target)("hello")
    if close_early:
        assert await anext(stream) == "first"
        await stream.aclose()
    else:
        assert [chunk async for chunk in stream] == ["first", "last"]
    assert_pairs(recorded_steps, 2)


@pytest.mark.parametrize("cancel_consumer", [False, True])
async def test_nested_stream_close_finishes_retained_target_immediately(cancel_consumer, recorded_steps):
    """Consumer exit closes retained generators immediately and restores trace state."""
    context = Context.get()
    manager = context.intermediate_step_manager
    initial_span = context.active_span_id
    initial_outstanding = manager.get_outstanding_step_count()
    target_closed = asyncio.Event()
    consumer_started = asyncio.Event()
    held_targets = []

    async def source(value):
        """Yield once, then record closure even when the waiting consumer is cancelled."""
        try:
            yield "first"
            await asyncio.Event().wait()
        finally:
            target_closed.set()

    def target(value):
        # Hold the generator so GC/finalizer scheduling cannot mask missing close delegation.
        """Retain the source iterator so garbage collection cannot hide missed closure."""
        stream = source(value)
        held_targets.append(stream)
        return stream

    def assert_closed():
        """Verify target closure, span restoration and reverse-order paired end events."""
        assert target_closed.is_set()
        assert manager.get_outstanding_step_count() == initial_outstanding
        assert context.active_span_id == initial_span
        starts, ends = assert_pairs(recorded_steps, 2)
        assert [step.UUID for step in ends] == [step.UUID for step in reversed(starts)]

    async def consume():
        """Hold the stream open until cancellation and close it during cleanup."""
        stream = chain(FunctionMiddleware(), FunctionMiddleware()).build_stream(target)("hello")
        try:
            assert await anext(stream) == "first"
            consumer_started.set()
            await asyncio.Event().wait()
        finally:
            await stream.aclose()
            assert_closed()

    try:
        if cancel_consumer:
            task = asyncio.create_task(consume())
            await consumer_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            stream = chain(FunctionMiddleware(), FunctionMiddleware()).build_stream(target)("hello")
            assert await anext(stream) == "first"
            await stream.aclose()
            assert_closed()
    finally:
        for stream in held_targets:
            await stream.aclose()


@pytest.mark.parametrize("event_type,state", [("GUARDRAIL_START", IntermediateStepState.START),
                                              ("GUARDRAIL_END", IntermediateStepState.END)])
def test_guardrail_events_are_classified(event_type, state):
    """Guardrail events expose their category and matching start/end state."""
    payload = IntermediateStepPayload(event_type=event_type)
    assert payload.event_category == IntermediateStepCategory.GUARDRAIL
    assert payload.event_state == state


async def test_guardrail_steps_reach_span_exporter():
    """Guardrail events export as spans with input, output and valid timestamps."""

    class Recorder(SpanExporter[Span, Span]):

        def __init__(self):
            """Initialize an exporter that records processed spans in memory."""
            super().__init__()
            self.spans = []

        async def export_processed(self, item: Span) -> None:
            """Retain the processed span for assertions."""
            self.spans.append(item)

    exporter = Recorder()
    async with exporter.start():
        with trace_middleware("guardrails.input", input_data="hello", function_name="echo", guardrail=True) as trace:
            trace.output = "safe"
        await exporter.wait_for_tasks()
    assert len(exporter.spans) == 1
    span = exporter.spans[0]
    assert span.attributes["nat.span.kind"] == "GUARDRAIL"
    assert span.attributes["input.value"] == "hello"
    assert span.attributes["output.value"] == "safe"
    assert span.end_time >= span.start_time
