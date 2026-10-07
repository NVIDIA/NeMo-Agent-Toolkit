# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise telemetry through the middleware chain, including early exits."""

import asyncio

import pytest

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStepCategory
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepState
from nat.data_models.span import Span
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.function_middleware import FunctionMiddlewareChain
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.telemetry import trace_middleware
from nat.observability.exporter.span_exporter import SpanExporter


@pytest.fixture
def recorded_steps():
    context = Context.get()
    steps = []
    subscription = context.intermediate_step_manager.subscribe(steps.append)
    initial_span = context.active_span_id
    yield steps
    subscription.unsubscribe()
    assert context.active_span_id == initial_span


def chain(*middleware):
    context = FunctionMiddlewareContext(name="echo",
                                        config=None,
                                        description=None,
                                        input_schema=None,
                                        single_output_schema=None,
                                        stream_output_schema=None)
    return FunctionMiddlewareChain(middleware=middleware, context=context)


def assert_pairs(steps, count):
    starts = [step for step in steps if step.payload.event_state == IntermediateStepState.START]
    ends = [step for step in steps if step.payload.event_state == IntermediateStepState.END]
    assert len(starts) == len(ends) == count
    assert {step.UUID for step in starts} == {step.UUID for step in ends}
    return starts, ends


async def echo(value):
    return value


async def test_nested_middleware_parentage_and_output(recorded_steps):
    call = chain(FunctionMiddleware(), FunctionMiddleware()).build_single(echo)
    assert await call("hello") == "hello"
    starts, ends = assert_pairs(recorded_steps, 2)
    assert starts[1].parent_id == starts[0].UUID
    assert all(step.payload.data.output == "hello" for step in ends)
    assert starts[0].function_ancestry.function_id == starts[1].function_ancestry.function_id


async def test_disabled_middleware_emits_no_steps(recorded_steps):

    class Disabled(FunctionMiddleware):

        @property
        def enabled(self):
            return False

    assert await chain(Disabled()).build_single(echo)("hello") == "hello"
    assert recorded_steps == []


@pytest.mark.parametrize("stage", ["pre", "target", "post"])
async def test_exception_still_pairs_spans(stage, recorded_steps):

    class Failing(FunctionMiddleware):

        async def pre_invoke(self, context):
            if stage == "pre":
                raise ValueError("failed")

        async def post_invoke(self, context):
            if stage == "post":
                raise ValueError("failed")

    async def target(value):
        if stage == "target":
            raise ValueError("failed")
        return value

    with pytest.raises(ValueError, match="failed"):
        await chain(Failing()).build_single(target)("hello")
    _, ends = assert_pairs(recorded_steps, 1)
    assert ends[0].payload.metadata["status"] == "ValueError"


async def test_cancellation_pairs_spans(recorded_steps):
    entered = asyncio.Event()

    async def target(value):
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

    async def target(value):
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
    context = Context.get()
    manager = context.intermediate_step_manager
    initial_span = context.active_span_id
    initial_outstanding = manager.get_outstanding_step_count()
    target_closed = asyncio.Event()
    consumer_started = asyncio.Event()
    held_targets = []

    async def source(value):
        try:
            yield "first"
            await asyncio.Event().wait()
        finally:
            target_closed.set()

    def target(value):
        # Hold the generator so GC/finalizer scheduling cannot mask missing close delegation.
        stream = source(value)
        held_targets.append(stream)
        return stream

    def assert_closed():
        assert target_closed.is_set()
        assert manager.get_outstanding_step_count() == initial_outstanding
        assert context.active_span_id == initial_span
        starts, ends = assert_pairs(recorded_steps, 2)
        assert [step.UUID for step in ends] == [step.UUID for step in reversed(starts)]

    async def consume():
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
    payload = IntermediateStepPayload(event_type=event_type)
    assert payload.event_category == IntermediateStepCategory.GUARDRAIL
    assert payload.event_state == state


async def test_guardrail_steps_reach_span_exporter():

    class Recorder(SpanExporter[Span, Span]):

        def __init__(self):
            super().__init__()
            self.spans = []

        async def export_processed(self, item: Span) -> None:
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
