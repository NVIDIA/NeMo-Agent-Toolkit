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
"""A function that raises is exported with an ERROR status, even when the run around it succeeds."""

import asyncio
from datetime import datetime

import pytest

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.span import Span
from nat.data_models.span import SpanStatusCode
from nat.observability.exporter.span_exporter import SpanExporter
from nat.observability.exporter_manager import ExporterManager
from nat.runtime.runner import Runner

EXPORTED: list[Span] = []


class CollectingSpanExporter(SpanExporter[Span, Span]):

    async def export_processed(self, item: Span) -> None:
        EXPORTED.append(item)


class FlakyToolConfig(FunctionBaseConfig, name="span_status_flaky_tool"):
    pass


class RecoveringWorkflowConfig(FunctionBaseConfig, name="span_status_recovering_workflow"):
    pass


@pytest.fixture(name="register_workflows", scope="module", autouse=True)
def register_workflows_fixture():

    @register_function(config_type=FlakyToolConfig)
    async def _tool(config, builder: Builder):

        async def _inner(message: str) -> str:
            raise TimeoutError("tool backend timed out")

        yield _inner

    @register_function(config_type=RecoveringWorkflowConfig)
    async def _workflow(config, builder: Builder):
        tool = await builder.get_function("tool")

        async def _inner(message: str) -> str:
            # like an agent that gets a tool error back and still answers
            try:
                return await tool.ainvoke(message, to_type=str)
            except TimeoutError:
                return "answered without the tool"

        yield _inner


@pytest.fixture(name="clear_exported", autouse=True)
def clear_exported_fixture():
    EXPORTED.clear()


def _status(name: str) -> list[SpanStatusCode]:
    return [span.status.code for span in EXPORTED if span.name == name]


async def test_a_tool_that_raises_is_marked_as_error_inside_a_successful_run():
    async with WorkflowBuilder() as builder:
        await builder.add_function("tool", FlakyToolConfig())
        entry_fn = await builder.add_function("workflow", RecoveringWorkflowConfig())
        exporter_manager = ExporterManager()
        exporter_manager.add_exporter("collect", CollectingSpanExporter())
        async with Runner(input_message="hi",
                          entry_fn=entry_fn,
                          context_state=ContextState(),
                          exporter_manager=exporter_manager) as runner:
            assert await runner.result(to_type=str) == "answered without the tool"

    assert _status("tool") == [SpanStatusCode.ERROR]
    tool_span = next(span for span in EXPORTED if span.name == "tool")
    assert tool_span.status.message == "TimeoutError"  # the type only, not the message
    # the workflow recovered, so its own spans are not failures
    assert all(span.status.code == SpanStatusCode.OK for span in EXPORTED if span.name != "tool")


def test_function_end_records_the_exception_type():
    events: list[IntermediateStep] = []
    context = Context.get()
    subscription = context.intermediate_step_manager.subscribe(events.append)
    try:
        with pytest.raises(ValueError):
            with context.push_active_function("parse", input_data="x"):
                raise ValueError("bad input: secret-ish text")
    finally:
        subscription.unsubscribe()

    end = next(event for event in events if event.event_type == IntermediateStepType.FUNCTION_END)
    assert end.payload.metadata.error == {"type": "ValueError"}


async def test_a_cancelled_function_records_the_cancellation():
    events: list[IntermediateStep] = []
    context = Context.get()
    subscription = context.intermediate_step_manager.subscribe(events.append)
    try:
        with pytest.raises(asyncio.CancelledError):
            with context.push_active_function("slow_tool", input_data="x"):
                raise asyncio.CancelledError
    finally:
        subscription.unsubscribe()

    end = next(event for event in events if event.event_type == IntermediateStepType.FUNCTION_END)
    assert end.payload.metadata.error == {"type": "CancelledError"}


def test_a_stream_closed_early_is_not_recorded_as_an_error():
    events: list[IntermediateStep] = []
    context = Context.get()
    subscription = context.intermediate_step_manager.subscribe(events.append)

    def stream():
        with context.push_active_function("stream", input_data="x"):
            yield 1
            yield 2

    try:
        chunks = stream()
        next(chunks)
        chunks.close()  # the consumer stopped reading
    finally:
        subscription.unsubscribe()

    end = next(event for event in events if event.event_type == IntermediateStepType.FUNCTION_END)
    assert getattr(end.payload.metadata, "error", None) is None


def _event(event_type: IntermediateStepType, uuid: str, metadata=None) -> IntermediateStep:
    return IntermediateStep(parent_id="root",
                            function_ancestry=InvocationNode(function_id=uuid, function_name="step", parent_id=None),
                            payload=IntermediateStepPayload(event_type=event_type,
                                                            UUID=uuid,
                                                            name="step",
                                                            event_timestamp=datetime.now().timestamp(),
                                                            data=StreamEventData(input="in"),
                                                            metadata=metadata))


@pytest.mark.parametrize(
    "metadata, expected",
    [
        (None, (SpanStatusCode.OK, None)),
        (TraceMetadata(error={"type": "RuntimeError"}), (SpanStatusCode.ERROR, "RuntimeError")),
        ({
            "error": {
                "type": "CancelledError"
            }
        }, (SpanStatusCode.ERROR, "CancelledError")),
        (TraceMetadata(chat_responses=["ok"]), (SpanStatusCode.OK, None)),
        (TraceMetadata(error="connection reset"), (SpanStatusCode.ERROR, "Error")),
        # error=str(exception) for an exception without a message, e.g. TimeoutError()
        (TraceMetadata(error=""), (SpanStatusCode.ERROR, "Error")),
    ],
    ids=["no_error", "trace_metadata_error", "dict_error", "other_metadata", "string_error", "empty_string_error"])
def test_span_exporter_sets_the_status_from_the_end_event(metadata, expected):
    exporter = CollectingSpanExporter()
    exporter.export(_event(IntermediateStepType.FUNCTION_START, "s1"))
    span = exporter._outstanding_spans["s1"]
    exporter.export(_event(IntermediateStepType.FUNCTION_END, "s1", metadata=metadata))

    assert (span.status.code, span.status.message) == expected
