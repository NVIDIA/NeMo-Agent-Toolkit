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
"""A run that fails or is cancelled still ends, and exports, its root workflow span."""

import asyncio
import json
from collections.abc import AsyncGenerator

import pytest

from nat.builder.builder import Builder
from nat.builder.context import ContextState
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.span import Span
from nat.observability.exporter.span_exporter import SpanExporter
from nat.observability.exporter_manager import ExporterManager
from nat.runtime.runner import Runner

EXPORTED: list[Span] = []
WORKFLOW_ENDS: list[str] = []
STARTED = asyncio.Event()


class CollectingSpanExporter(SpanExporter[Span, Span]):

    def export(self, event) -> None:
        if getattr(event, "event_type", None) == IntermediateStepType.WORKFLOW_END:
            WORKFLOW_ENDS.append(event.UUID)
        super().export(event)

    async def export_processed(self, item: Span) -> None:
        EXPORTED.append(item)


class FailingToolConfig(FunctionBaseConfig, name="root_span_failing_tool"):
    pass


class CallsToolConfig(FunctionBaseConfig, name="root_span_workflow"):
    pass


class StreamsThenFailsConfig(FunctionBaseConfig, name="root_span_stream_workflow"):
    pass


class WaitsForeverConfig(FunctionBaseConfig, name="root_span_slow_workflow"):
    pass


@pytest.fixture(name="register_workflows", scope="module", autouse=True)
def register_workflows_fixture():

    @register_function(config_type=FailingToolConfig)
    async def _tool(config, builder: Builder):

        async def _inner(message: str) -> str:
            if "fail" in message:
                raise RuntimeError("tool backend unavailable")
            return f"tool: {message}"

        yield _inner

    @register_function(config_type=CallsToolConfig)
    async def _workflow(config, builder: Builder):
        tool = await builder.get_function("tool")

        async def _inner(message: str) -> str:
            return await tool.ainvoke(message, to_type=str)

        yield _inner

    @register_function(config_type=StreamsThenFailsConfig)
    async def _stream_workflow(config, builder: Builder):

        async def _inner(message: str) -> AsyncGenerator[str]:
            yield "first chunk"
            if "fail" in message:
                raise RuntimeError("stream broke")
            yield "second chunk"

        yield _inner

    @register_function(config_type=WaitsForeverConfig)
    async def _slow_workflow(config, builder: Builder):

        async def _inner(message: str) -> str:
            STARTED.set()
            await asyncio.sleep(3600)
            return message

        yield _inner


@pytest.fixture(name="reset_state", autouse=True)
def reset_state_fixture():
    EXPORTED.clear()
    WORKFLOW_ENDS.clear()
    STARTED.clear()


def _runner(entry_fn, message: str) -> Runner:
    exporter_manager = ExporterManager()
    exporter_manager.add_exporter("collect", CollectingSpanExporter())
    return Runner(input_message=message,
                  entry_fn=entry_fn,
                  context_state=ContextState(),
                  exporter_manager=exporter_manager)


def _roots() -> list[Span]:
    """The exported root spans; also checks the run sent exactly one WORKFLOW_END."""
    assert len(WORKFLOW_ENDS) == 1, f"expected one WORKFLOW_END, got {len(WORKFLOW_ENDS)}"
    return [span for span in EXPORTED if span.parent is None]


def _end_error(span: Span):
    """The `error` entry the WORKFLOW_END event put in the span's metadata, if any."""
    metadata = json.loads(span.attributes["nat.metadata"])
    return metadata.get("error")


def _assert_complete_trace():
    span_ids = {span.context.span_id for span in EXPORTED}
    assert all(span.parent.context.span_id in span_ids for span in EXPORTED if span.parent is not None)


async def test_a_successful_run_exports_one_root_span_without_an_error():
    async with WorkflowBuilder() as builder:
        await builder.add_function("tool", FailingToolConfig())
        entry_fn = await builder.add_function("workflow", CallsToolConfig())
        async with _runner(entry_fn, "hello") as runner:
            assert await runner.result(to_type=str) == "tool: hello"

    roots = _roots()
    assert len(roots) == 1
    assert _end_error(roots[0]) is None


async def test_a_failed_run_exports_its_root_span():
    async with WorkflowBuilder() as builder:
        await builder.add_function("tool", FailingToolConfig())
        entry_fn = await builder.add_function("workflow", CallsToolConfig())
        with pytest.raises(RuntimeError, match="tool backend unavailable"):
            async with _runner(entry_fn, "please fail") as runner:
                await runner.result(to_type=str)

    roots = _roots()
    assert len(roots) == 1, "the workflow's root span must be exported"
    assert roots[0].attributes["nat.event_type"] == IntermediateStepType.WORKFLOW_START.value
    assert _end_error(roots[0]) == {"type": "RuntimeError"}
    _assert_complete_trace()


async def test_a_stream_that_fails_exports_its_root_span_without_a_final_output():
    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function("workflow", StreamsThenFailsConfig())
        with pytest.raises(RuntimeError, match="stream broke"):
            async with _runner(entry_fn, "please fail") as runner:
                async for _ in runner.result_stream(to_type=str):
                    pass

    roots = _roots()
    assert len(roots) == 1
    assert _end_error(roots[0]) == {"type": "RuntimeError"}
    # the partial chunks are not reported as the run's output
    assert "output.value" not in roots[0].attributes


async def test_a_cancelled_run_exports_its_root_span():
    """A client that disconnects cancels the run; the trace should still show it."""

    async def run(entry_fn):
        async with _runner(entry_fn, "hi") as runner:
            return await runner.result(to_type=str)

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function("workflow", WaitsForeverConfig())
        task = asyncio.create_task(run(entry_fn))
        await asyncio.wait_for(STARTED.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    roots = _roots()
    assert len(roots) == 1
    assert _end_error(roots[0]) == {"type": "CancelledError"}


async def test_a_consumer_that_stops_reading_ends_the_root_span_without_an_error():
    """Closing the stream early (a client that got what it needed) is not a failure."""
    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function("workflow", StreamsThenFailsConfig())
        runner = _runner(entry_fn, "hi")
        await runner.__aenter__()
        stream = runner.result_stream(to_type=str)
        assert await anext(stream) == "first chunk"
        await stream.aclose()
        # unchanged from before: the runner never saw the workflow complete
        with pytest.raises(ValueError, match="without completing the workflow"):
            await runner.__aexit__(None, None, None)

    roots = _roots()
    assert len(roots) == 1
    assert _end_error(roots[0]) is None
    # the chunks the consumer did read are kept as the run's output
    assert "first chunk" in roots[0].attributes["output.value"]
