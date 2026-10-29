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
"""Span batching keeps working after the first run of a long-lived workflow (nat serve, nat eval)."""

import asyncio

import pytest

from nat.builder.builder import Builder
from nat.builder.context import ContextState
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.span import Span
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.processor.batching_processor import BatchingProcessor
from nat.observability.processor.processor import Processor
from nat.plugins.opentelemetry.otel_span import OtelSpan
from nat.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter
from nat.runtime.session import SessionManager

EXPORT_CALLS: list[int] = []
RUNS = 4
SPANS_PER_RUN = 6  # workflow + its function + four tool calls


class SlowProcessor(Processor[Span, Span]):
    """A processor that awaits before passing the span on, like the redaction processors do."""

    def __init__(self, delay: float):
        super().__init__()
        self._delay = delay

    async def process(self, item: Span) -> Span:
        await asyncio.sleep(self._delay)
        return item


class ShutdownOnlyProcessor(Processor[Span, Span]):
    """A user processor with cleanup but no flush."""

    def __init__(self):
        super().__init__()
        self.shut_down = False

    async def process(self, item: Span) -> Span:
        return item

    async def shutdown(self) -> None:
        self.shut_down = True


class FlushRecorder(Processor[Span, Span]):
    """A buffering processor that records when its flush starts and ends."""

    def __init__(self, name: str, events: list[str]):
        super().__init__()
        self._name = name
        self._events = events

    async def process(self, item: Span) -> Span:
        return item

    async def flush(self) -> None:
        self._events.append(f"{self._name} start")
        await asyncio.sleep(0.01)
        self._events.append(f"{self._name} end")


class RecordingOtelExporter(OtelSpanExporter):

    async def export_otel_spans(self, spans: list[OtelSpan]):
        EXPORT_CALLS.append(len(spans))


class RecordingExporterConfig(TelemetryExporterBaseConfig, name="batching_test_otel"):
    batch_size: int = 100
    flush_interval: float = 3600.0
    slow_first_processor: bool = False
    slow_processor_delay: float = 0.02


class EchoToolConfig(FunctionBaseConfig, name="batching_test_tool"):
    pass


class CallsToolConfig(FunctionBaseConfig, name="batching_test_workflow"):
    pass


@pytest.fixture(name="register_components", scope="module", autouse=True)
def register_components_fixture():

    @register_telemetry_exporter(config_type=RecordingExporterConfig)
    async def _exporter(config: RecordingExporterConfig, builder: Builder):
        exporter = RecordingOtelExporter(batch_size=config.batch_size, flush_interval=config.flush_interval)
        if config.slow_first_processor:
            exporter.add_processor(SlowProcessor(config.slow_processor_delay), position=0)
        yield exporter

    @register_function(config_type=EchoToolConfig)
    async def _tool(config, builder: Builder):

        async def _inner(text: str) -> str:
            return text

        yield FunctionInfo.from_fn(_inner, description="echo")

    @register_function(config_type=CallsToolConfig)
    async def _workflow(config, builder: Builder):
        tool = await builder.get_function("tool")

        async def _inner(text: str) -> str:
            for _ in range(4):
                text = await tool.ainvoke(text)
            return text

        yield FunctionInfo.from_fn(_inner, description="calls the tool four times")


@pytest.fixture(name="clear_calls", autouse=True)
def clear_calls_fixture():
    EXPORT_CALLS.clear()


@pytest.mark.parametrize("slow_first_processor", [False, True], ids=["plain", "slow_processor_before_batcher"])
async def test_spans_stay_batched_after_the_first_run(slow_first_processor):
    config = Config.model_validate({
        "general": {
            "telemetry": {
                "tracing": {
                    "recording": {
                        "_type": "batching_test_otel", "slow_first_processor": slow_first_processor
                    }
                }
            }
        },
        "functions": {
            "tool": {
                "_type": "batching_test_tool"
            }
        },
        "workflow": {
            "_type": "batching_test_workflow"
        },
    })
    calls_per_run: list[list[int]] = []
    async with WorkflowBuilder.from_config(config=config) as builder:
        session_manager = await SessionManager.create(config=config, shared_builder=builder, max_concurrency=1)
        try:
            for _ in range(RUNS):
                before = len(EXPORT_CALLS)
                async with session_manager.session() as session:
                    async with session.run("hi") as runner:
                        assert await runner.result(to_type=str) == "hi"
                await asyncio.sleep(0)
                calls_per_run.append(EXPORT_CALLS[before:])
        finally:
            await session_manager.shutdown()

    # every run's spans leave in one batch, not one export call per span, and none are left behind
    assert calls_per_run == [[SPANS_PER_RUN]] * RUNS
    assert sum(EXPORT_CALLS) == SPANS_PER_RUN * RUNS


async def test_only_the_original_exporter_shuts_its_batcher_down():
    exporter = RecordingOtelExporter(batch_size=100, flush_interval=3600.0)
    batcher = next(p for p in exporter._processors if isinstance(p, BatchingProcessor))

    await exporter.create_isolated_instance(ContextState())._cleanup()
    assert not batcher._shutdown_requested

    await exporter._cleanup()
    assert batcher._shutdown_requested


async def test_an_export_still_running_when_the_run_ends_is_not_cancelled(monkeypatch):
    from nat.observability.exporter import processing_exporter

    # the run's spans take longer to reach the batcher than the run end waits for
    monkeypatch.setattr(processing_exporter, "_RUN_END_WAIT_SECONDS", 0.05)
    config = Config.model_validate({
        "general": {
            "telemetry": {
                "tracing": {
                    "recording": {
                        "_type": "batching_test_otel",
                        "slow_first_processor": True,
                        "slow_processor_delay": 0.3,
                        "flush_interval": 0.2,
                    }
                }
            }
        },
        "functions": {
            "tool": {
                "_type": "batching_test_tool"
            }
        },
        "workflow": {
            "_type": "batching_test_workflow"
        },
    })
    async with WorkflowBuilder.from_config(config=config) as builder:
        session_manager = await SessionManager.create(config=config, shared_builder=builder, max_concurrency=1)
        try:
            async with session_manager.session() as session:
                async with session.run("hi") as runner:
                    assert await runner.result(to_type=str) == "hi"
            await asyncio.sleep(1.0)  # the late spans reach the batcher and leave with its timed flush
        finally:
            await session_manager.shutdown()

    assert sum(EXPORT_CALLS) == SPANS_PER_RUN


async def test_a_run_never_shuts_down_a_shared_processor():
    exporter = RecordingOtelExporter(batch_size=100, flush_interval=3600.0)
    processor = ShutdownOnlyProcessor()
    exporter.add_processor(processor, position=0)

    await exporter.create_isolated_instance(ContextState())._cleanup()
    assert not processor.shut_down

    await exporter._cleanup()
    assert processor.shut_down


async def test_a_run_flushes_processors_in_pipeline_order():
    """An upstream flush can forward items to a downstream buffer, so the downstream one flushes after it."""
    events: list[str] = []
    exporter = RecordingOtelExporter(batch_size=100, flush_interval=3600.0)
    exporter.add_processor(FlushRecorder("downstream", events), position=0)
    exporter.add_processor(FlushRecorder("upstream", events), position=0)

    await exporter.create_isolated_instance(ContextState())._cleanup()

    assert events[:4] == ["upstream start", "upstream end", "downstream start", "downstream end"]


async def test_spans_still_on_their_way_when_the_builder_exits_are_exported(monkeypatch):
    """The builder's exit drains the batcher its per-run copies shared, even though the original never started."""
    from nat.observability.exporter import processing_exporter

    monkeypatch.setattr(processing_exporter, "_RUN_END_WAIT_SECONDS", 0.05)
    config = Config.model_validate({
        "general": {
            "telemetry": {
                "tracing": {
                    "recording": {
                        "_type": "batching_test_otel", "slow_first_processor": True, "slow_processor_delay": 0.3
                    }
                }
            }
        },
        "functions": {
            "tool": {
                "_type": "batching_test_tool"
            }
        },
        "workflow": {
            "_type": "batching_test_workflow"
        },
    })
    async with WorkflowBuilder.from_config(config=config) as builder:
        session_manager = await SessionManager.create(config=config, shared_builder=builder, max_concurrency=1)
        try:
            async with session_manager.session() as session:
                async with session.run("hi") as runner:
                    assert await runner.result(to_type=str) == "hi"
        finally:
            await session_manager.shutdown()
    await asyncio.sleep(0.6)  # the late spans reach the batcher after the builder has closed

    assert sum(EXPORT_CALLS) == SPANS_PER_RUN
