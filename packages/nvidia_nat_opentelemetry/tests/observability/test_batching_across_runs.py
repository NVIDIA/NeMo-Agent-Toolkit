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
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter
from nat.runtime.session import SessionManager

EXPORT_CALLS: list[int] = []
RUNS = 4
SPANS_PER_RUN = 6  # workflow + its function + four tool calls


class RecordingOtelExporter(OtelSpanExporter):

    async def export_otel_spans(self, spans):
        EXPORT_CALLS.append(len(spans))


class RecordingExporterConfig(TelemetryExporterBaseConfig, name="batching_test_otel"):
    batch_size: int = 100
    flush_interval: float = 3600.0


class EchoToolConfig(FunctionBaseConfig, name="batching_test_tool"):
    pass


class CallsToolConfig(FunctionBaseConfig, name="batching_test_workflow"):
    pass


@pytest.fixture(name="register_components", scope="module", autouse=True)
def register_components_fixture():

    @register_telemetry_exporter(config_type=RecordingExporterConfig)
    async def _exporter(config: RecordingExporterConfig, builder: Builder):
        yield RecordingOtelExporter(batch_size=config.batch_size, flush_interval=config.flush_interval)

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


async def test_spans_stay_batched_after_the_first_run():
    config = Config.model_validate({
        "general": {
            "telemetry": {
                "tracing": {
                    "recording": {
                        "_type": "batching_test_otel"
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
        exporter = (await session_manager.workflow.get_all_exporters())["recording"]
        try:
            for _ in range(RUNS):
                before = len(EXPORT_CALLS)
                async with session_manager.session() as session:
                    async with session.run("hi") as runner:
                        assert await runner.result(to_type=str) == "hi"
                await exporter.wait_for_tasks()
                await asyncio.sleep(0)
                calls_per_run.append(EXPORT_CALLS[before:])
        finally:
            await session_manager.shutdown()

    # every run's spans leave in one batch, not one export call per span
    assert calls_per_run == [[SPANS_PER_RUN]] * RUNS
