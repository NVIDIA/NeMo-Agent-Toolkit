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
"""Spans of concurrent turns are exported under their own turn's identity."""

import asyncio
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.plugins.a365.telemetry import a365_exporter
from nat.plugins.a365.turn_context import A365TurnIdentity
from nat.plugins.a365.turn_context import set_turn_identity
from nat.runtime.session import SessionManager

ROUNDS = 3


class TurnTestExporterConfig(TelemetryExporterBaseConfig, name="a365_turn_identity_test"):
    pass


class SlowEchoConfig(FunctionBaseConfig, name="a365_turn_identity_test_tool"):
    pass


class CallsToolConfig(FunctionBaseConfig, name="a365_turn_identity_test_workflow"):
    pass


@pytest.fixture(name="register_components", scope="module", autouse=True)
def register_components_fixture():

    @register_telemetry_exporter(config_type=TurnTestExporterConfig)
    async def _exporter(config: TurnTestExporterConfig, builder: Builder):
        # the batch settings a default A365TelemetryExporter config passes
        yield a365_exporter.A365OtelExporter(agent_id="fallback-agent",
                                             tenant_id="fallback-tenant",
                                             token_resolver=lambda agent_id, tenant_id: "token",
                                             batch_size=100,
                                             flush_interval=5.0)

    @register_function(config_type=SlowEchoConfig)
    async def _tool(config, builder: Builder):

        async def _inner(text: str) -> str:
            await asyncio.sleep(0.05)  # lets the two turns interleave
            return text

        yield FunctionInfo.from_fn(_inner, description="echo")

    @register_function(config_type=CallsToolConfig)
    async def _workflow(config, builder: Builder):
        tool = await builder.get_function("tool")

        async def _inner(text: str) -> str:
            for _ in range(3):
                text = await tool.ainvoke(text)
            return text

        yield FunctionInfo.from_fn(_inner, description="calls the tool three times")


async def test_concurrent_turns_keep_their_own_identity():
    exported: list[tuple[str, str, str]] = []  # (agent id and tenant id on the span, turn that produced it)

    def export(readable_spans):
        for span in readable_spans:
            turn = next(value for value in span.attributes.values() if isinstance(value, str) and "turn-" in value)
            exported.append((span.attributes[a365_exporter.GEN_AI_AGENT_ID_KEY],
                             span.attributes[a365_exporter.TENANT_ID_KEY],
                             turn.split("turn-")[1][0]))
        return SpanExportResult.SUCCESS

    config = Config.model_validate({
        "general": {
            "telemetry": {
                "tracing": {
                    "a365": {
                        "_type": "a365_turn_identity_test"
                    }
                }
            }
        },
        "functions": {
            "tool": {
                "_type": "a365_turn_identity_test_tool"
            }
        },
        "workflow": {
            "_type": "a365_turn_identity_test_workflow"
        },
    })

    async def run_turn(session_manager: SessionManager, name: str):
        identity = A365TurnIdentity(agent_app_id=f"agent-{name}", tenant_id=f"tenant-{name}", on_behalf_user_id=None)
        with set_turn_identity(identity):
            async with session_manager.session() as session:
                async with session.run(f"turn-{name}") as runner:
                    await runner.result(to_type=str)

    with patch.object(a365_exporter, "Agent365Exporter") as sdk_exporter:
        sdk_exporter.return_value = Mock(export=Mock(side_effect=export))
        async with WorkflowBuilder.from_config(config=config) as builder:
            session_manager = await SessionManager.create(config=config, shared_builder=builder, max_concurrency=8)
            try:
                for _ in range(ROUNDS):
                    await asyncio.gather(run_turn(session_manager, "A"), run_turn(session_manager, "B"))
            finally:
                await session_manager.shutdown()

    assert len(exported) == ROUNDS * 2 * 5  # workflow, its function and three tool calls per turn
    assert [(agent, tenant, turn) for agent, tenant, turn in exported
            if agent != f"agent-{turn}" or tenant != f"tenant-{turn}"] == []
