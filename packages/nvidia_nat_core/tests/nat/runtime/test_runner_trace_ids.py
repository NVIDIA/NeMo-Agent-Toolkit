# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing
from unittest.mock import patch

import pytest

from nat.builder.component_utils import WORKFLOW_COMPONENT_NAME
from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.function import Function
from nat.builder.intermediate_step_manager import IntermediateStepManager
from nat.observability.exporter_manager import ExporterManager
from nat.runtime.runner import Runner


class _DummyConfig:
    """Mock config for _DummyFunction."""
    name = None
    type = "dummy_workflow"


class _DummyFunction:
    has_single_output = True
    has_streaming_output = True
    instance_name = "workflow"
    config = _DummyConfig()

    def convert(self, v, to_type):
        return v

    async def ainvoke(self, _message, to_type=None):
        ctx = Context.get()
        assert isinstance(ctx.workflow_trace_id, int) and ctx.workflow_trace_id != 0
        return {"ok": True}

    async def astream(self, _message, to_type=None):
        ctx = Context.get()
        assert isinstance(ctx.workflow_trace_id, int) and ctx.workflow_trace_id != 0
        yield "chunk-1"


class _DummyExporterManager:

    def start(self, context_state=None):

        class _Ctx:

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return _Ctx()


@pytest.mark.parametrize("method", ["result", "result_stream"])  # result vs stream
@pytest.mark.parametrize("existing_run", [True, False])
@pytest.mark.parametrize("existing_trace", [True, False])
async def test_runner_trace_and_run_ids(existing_trace: bool, existing_run: bool, method: str):
    ctx_state = ContextState.get()

    # Seed existing values according to parameters
    seeded_trace = int("f" * 32, 16) if existing_trace else None
    seeded_run = "existing-run-id" if existing_run else None

    tkn_trace = ctx_state.workflow_trace_id.set(seeded_trace)
    tkn_run = ctx_state.workflow_run_id.set(seeded_run)

    try:
        runner = Runner(
            "msg",
            typing.cast(Function, _DummyFunction()),
            ctx_state,
            typing.cast(ExporterManager, _DummyExporterManager()),
        )
        async with runner:
            if method == "result":
                out = await runner.result()
                assert out == {"ok": True}
            else:
                chunks: list[str] = []
                async for c in runner.result_stream():
                    chunks.append(c)
                assert chunks == ["chunk-1"]

        # After run, context should be restored to seeded values
        assert ctx_state.workflow_trace_id.get() == seeded_trace
        assert ctx_state.workflow_run_id.get() == seeded_run
    finally:
        ctx_state.workflow_trace_id.reset(tkn_trace)
        ctx_state.workflow_run_id.reset(tkn_run)


@pytest.mark.parametrize(
    "config_name,instance_name,config_type,expected_workflow_name",
    [
        # Case 1: config.name is set - should use it
        ("custom_name", "some_instance", "some_type", "custom_name"),
        # Case 2: config.name is None, instance_name is valid - should use instance_name
        (None, "my_workflow", "some_type", "my_workflow"),
        # Case 3: config.name is None, instance_name is placeholder - should fall back to config.type
        (None, WORKFLOW_COMPONENT_NAME, "react_agent", "react_agent"),
    ],
    ids=["config_name_set", "instance_name_fallback", "config_type_fallback"],
)
async def test_runner_workflow_name_resolution(
    config_name: str | None,
    instance_name: str,
    config_type: str,
    expected_workflow_name: str,
):
    """Test that Runner resolves workflow_name correctly based on config and instance_name."""

    class _TestConfig:
        name = config_name
        type = config_type

    class _TestFunction:
        has_single_output = True
        has_streaming_output = False
        config = _TestConfig()

        def __init__(self):
            self.instance_name = instance_name

        def convert(self, v, to_type):
            return v

        async def ainvoke(self, _message, to_type=None):
            return {"ok": True}

    ctx_state = ContextState.get()

    # Capture the workflow_name passed to intermediate step manager
    captured_workflow_name = None
    original_push = IntermediateStepManager.push_intermediate_step

    def capture_push(self, payload):
        nonlocal captured_workflow_name
        # Capture the name from WORKFLOW_START event
        if payload.event_type.name == "WORKFLOW_START":
            captured_workflow_name = payload.name
        return original_push(self, payload)

    with patch.object(
            IntermediateStepManager,
            "push_intermediate_step",
            capture_push,
    ):
        runner = Runner(
            "msg",
            typing.cast(Function, _TestFunction()),
            ctx_state,
            typing.cast(ExporterManager, _DummyExporterManager()),
        )
        async with runner:
            await runner.result()

    assert captured_workflow_name == expected_workflow_name
