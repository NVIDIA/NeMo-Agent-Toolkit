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

import typing

import pytest

from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.function import Function
from nat.observability.exporter_manager import ExporterManager
from nat.runtime.runner import Runner


class _DummyFunction:
    has_single_output = True
    has_streaming_output = True
    instance_name = "workflow"

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
@pytest.mark.asyncio
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
