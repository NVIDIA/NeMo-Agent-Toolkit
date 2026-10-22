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
"""Weave shows a failed run's root call as failed, not as a normal completion."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.builder import Builder
from nat.builder.context import ContextState
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.observability.exporter_manager import ExporterManager
from nat.runtime.runner import Runner


class WeaveFailingWorkflowConfig(FunctionBaseConfig, name="weave_failing_workflow"):
    pass


@pytest.fixture(scope="module", autouse=True)
def _register():

    @register_function(config_type=WeaveFailingWorkflowConfig)
    async def _workflow(config, builder: Builder):

        async def _inner(message: str) -> str:
            if "fail" in message:
                raise RuntimeError("backend unavailable")
            return f"done: {message}"

        yield _inner


async def _run(message: str):
    from nat.plugins.weave.weave_exporter import WeaveExporter

    client = MagicMock()
    calls: list[MagicMock] = []

    def create_call(*args, **kwargs):
        calls.append(MagicMock(id=f"call-{len(calls)}", summary={}))
        return calls[-1]

    client.create_call.side_effect = create_call
    # like Weave's own call stack: the latest call is the parent of the next one
    with patch("nat.plugins.weave.weave_exporter.weave_client_context.require_weave_client", return_value=client), \
         patch("nat.plugins.weave.weave_exporter.get_current_call", side_effect=lambda: calls[-1] if calls else None):
        exporter_manager = ExporterManager()
        exporter_manager.add_exporter("weave", WeaveExporter(project="test"))
        async with WorkflowBuilder() as builder:
            entry_fn = await builder.add_function("workflow", WeaveFailingWorkflowConfig())
            try:
                async with Runner(input_message=message,
                                  entry_fn=entry_fn,
                                  context_state=ContextState(),
                                  exporter_manager=exporter_manager) as runner:
                    await runner.result(to_type=str)
            except RuntimeError:
                pass
    return client


async def test_failed_run_root_call_is_finished_with_the_error():
    client = await _run("please fail")

    finishes = client.finish_call.call_args_list
    assert finishes, "the root call must be finished"
    root_finish = finishes[-1]
    exception = root_finish.kwargs.get("exception")
    assert isinstance(exception, Exception)
    assert "RuntimeError" in str(exception)


async def test_successful_run_root_call_is_finished_without_an_error():
    client = await _run("hello")

    assert client.finish_call.call_args_list
    assert all(call.kwargs.get("exception") is None for call in client.finish_call.call_args_list)
