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


@pytest.fixture(name="register_workflow", scope="module", autouse=True)
def register_workflow_fixture():

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
    # every call is finished exactly once
    assert len(finishes) == client.create_call.call_count
    assert len({id(finish.args[0]) for finish in finishes}) == len(finishes)
    root_finish = finishes[-1]
    exception = root_finish.kwargs.get("exception")
    assert isinstance(exception, Exception)
    assert "RuntimeError" in str(exception)
    # the outputs dict is still passed (it holds usage info); only the exception is added
    assert isinstance(root_finish.args[1], dict)


@pytest.mark.parametrize(
    "metadata, expected",
    [
        (None, None),
        ({
            "error": {
                "type": "TimeoutError"
            }
        }, {
            "type": "TimeoutError"
        }),
        ({
            "chat_responses": ["ok"]
        }, None),
        ({
            "error": "connection reset"
        }, {
            "type": "Error"
        }),
        (
            {
                # error=str(exception) for an exception without a message, e.g. TimeoutError()
                "error": ""
            },
            {
                "type": "Error"
            }),
    ],
    ids=["no_metadata", "error_entry", "other_metadata", "string_error", "empty_string_error"])
def test_end_event_error(metadata, expected):
    """AutoGen's handler already reports failures as a string `error`; those count as failed too."""
    from nat.data_models.intermediate_step import IntermediateStep
    from nat.data_models.intermediate_step import IntermediateStepPayload
    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.data_models.invocation_node import InvocationNode
    from nat.plugins.weave.weave_exporter import WeaveExporter

    step = IntermediateStep(parent_id="root",
                            function_ancestry=InvocationNode(function_id="f", function_name="f"),
                            payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_END, metadata=metadata))
    assert WeaveExporter._end_event_error(step) == expected


async def test_successful_run_root_call_is_finished_without_an_error():
    client = await _run("hello")

    assert client.finish_call.call_args_list
    assert all(call.kwargs.get("exception") is None for call in client.finish_call.call_args_list)
