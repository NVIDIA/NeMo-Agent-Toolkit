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
"""A workflow that fails mid-stream sends its error as an SSE event, which SSE clients can read."""

import json
from collections.abc import AsyncGenerator

import pytest
from httpx_sse import aconnect_sse

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.test.utils import build_nat_client


class FailsMidStreamConfig(FunctionBaseConfig, name="sse_error_test_fails_mid_stream"):
    pass


@pytest.fixture(name="register_workflow", scope="module", autouse=True)
def register_workflow_fixture():

    @register_function(config_type=FailsMidStreamConfig)
    async def _workflow(config, builder: Builder):

        async def _inner(message: ChatRequest) -> AsyncGenerator[ChatResponseChunk]:
            yield ChatResponseChunk.from_string("partial ", id_="chatcmpl-1", model="test-model")
            raise RuntimeError("upstream LLM failed")

        yield FunctionInfo.from_fn(_inner, description="fails after the first chunk")


MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("path, body",
                         [
                             ("/v1/chat/completions", {
                                 "messages": MESSAGES, "stream": True
                             }),
                             ("/generate/stream", {
                                 "messages": MESSAGES
                             }),
                             ("/generate/full", {
                                 "messages": MESSAGES
                             }),
                             ("/v1/workflow/atif", {
                                 "messages": MESSAGES
                             }),
                             ("/v1/workflow/full", {
                                 "messages": MESSAGES
                             }),
                         ],
                         ids=["chat_completions", "generate_stream", "generate_full", "workflow_atif", "workflow_full"])
async def test_a_mid_stream_failure_arrives_as_an_error_event(tmp_path, monkeypatch, path, body):
    config_file = tmp_path / "config.yml"
    config_file.write_text("workflow:\n  _type: sse_error_test_fails_mid_stream\n")
    monkeypatch.setenv("NAT_CONFIG_FILE", str(config_file))
    config = Config.model_validate({"workflow": {"_type": "sse_error_test_fails_mid_stream"}})

    async with build_nat_client(config) as client:
        async with aconnect_sse(client, "POST", path, json=body) as event_source:
            events = [event async for event in event_source.aiter_sse()]

    errors = [event for event in events if event.event == "error"]
    assert len(errors) == 1, [(event.event, event.data[:80]) for event in events]
    error = json.loads(errors[0].data)
    assert error["code"] == "workflow_error"
    assert error["message"] == "upstream LLM failed"
    assert error["details"] == "RuntimeError"
    # the chunk sent before the failure still reaches the client
    assert any(event.event != "error" and "partial " in event.data for event in events[:-1])
    # the error is the last thing in the stream
    assert events[-1] is errors[0]
