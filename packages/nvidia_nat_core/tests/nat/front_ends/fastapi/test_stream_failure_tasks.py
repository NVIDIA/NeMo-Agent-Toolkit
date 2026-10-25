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
"""A streamed request whose workflow fails leaves no task with an unretrieved exception behind."""

import asyncio
import gc
from collections.abc import AsyncGenerator

import pytest

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.test.utils import build_nat_client


class StreamFailsConfig(FunctionBaseConfig, name="stream_failure_tasks_test_workflow"):
    pass


@pytest.fixture(name="register_workflow", scope="module", autouse=True)
def register_workflow_fixture():

    @register_function(config_type=StreamFailsConfig)
    async def _workflow(config: StreamFailsConfig, builder: Builder):

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
                             ("/v1/workflow/atif", {
                                 "messages": MESSAGES
                             }),
                         ],
                         ids=["chat_completions", "generate_stream", "workflow_atif"])
async def test_a_failed_stream_leaves_no_unretrieved_task_exception(tmp_path, monkeypatch, path, body):
    config_file = tmp_path / "config.yml"
    config_file.write_text("workflow:\n  _type: stream_failure_tasks_test_workflow\n")
    monkeypatch.setenv("NAT_CONFIG_FILE", str(config_file))
    config = Config.model_validate({"workflow": {"_type": "stream_failure_tasks_test_workflow"}})

    loop = asyncio.get_running_loop()
    reported: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    try:
        async with build_nat_client(config) as client:
            response = await client.post(path, json=body)
            assert response.status_code == 200
        for _ in range(3):
            await asyncio.sleep(0)
        gc.collect()  # a task's unretrieved exception is reported when the task is collected
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert [context.get("message") for context in reported] == []


async def test_an_item_put_after_the_queue_closed_is_dropped():
    from nat.front_ends.fastapi.intermediate_steps_subscriber import _put_unless_closed
    from nat.utils.producer_consumer_queue import AsyncIOProducerConsumerQueue

    queue: AsyncIOProducerConsumerQueue[str] = AsyncIOProducerConsumerQueue()
    await queue.close()
    await _put_unless_closed(queue, "late step")  # no QueueClosed


async def test_other_put_errors_still_surface():
    from nat.front_ends.fastapi.intermediate_steps_subscriber import _put_unless_closed

    class BrokenQueue:

        async def put(self, item):
            raise RuntimeError("queue broke")

    with pytest.raises(RuntimeError, match="queue broke"):
        await _put_unless_closed(BrokenQueue(), "step")
