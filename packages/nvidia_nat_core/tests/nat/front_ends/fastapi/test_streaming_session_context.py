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
"""A streamed workflow run sees the user and conversation of its request, like a non-streamed one."""

import asyncio

import pytest

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.runtime.user_manager import UserManager
from nat.test.utils import build_nat_client


class EchoContextConfig(FunctionBaseConfig, name="stream_session_test_echo_context"):
    pass


@pytest.fixture(name="register_workflow", scope="module", autouse=True)
def register_workflow_fixture():

    @register_function(config_type=EchoContextConfig)
    async def _workflow(config, builder: Builder):

        async def _inner(message: str) -> str:
            context = Context.get()
            return f"user={context.user_id} conversation={context.conversation_id}"

        yield FunctionInfo.from_fn(_inner, description="echoes the request context")


MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("path, body",
                         [
                             ("/v1/chat/completions", {
                                 "messages": MESSAGES, "stream": True
                             }),
                             ("/chat/stream", {
                                 "messages": MESSAGES
                             }),
                             ("/generate/stream", {
                                 "message": "hi"
                             }),
                             ("/generate/full", {
                                 "message": "hi"
                             }),
                         ],
                         ids=["chat_completions", "chat_stream", "generate_stream", "generate_full"])
async def test_streamed_run_sees_the_request_context(tmp_path, monkeypatch, path, body):
    config_file = tmp_path / "config.yml"
    config_file.write_text("workflow:\n  _type: stream_session_test_echo_context\n")
    monkeypatch.setenv("NAT_CONFIG_FILE", str(config_file))
    config = Config.model_validate({"workflow": {"_type": "stream_session_test_echo_context"}})
    expected_user = UserManager._user_info_from_session_cookie("alice-session").get_user_id()
    headers = {"Cookie": "nat-session=alice-session", "conversation-id": "conv-42"}

    async with build_nat_client(config) as client:
        response = await client.post(path, json=body, headers=headers)

    assert response.status_code == 200
    assert f"user={expected_user} conversation=conv-42" in response.text, response.text[:500]


async def _first_body_chunk_over_asgi_2_3(app, path: str) -> bytes:
    """Send a GET to ``app`` the way uvicorn does (ASGI spec 2.3) and return the first non-empty body chunk.

    With spec 2.3, ``StreamingResponse`` listens for a disconnect on ``receive`` while it streams, so that
    listener takes the request's only ``http.request`` message. ``receive`` then blocks until the client leaves.
    """
    request_sent = False
    client_gone = asyncio.Event()
    first_chunk: asyncio.Future = asyncio.get_running_loop().create_future()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await client_gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body") and not first_chunk.done():
            first_chunk.set_result(message["body"])

    scope = {
        "type": "http",
        "asgi": {
            "version": "3.0", "spec_version": "2.3"
        },
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"test")],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        return await asyncio.wait_for(first_chunk, timeout=10)
    finally:
        client_gone.set()
        await asyncio.wait_for(task, timeout=10)


@pytest.mark.parametrize("path", ["/echo_get/stream", "/echo_get/full"], ids=["stream", "full"])
async def test_get_streaming_route_does_not_wait_for_the_request_body(tmp_path, monkeypatch, path):
    from asgi_lifespan import LifespanManager

    from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

    config_file = tmp_path / "config.yml"
    config_file.write_text("workflow:\n  _type: stream_session_test_echo_context\n")
    monkeypatch.setenv("NAT_CONFIG_FILE", str(config_file))
    config = Config.model_validate({
        "general": {
            "front_end":
                FastApiFrontEndConfig(endpoints=[
                    FastApiFrontEndConfig.Endpoint(
                        path="/echo_get", method="GET", description="echo", function_name="echo")
                ])
        },
        "functions": {
            "echo": {
                "_type": "stream_session_test_echo_context"
            }
        },
        "workflow": {
            "_type": "stream_session_test_echo_context"
        },
    })
    app = FastApiFrontEndPluginWorker(config).build_app()

    async with LifespanManager(app):
        chunk = await _first_body_chunk_over_asgi_2_3(app, path)

    assert chunk
