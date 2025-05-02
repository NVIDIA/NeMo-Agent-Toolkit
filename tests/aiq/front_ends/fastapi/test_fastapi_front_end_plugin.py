# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from contextlib import asynccontextmanager

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport
from httpx import AsyncClient
from httpx_sse import aconnect_sse

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponse
from aiq.data_models.api_server import AIQChatResponseChunk
from aiq.data_models.api_server import Message
from aiq.data_models.config import AIQConfig
from aiq.data_models.config import GeneralConfig
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from aiq.test.functions import EchoFunctionConfig
from aiq.test.functions import StreamingEchoFunctionConfig
from aiq.utils.type_utils import override


class TestCustomWorker(FastApiFrontEndPluginWorker):

    @override
    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder):

        await super().add_routes(app, builder)

        # Add custom routes here
        @app.get("/custom")
        async def custom_route():
            return {"message": "This is a custom route"}


@asynccontextmanager
async def _build_client(config: AIQConfig,
                        worker_class: type[FastApiFrontEndPluginWorker] = FastApiFrontEndPluginWorker):

    worker = worker_class(config)

    app = worker.build_app()

    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client


@pytest.mark.parametrize("fn_use_openai_api", [True, False])
async def test_generate_and_openai_single(fn_use_openai_api: bool):

    front_end_config = FastApiFrontEndConfig()

    config = AIQConfig(
        general=GeneralConfig(front_end=front_end_config),
        workflow=EchoFunctionConfig(use_openai_api=fn_use_openai_api),
    )

    workflow_path = front_end_config.workflow.path
    oai_path = front_end_config.workflow.openai_api_path

    assert workflow_path is not None
    assert oai_path is not None

    async with _build_client(config) as client:

        # Test both the function accepting OAI and also using the OAI API
        if (fn_use_openai_api):
            response = await client.post(
                workflow_path, json=AIQChatRequest(messages=[Message(content="Hello", role="user")]).model_dump())

            assert response.status_code == 200
            assert AIQChatResponse.model_validate(response.json()).choices[0].message.content == "Hello"

        else:
            response = await client.post(workflow_path, json={"message": "Hello"})

            assert response.status_code == 200
            assert response.json() == {"value": "Hello"}

        response = await client.post(oai_path,
                                     json=AIQChatRequest(messages=[Message(content="Hello", role="user")]).model_dump())

        assert response.status_code == 200
        oai_response = AIQChatResponse.model_validate(response.json())

        assert oai_response.choices[0].message.content == "Hello"


@pytest.mark.parametrize("fn_use_openai_api", [True, False])
async def test_generate_and_openai_stream(fn_use_openai_api: bool):

    if (fn_use_openai_api):
        values = AIQChatRequest(messages=[Message(content="Hello", role="user")]).model_dump()
    values = ["a", "b", "c", "d"]

    front_end_config = FastApiFrontEndConfig()

    config = AIQConfig(
        general=GeneralConfig(front_end=front_end_config),
        workflow=StreamingEchoFunctionConfig(use_openai_api=fn_use_openai_api),
    )

    workflow_path = front_end_config.workflow.path
    oai_path = front_end_config.workflow.openai_api_path

    assert workflow_path is not None
    assert oai_path is not None

    async with _build_client(config) as client:

        response = []

        if (fn_use_openai_api):
            async with aconnect_sse(client,
                                    "POST",
                                    f"{workflow_path}/stream",
                                    json=AIQChatRequest(messages=[Message(content=x, role="user")
                                                                  for x in values]).model_dump()) as event_source:
                async for sse in event_source.aiter_sse():
                    response.append(AIQChatResponseChunk.model_validate(sse.json()).choices[0].message.content or "")

                assert event_source.response.status_code == 200
                assert response == values

        else:
            async with aconnect_sse(client, "POST", f"{workflow_path}/stream",
                                    json={"message": values}) as event_source:
                async for sse in event_source.aiter_sse():
                    response.append(sse.json()["value"])

                assert event_source.response.status_code == 200
                assert response == values

        response_oai: list[str] = []

        async with aconnect_sse(client,
                                "POST",
                                f"{oai_path}/stream",
                                json=AIQChatRequest(messages=[Message(content=x, role="user")
                                                              for x in values]).model_dump()) as event_source:
            async for sse in event_source.aiter_sse():
                response_oai.append(AIQChatResponseChunk.model_validate(sse.json()).choices[0].message.content or "")

            assert event_source.response.status_code == 200
            assert response_oai == values


async def test_custom_endpoint():

    config = AIQConfig(
        general=GeneralConfig(front_end=FastApiFrontEndConfig()),
        workflow=EchoFunctionConfig(),
    )

    async with _build_client(config, worker_class=TestCustomWorker) as client:
        response = await client.get("/custom")

        assert response.status_code == 200
        assert response.json() == {"message": "This is a custom route"}


async def test_specified_endpoints():

    config = AIQConfig(
        general=GeneralConfig(front_end=FastApiFrontEndConfig(endpoints=[
            # TODO(MDD): Uncomment this when the constant function is implemented
            # FastApiFrontEndConfig.Endpoint(
            #     path="/constant_get", method="GET", description="Constant function", function_name="constant"),
            FastApiFrontEndConfig.Endpoint(
                path="/echo_post", method="POST", description="Echo function", function_name="echo"),
        ])),
        functions={
            "echo": EchoFunctionConfig(),  # "constant": ConstantFunctionConfig(response="Constant"),
        },
        workflow=EchoFunctionConfig(),
    )

    async with _build_client(config) as client:
        # response = await client.get("/constant_get")

        # assert response.status_code == 200
        # assert response.json() == {"message": "Constant"}

        response = await client.post("/echo_post", json={"message": "Hello"})

        assert response.status_code == 200
        assert response.json() == {"value": "Hello"}
