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

import io
import time

import pytest
from fastapi import FastAPI
from httpx_sse import aconnect_sse

from _utils.dask_utils import await_job
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import Message
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.object_store.in_memory_object_store import InMemoryObjectStoreConfig
from nat.test.functions import EchoFunctionConfig
from nat.test.functions import StreamingEchoFunctionConfig
from nat.test.utils import build_nat_client
from nat.utils.type_utils import override


class CustomWorker(FastApiFrontEndPluginWorker):

    @override
    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder):

        await super().add_routes(app, builder)

        # Add custom routes here
        @app.get("/custom")
        async def custom_route():
            return {"message": "This is a custom route"}


@pytest.mark.parametrize("fn_use_openai_api", [True, False])
async def test_generate_and_openai_single(fn_use_openai_api: bool):

    front_end_config = FastApiFrontEndConfig()

    config = Config(
        general=GeneralConfig(front_end=front_end_config),
        workflow=EchoFunctionConfig(use_openai_api=fn_use_openai_api),
    )

    workflow_path = front_end_config.workflow.path
    oai_path = front_end_config.workflow.openai_api_path

    assert workflow_path is not None
    assert oai_path is not None

    async with build_nat_client(config) as client:

        # Test both the function accepting OAI and also using the OAI API
        if (fn_use_openai_api):
            response = await client.post(
                workflow_path, json=ChatRequest(messages=[Message(content="Hello", role="user")]).model_dump())

            assert response.status_code == 200
            assert ChatResponse.model_validate(response.json()).choices[0].message.content == "Hello"

        else:
            response = await client.post(workflow_path, json={"message": "Hello"})

            assert response.status_code == 200
            assert response.json() == {"value": "Hello"}

        response = await client.post(oai_path,
                                     json=ChatRequest(messages=[Message(content="Hello", role="user")]).model_dump())

        assert response.status_code == 200
        oai_response = ChatResponse.model_validate(response.json())

        assert oai_response.choices[0].message.content == "Hello"


@pytest.mark.parametrize("fn_use_openai_api", [True, False])
async def test_generate_and_openai_stream(fn_use_openai_api: bool):

    if (fn_use_openai_api):
        values = ChatRequest(messages=[Message(content="Hello", role="user")]).model_dump()
    values = ["a", "b", "c", "d"]

    front_end_config = FastApiFrontEndConfig()

    config = Config(
        general=GeneralConfig(front_end=front_end_config),
        workflow=StreamingEchoFunctionConfig(use_openai_api=fn_use_openai_api),
    )

    workflow_path = front_end_config.workflow.path
    oai_path = front_end_config.workflow.openai_api_path

    assert workflow_path is not None
    assert oai_path is not None

    async with build_nat_client(config) as client:

        response = []

        if (fn_use_openai_api):
            async with aconnect_sse(client,
                                    "POST",
                                    f"{workflow_path}/stream",
                                    json=ChatRequest(messages=[Message(content=x, role="user")
                                                               for x in values]).model_dump()) as event_source:
                async for sse in event_source.aiter_sse():
                    response.append(ChatResponseChunk.model_validate(sse.json()).choices[0].delta.content or "")

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
                                json=ChatRequest(messages=[Message(content=x, role="user")
                                                           for x in values]).model_dump()) as event_source:
            async for sse in event_source.aiter_sse():
                response_oai.append(ChatResponseChunk.model_validate(sse.json()).choices[0].delta.content or "")

            assert event_source.response.status_code == 200
            assert response_oai == values


async def test_custom_endpoint():

    config = Config(
        general=GeneralConfig(front_end=FastApiFrontEndConfig()),
        workflow=EchoFunctionConfig(),
    )

    async with build_nat_client(config, worker_class=CustomWorker) as client:
        response = await client.get("/custom")

        assert response.status_code == 200
        assert response.json() == {"message": "This is a custom route"}


async def test_specified_endpoints():

    config = Config(
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

    async with build_nat_client(config) as client:
        # response = await client.get("/constant_get")

        # assert response.status_code == 200
        # assert response.json() == {"message": "Constant"}

        response = await client.post("/echo_post", json={"message": "Hello"})

        assert response.status_code == 200
        assert response.json() == {"value": "Hello"}


@pytest.mark.parametrize("use_sync_timeout", [True, False])
@pytest.mark.parametrize("fn_use_openai_api", [True, False])
async def test_generate_async(fn_use_openai_api: bool, use_sync_timeout: bool):
    if (fn_use_openai_api):
        pytest.skip("Async support for OpenAI API is not implemented yet")

    front_end_config = FastApiFrontEndConfig()

    config = Config(
        general=GeneralConfig(front_end=front_end_config),
        workflow=EchoFunctionConfig(use_openai_api=fn_use_openai_api),
    )

    job_id = f"test_generate_async_{use_sync_timeout}_{fn_use_openai_api}"

    workflow_path = f"{front_end_config.workflow.path}/async"
    # oai_path = front_end_config.workflow.openai_api_path
    async with build_nat_client(config) as client:

        # Test both the function accepting OAI and also using the OAI API
        if (fn_use_openai_api):
            # response = await client.post(
            #     workflow_path, json=ChatRequest(messages=[Message(content="Hello", role="user")]).model_dump())

            # assert response.status_code == 200
            # assert ChatResponse.model_validate(response.json()).choices[0].message.content == "Hello"
            assert True  # TODO: Implement async support in the EchoFunctionConfig

        else:
            payload = {"message": "Hello", "job_id": job_id}
            if use_sync_timeout:
                payload["sync_timeout"] = 10

            response = await client.post(workflow_path, json=payload)

            if use_sync_timeout:
                assert response.status_code == 200
                response_body = response.json()
                assert response_body["job_id"] == job_id
                assert response_body["status"] == "success"
                assert response_body["output"] == {"value": "Hello"}
            else:
                assert response.status_code == 202
                assert response.json() == {"job_id": job_id, "status": "submitted"}

        expected_status_values = ("running", "success", "submitted")
        status_path = f"{workflow_path}/job/{job_id}"

        status = None
        timeout = 10  # Wait for up to 10 seconds
        deadline = time.time() + timeout
        while status != "success":
            response = await client.get(status_path)

            assert response.status_code == 200
            data = response.json()
            status = data["status"]

            assert status in expected_status_values
            if status != "success":
                assert time.time() < deadline, "Job did not complete in time"
                await await_job(job_id, timeout=timeout)


async def test_async_job_status_not_found():
    front_end_config = FastApiFrontEndConfig()

    config = Config(
        general=GeneralConfig(front_end=front_end_config),
        workflow=EchoFunctionConfig(use_openai_api=False),
    )

    workflow_path = f"{front_end_config.workflow.path}/async"

    async with build_nat_client(config) as client:
        status_path = f"{workflow_path}/job/non_existent_job"

        response = await client.get(status_path)

        assert response.status_code == 404


async def test_static_file_endpoints():
    # Configure the in-memory object store
    object_store_name = "test_store"
    file_path = "folder/testfile.txt"
    file_content = b"Hello, world!"
    updated_content = b"Updated content!"
    content_type = "text/plain"

    config = Config(
        general=GeneralConfig(front_end=FastApiFrontEndConfig(object_store=object_store_name)),
        object_stores={object_store_name: InMemoryObjectStoreConfig()},
        workflow=EchoFunctionConfig(),  # Dummy workflow, not used here
    )

    async with build_nat_client(config) as client:
        # POST: Upload a new file
        response = await client.post(
            f"/static/{file_path}",
            files={"file": ("testfile.txt", io.BytesIO(file_content), content_type)},
        )
        assert response.status_code == 200
        assert response.json()["filename"] == file_path

        # GET: Retrieve the file
        response = await client.get(f"/static/{file_path}")
        assert response.status_code == 200
        assert response.content == file_content
        assert response.headers["content-type"].startswith(content_type)
        assert response.headers["content-disposition"].endswith("testfile.txt")

        # POST again: Should fail with 409 (already exists)
        response = await client.post(
            f"/static/{file_path}",
            files={"file": ("testfile.txt", io.BytesIO(file_content), content_type)},
        )
        assert response.status_code == 409

        # PUT: Upsert (update) the file
        response = await client.put(
            f"/static/{file_path}",
            files={"file": ("testfile.txt", io.BytesIO(updated_content), content_type)},
        )
        assert response.status_code == 200
        assert response.json()["filename"] == file_path

        # GET: Retrieve the updated file
        response = await client.get(f"/static/{file_path}")
        assert response.status_code == 200
        assert response.content == updated_content

        # DELETE: Remove the file
        response = await client.delete(f"/static/{file_path}")
        assert response.status_code == 204

        # DELETE: Delete again (idempotent but should still result in a 404)
        response = await client.delete(f"/static/{file_path}")
        assert response.status_code == 404

        # GET: Should now 404
        response = await client.get(f"/static/{file_path}")
        assert response.status_code == 404
