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

import logging
import os
import typing
from abc import ABC
from abc import abstractmethod
from contextlib import asynccontextmanager
from functools import partial

from fastapi import Body
from fastapi import FastAPI
from fastapi import Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponse
from aiq.data_models.api_server import AIQChatResponseChunk
from aiq.data_models.api_server import AIQResponseIntermediateStep
from aiq.data_models.config import AIQConfig
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from aiq.front_ends.fastapi.response_helpers import generate_single_response
from aiq.front_ends.fastapi.response_helpers import generate_streaming_response_as_str
from aiq.front_ends.fastapi.step_adaptor import StepAdaptor
from aiq.front_ends.fastapi.websocket import AIQWebSocket
from aiq.runtime.session import AIQSessionManager

logger = logging.getLogger(__name__)


class FastApiFrontEndPluginWorkerBase(ABC):

    def __init__(self, config: AIQConfig):
        self._config = config

        assert isinstance(config.general.front_end,
                          FastApiFrontEndConfig), ("Front end config is not FastApiFrontEndConfig")

        self._front_end_config = config.general.front_end

    @property
    def config(self) -> AIQConfig:
        return self._config

    @property
    def front_end_config(self) -> FastApiFrontEndConfig:

        return self._front_end_config

    def build_app(self) -> FastAPI:

        # Create the FastAPI app and configure it
        @asynccontextmanager
        async def lifespan(starting_app: FastAPI):

            logger.debug("Starting AgentIQ server from process %s", os.getpid())

            async with WorkflowBuilder.from_config(self.config) as builder:

                await self.configure(starting_app, builder)

                yield

            logger.debug("Closing AgentIQ server from process %s", os.getpid())

        aiq_app = FastAPI(lifespan=lifespan)

        return aiq_app

    @abstractmethod
    async def configure(self, app: FastAPI, builder: WorkflowBuilder):
        pass

    @abstractmethod
    def get_step_adaptor(self) -> StepAdaptor:
        pass


class RouteInfo(BaseModel):

    function_name: str | None


class FastApiFrontEndPluginWorker(FastApiFrontEndPluginWorkerBase):

    def get_step_adaptor(self) -> StepAdaptor:

        return StepAdaptor(self.front_end_config.step_adaptor)

    async def configure(self, app: FastAPI, builder: WorkflowBuilder):

        # Do things like setting the base URL and global configuration options
        app.root_path = self.front_end_config.root_path

        await self.add_routes(app, builder)

    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder):

        await self.add_default_route(app, AIQSessionManager(builder.build()))

        for ep in self.front_end_config.endpoints:

            entry_workflow = builder.build(entry_function=ep.function_name)

            await self.add_route(app, endpoint=ep, session_manager=AIQSessionManager(entry_workflow))

    async def add_default_route(self, app: FastAPI, session_manager: AIQSessionManager):

        await self.add_route(app, self.front_end_config.workflow, session_manager)

    async def add_route(self,
                        app: FastAPI,
                        endpoint: FastApiFrontEndConfig.EndpointBase,
                        session_manager: AIQSessionManager):

        workflow = session_manager.workflow

        if (endpoint.websocket_path):
            app.add_websocket_route(endpoint.websocket_path,
                                    partial(AIQWebSocket, session_manager, self.get_step_adaptor()))

        GenerateBodyType = workflow.input_schema  # pylint: disable=invalid-name
        GenerateStreamResponseType = workflow.streaming_output_schema  # pylint: disable=invalid-name
        GenerateSingleResponseType = workflow.single_output_schema  # pylint: disable=invalid-name

        # Ensure that the input is in the body. POD types are treated as query parameters
        if (not issubclass(GenerateBodyType, BaseModel)):
            GenerateBodyType = typing.Annotated[GenerateBodyType, Body()]

        response_500 = {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred"
                    }
                }
            },
        }

        def get_single_endpoint(result_type: type | None):

            async def get_single(response: Response):

                response.headers["Content-Type"] = "application/json"

                return await generate_single_response(None, session_manager, result_type=result_type)

            return get_single

        def get_streaming_endpoint(streaming: bool, result_type: type | None, output_type: type | None):

            async def get_stream():

                return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                         content=generate_streaming_response_as_str(
                                             None,
                                             session_manager=session_manager,
                                             streaming=streaming,
                                             step_adaptor=self.get_step_adaptor(),
                                             result_type=result_type,
                                             output_type=output_type))

            return get_stream

        def post_single_endpoint(request_type: type, result_type: type | None):

            async def post_single(response: Response, payload: request_type):

                response.headers["Content-Type"] = "application/json"

                return await generate_single_response(payload, session_manager, result_type=result_type)

            return post_single

        def post_streaming_endpoint(request_type: type,
                                    streaming: bool,
                                    result_type: type | None,
                                    output_type: type | None):

            async def post_stream(payload: request_type):

                return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                         content=generate_streaming_response_as_str(
                                             payload,
                                             session_manager=session_manager,
                                             streaming=streaming,
                                             step_adaptor=self.get_step_adaptor(),
                                             result_type=result_type,
                                             output_type=output_type))

            return post_stream

        if (endpoint.path):
            if (endpoint.method == "GET"):

                app.add_api_route(
                    path=endpoint.path,
                    endpoint=get_single_endpoint(result_type=GenerateSingleResponseType),
                    methods=[endpoint.method],
                    response_model=GenerateSingleResponseType,
                    description=endpoint.description,
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.path}/stream",
                    endpoint=get_streaming_endpoint(streaming=True,
                                                    result_type=GenerateStreamResponseType,
                                                    output_type=GenerateStreamResponseType),
                    methods=[endpoint.method],
                    response_model=GenerateStreamResponseType,
                    description=endpoint.description,
                    responses={500: response_500},
                )

            elif (endpoint.method == "POST"):

                app.add_api_route(
                    path=endpoint.path,
                    endpoint=post_single_endpoint(request_type=GenerateBodyType,
                                                  result_type=GenerateSingleResponseType),
                    methods=[endpoint.method],
                    response_model=GenerateSingleResponseType,
                    description=endpoint.description,
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.path}/stream",
                    endpoint=post_streaming_endpoint(request_type=GenerateBodyType,
                                                     streaming=True,
                                                     result_type=GenerateStreamResponseType,
                                                     output_type=GenerateStreamResponseType),
                    methods=[endpoint.method],
                    response_model=GenerateStreamResponseType,
                    description=endpoint.description,
                    responses={500: response_500},
                )

            else:
                raise ValueError(f"Unsupported method {endpoint.method}")

        if (endpoint.openai_api_path):
            if (endpoint.method == "GET"):

                app.add_api_route(
                    path=endpoint.openai_api_path,
                    endpoint=get_single_endpoint(result_type=AIQChatResponse),
                    methods=[endpoint.method],
                    response_model=AIQChatResponse,
                    description=endpoint.description,
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.openai_api_path}/stream",
                    endpoint=get_streaming_endpoint(streaming=True,
                                                    result_type=AIQChatResponseChunk,
                                                    output_type=AIQChatResponseChunk),
                    methods=[endpoint.method],
                    response_model=AIQChatResponseChunk,
                    description=endpoint.description,
                    responses={500: response_500},
                )

            elif (endpoint.method == "POST"):

                app.add_api_route(
                    path=endpoint.openai_api_path,
                    endpoint=post_single_endpoint(request_type=AIQChatRequest, result_type=AIQChatResponse),
                    methods=[endpoint.method],
                    response_model=AIQChatResponse,
                    description=endpoint.description,
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.openai_api_path}/stream",
                    endpoint=post_streaming_endpoint(request_type=AIQChatRequest,
                                                     streaming=True,
                                                     result_type=AIQChatResponseChunk,
                                                     output_type=AIQChatResponseChunk),
                    methods=[endpoint.method],
                    response_model=AIQChatResponseChunk | AIQResponseIntermediateStep,
                    description=endpoint.description,
                    responses={500: response_500},
                )

            else:
                raise ValueError(f"Unsupported method {endpoint.method}")
