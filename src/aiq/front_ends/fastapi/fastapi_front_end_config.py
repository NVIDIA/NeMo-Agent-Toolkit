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
import typing

from pydantic import BaseModel
from pydantic import Field

from aiq.data_models.front_end import FrontEndBaseConfig
from aiq.data_models.step_adaptor import StepAdaptorConfig

logger = logging.getLogger(__name__)


class FastApiFrontEndConfig(FrontEndBaseConfig, name="fastapi"):
    """
    A FastAPI based front end that allows an AgentIQ workflow to be served as a microservice.
    """

    class EndpointBase(BaseModel):

        method: typing.Literal["GET", "POST", "PUT", "DELETE"]
        description: str
        path: str | None = Field(
            default=None,
            description=("Path for the default workflow. If None, no workflow endpoint is created."),
        )
        websocket_path: str | None = Field(
            default=None,
            description=("Path for the websocket. If None, no websocket is created."),
        )
        openai_api_path: str | None = Field(
            default=None,
            description=("Path for the default workflow using the OpenAI API Specification. "
                         "If None, no workflow endpoint with the OpenAI API Specification is created."),
        )

    class Endpoint(EndpointBase):
        function_name: str = Field(description="The name of the function to call for this endpoint")

    class CrossOriginResourceSharing(BaseModel):
        allow_origins: list[str] | None = Field(
            default=None, description=" A list of origins that should be permitted to make cross-origin requests.")
        allow_origin_regex: str | None = Field(
            default=None,
            description="A permitted regex string to match against origins to make cross-origin requests",
        )
        allow_methods: list[str] | None = Field(
            default_factory=lambda: ['GET'],
            description="A list of HTTP methods that should be allowed for cross-origin requests.")
        allow_headers: list[str] | None = Field(
            default_factory=list,
            description="A list of HTTP request headers that should be supported for cross-origin requests.")
        allow_credentials: bool | None = Field(
            default=False,
            description="Indicate that cookies should be supported for cross-origin requests.",
        )
        expose_headers: list[str] | None = Field(
            default_factory=list,
            description="Indicate any response headers that should be made accessible to the browser.",
        )
        max_age: int | None = Field(
            default=600,
            description="Sets a maximum time in seconds for browsers to cache CORS responses.",
        )

    root_path: str = Field(default="", description="The root path for the API")
    host: str = Field(default="localhost", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to bind the server to", ge=0, le=65535)
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    workers: int = Field(default=1, description="Number of workers to run", ge=1)
    step_adaptor: StepAdaptorConfig = StepAdaptorConfig()

    workflow: typing.Annotated[EndpointBase, Field(description="Endpoint for the default workflow.")] = EndpointBase(
        method="POST",
        path="/generate",
        websocket_path="/websocket",
        openai_api_path="/chat",
        description="Executes the default AgentIQ workflow from the loaded configuration ",
    )

    endpoints: list[Endpoint] = Field(
        default_factory=list,
        description=(
            "Additional endpoints to add to the FastAPI app which run functions within the AgentIQ configuration. "
            "Each endpoint must have a unique path."))

    cors: CrossOriginResourceSharing = Field(
        default_factory=CrossOriginResourceSharing,
        description="Cross origin resource sharing configuration for the FastAPI app")

    use_gunicorn: bool = Field(
        default=False,
        description="Use Gunicorn to run the FastAPI app",
    )
    runner_class: str | None = Field(
        default=None,
        description=("The AgentIQ runner class to use when launching the FastAPI app from multiple processes. "
                     "Each runner is responsible for loading and running the AgentIQ workflow. "
                     "Note: This is different from the worker class used by Gunicorn."),
    )
