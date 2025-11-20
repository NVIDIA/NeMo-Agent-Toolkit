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
import sys
import typing
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.front_end import FrontEndBaseConfig
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem

logger = logging.getLogger(__name__)

YAML_EXTENSIONS = (".yaml", ".yml")


def _is_reserved(path: Path) -> bool:
    """
    Check if a path is reserved in the current Python version and platform.

    On Windows, this function checks if the path is reserved in the current Python version.
    On other platforms, returns False
    """
    if sys.platform != "win32":
        return False
    if sys.version_info >= (3, 13):
        return os.path.isreserved(path)
    return path.is_reserved()


class EvaluateRequest(BaseModel):
    """Request model for the evaluate endpoint."""
    config_file: str = Field(description="Path to the configuration file for evaluation")
    job_id: str | None = Field(default=None, description="Unique identifier for the evaluation job")
    reps: int = Field(default=1, gt=0, description="Number of repetitions for the evaluation, defaults to 1")
    expiry_seconds: int = Field(
        default=3600,
        gt=0,
        description="Optional time (in seconds) before the job expires. Clamped between 600 (10 min) and 86400 (24h).")

    @field_validator('job_id', mode='after')
    @classmethod
    def validate_job_id(cls, job_id: str):
        job_id = job_id.strip()
        job_id_path = Path(job_id)
        if len(job_id_path.parts) > 1 or job_id_path.resolve().name != job_id:
            raise ValueError(
                f"Job ID '{job_id}' contains invalid characters. Only alphanumeric characters and underscores are"
                " allowed.")

        if _is_reserved(job_id_path):
            # reserved names is Windows specific
            raise ValueError(f"Job ID '{job_id}' is a reserved name. Please choose a different name.")

        return job_id

    @field_validator('config_file', mode='after')
    @classmethod
    def validate_config_file(cls, config_file: str):
        config_file = config_file.strip()
        config_file_path = Path(config_file).resolve()

        # Ensure the config file is a YAML file
        if config_file_path.suffix.lower() not in YAML_EXTENSIONS:
            raise ValueError(f"Config file '{config_file}' must be a YAML file with one of the following extensions: "
                             f"{', '.join(YAML_EXTENSIONS)}")

        if _is_reserved(config_file_path):
            # reserved names is Windows specific
            raise ValueError(f"Config file '{config_file}' is a reserved name. Please choose a different name.")

        if not config_file_path.exists():
            raise ValueError(f"Config file '{config_file}' does not exist. Please provide a valid path.")

        return config_file


class BaseAsyncResponse(BaseModel):
    """Base model for async responses."""
    job_id: str = Field(description="Unique identifier for the job")
    status: str = Field(description="Current status of the job")


class EvaluateResponse(BaseAsyncResponse):
    """Response model for the evaluate endpoint."""
    pass


class AsyncGenerateResponse(BaseAsyncResponse):
    """Response model for the async generation endpoint."""
    pass


class BaseAsyncStatusResponse(BaseModel):
    """Base model for async status responses."""
    job_id: str = Field(description="Unique identifier for the evaluation job")
    status: str = Field(description="Current status of the evaluation job")
    error: str | None = Field(default=None, description="Error message if the job failed")
    created_at: datetime = Field(description="Timestamp when the job was created")
    updated_at: datetime = Field(description="Timestamp when the job was last updated")
    expires_at: datetime | None = Field(default=None, description="Timestamp when the job will expire")


class EvaluateStatusResponse(BaseAsyncStatusResponse):
    """Response model for the evaluate status endpoint."""
    config_file: str = Field(description="Path to the configuration file used for evaluation")
    output_path: str | None = Field(default=None,
                                    description="Path to the output file if the job completed successfully")


class AsyncGenerationStatusResponse(BaseAsyncStatusResponse):
    output: dict | None = Field(
        default=None,
        description="Output of the generate request, this is only available if the job completed successfully.")


class EvaluateItemRequest(BaseModel):
    """Request model for single-item evaluation endpoint."""
    item: EvalInputItem = Field(description="Single evaluation input item to evaluate")
    evaluator_name: str = Field(description="Name of the evaluator to use (must match config)")


class EvaluateItemResponse(BaseModel):
    """Response model for single-item evaluation endpoint."""
    success: bool = Field(description="Whether the evaluation completed successfully")
    result: EvalOutputItem | None = Field(default=None, description="Evaluation result if successful")
    error: str | None = Field(default=None, description="Error message if evaluation failed")


class FastApiFrontEndConfig(FrontEndBaseConfig, name="fastapi"):
    """
    A FastAPI based front end that allows a NAT workflow to be served as a microservice.
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
        openai_api_v1_path: str | None = Field(
            default=None,
            description=("Path for the OpenAI v1 Chat Completions API compatible endpoint. "
                         "If provided, creates a single endpoint that handles both streaming and "
                         "non-streaming requests based on the 'stream' parameter, following the "
                         "OpenAI Chat Completions API specification exactly."),
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
    scheduler_address: str | None = Field(
        default=None,
        description=("Address of the Dask scheduler to use for async jobs. If None, a Dask local cluster is created. "
                     "Note: This requires the optional dask dependency to be installed."))
    db_url: str | None = Field(
        default=None,
        description=
        "SQLAlchemy database URL for storing async job metadata, if unset a temporary SQLite database is used.")
    max_running_async_jobs: int = Field(
        default=10,
        description=(
            "Maximum number of async jobs to run concurrently, this controls the number of dask workers created. "
            "This parameter is only used when scheduler_address is `None` and a Dask local cluster is created."),
        ge=1)
    dask_workers: typing.Literal["threads", "processes"] = Field(
        default="processes",
        description=(
            "Type of Dask workers to use. Options are 'threads' for Threaded Dask workers or 'processes' for "
            "Process based Dask workers. This parameter is only used when scheduler_address is `None` and a local Dask "
            "cluster is created."),
    )
    dask_log_level: str = Field(
        default="WARNING",
        description="Logging level for Dask.",
    )
    step_adaptor: StepAdaptorConfig = StepAdaptorConfig()

    workflow: typing.Annotated[EndpointBase, Field(description="Endpoint for the default workflow.")] = EndpointBase(
        method="POST",
        path="/generate",
        websocket_path="/websocket",
        openai_api_path="/chat",
        openai_api_v1_path="/v1/chat/completions",
        description="Executes the default NAT workflow from the loaded configuration ",
    )

    evaluate: typing.Annotated[EndpointBase, Field(description="Endpoint for evaluating workflows.")] = EndpointBase(
        method="POST",
        path="/evaluate",
        description="Evaluates the performance and accuracy of the workflow on a dataset",
    )

    evaluate_item: typing.Annotated[EndpointBase,
                                    Field(description="Endpoint for evaluating a single item.")] = EndpointBase(
                                        method="POST",
                                        path="/evaluate/item",
                                        description="Evaluate a single item with a specified evaluator",
                                    )

    oauth2_callback_path: str | None = Field(
        default="/auth/redirect",
        description="OAuth2.0 authentication callback endpoint. If None, no OAuth2 callback endpoint is created.")

    endpoints: list[Endpoint] = Field(
        default_factory=list,
        description=("Additional endpoints to add to the FastAPI app which run functions within the NAT configuration. "
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
        description=("The NAT runner class to use when launching the FastAPI app from multiple processes. "
                     "Each runner is responsible for loading and running the NAT workflow. "
                     "Note: This is different from the worker class used by Gunicorn."),
    )

    object_store: ObjectStoreRef | None = Field(
        default=None,
        description=(
            "Object store reference for the FastAPI app. If present, static files can be uploaded via a POST "
            "request to '/static' and files will be served from the object store. The files will be served from the "
            "object store at '/static/{file_name}'."))


# Compatibility aliases with previous releases
AIQEvaluateRequest = EvaluateRequest
AIQEvaluateResponse = EvaluateResponse
AIQAsyncGenerateResponse = AsyncGenerateResponse
AIQEvaluateStatusResponse = EvaluateStatusResponse
AIQAsyncGenerationStatusResponse = AsyncGenerationStatusResponse
