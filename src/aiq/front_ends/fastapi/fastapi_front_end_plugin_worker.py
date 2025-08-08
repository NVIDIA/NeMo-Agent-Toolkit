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

import asyncio
import logging
import os
import time
import typing
from abc import ABC
from abc import abstractmethod
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks
from fastapi import Body
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi import UploadFile
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import Field
from starlette.websockets import WebSocket

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.api_server import ChatRequest
from aiq.data_models.api_server import ChatResponse
from aiq.data_models.api_server import ChatResponseChunk
from aiq.data_models.api_server import ResponseIntermediateStep
from aiq.data_models.config import Config
from aiq.data_models.object_store import KeyAlreadyExistsError
from aiq.data_models.object_store import NoSuchKeyError
from aiq.eval.config import EvaluationRunOutput
from aiq.eval.evaluate import EvaluationRun
from aiq.eval.evaluate import EvaluationRunConfig
from aiq.front_ends.fastapi.auth_flow_handlers.http_flow_handler import HTTPAuthenticationFlowHandler
from aiq.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import FlowState
from aiq.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler
from aiq.front_ends.fastapi.fastapi_front_end_config import AsyncGenerateResponse
from aiq.front_ends.fastapi.fastapi_front_end_config import AsyncGenerationStatusResponse
from aiq.front_ends.fastapi.fastapi_front_end_config import EvaluateRequest
from aiq.front_ends.fastapi.fastapi_front_end_config import EvaluateResponse
from aiq.front_ends.fastapi.fastapi_front_end_config import EvaluateStatusResponse
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from aiq.front_ends.fastapi.job_store import JobInfo
from aiq.front_ends.fastapi.job_store import JobStore
from aiq.front_ends.fastapi.message_handler import WebSocketMessageHandler
from aiq.front_ends.fastapi.response_helpers import generate_single_response
from aiq.front_ends.fastapi.response_helpers import generate_streaming_response_as_str
from aiq.front_ends.fastapi.response_helpers import generate_streaming_response_full_as_str
from aiq.front_ends.fastapi.step_adaptor import StepAdaptor
from aiq.object_store.models import ObjectStoreItem
from aiq.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class FastApiFrontEndPluginWorkerBase(ABC):

    def __init__(self, config: Config):
        self._config = config

        assert isinstance(config.general.front_end,
                          FastApiFrontEndConfig), ("Front end config is not FastApiFrontEndConfig")

        self._front_end_config = config.general.front_end

        self._cleanup_tasks: list[str] = []
        self._cleanup_tasks_lock = asyncio.Lock()
        self._http_flow_handler: HTTPAuthenticationFlowHandler | None = HTTPAuthenticationFlowHandler()

    @property
    def config(self) -> Config:
        return self._config

    @property
    def front_end_config(self) -> FastApiFrontEndConfig:
        return self._front_end_config

    def build_app(self) -> FastAPI:

        # Create the FastAPI app and configure it
        @asynccontextmanager
        async def lifespan(starting_app: FastAPI):

            logger.debug("Starting AIQ Toolkit server from process %s", os.getpid())

            async with WorkflowBuilder.from_config(self.config) as builder:

                await self.configure(starting_app, builder)

                yield

                # If a cleanup task is running, cancel it
                async with self._cleanup_tasks_lock:

                    # Cancel all cleanup tasks
                    for task_name in self._cleanup_tasks:
                        cleanup_task: asyncio.Task | None = getattr(starting_app.state, task_name, None)
                        if cleanup_task is not None:
                            logger.info("Cancelling %s cleanup task", task_name)
                            cleanup_task.cancel()
                        else:
                            logger.warning("No cleanup task found for %s", task_name)

                    self._cleanup_tasks.clear()

            logger.debug("Closing AIQ Toolkit server from process %s", os.getpid())

        aiq_app = FastAPI(lifespan=lifespan)

        # Configure app CORS.
        self.set_cors_config(aiq_app)

        @aiq_app.middleware("http")
        async def authentication_log_filter(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
            return await self._suppress_authentication_logs(request, call_next)

        return aiq_app

    def set_cors_config(self, aiq_app: FastAPI) -> None:
        """
        Set the cross origin resource sharing configuration.
        """
        cors_kwargs = {}

        if self.front_end_config.cors.allow_origins is not None:
            cors_kwargs["allow_origins"] = self.front_end_config.cors.allow_origins

        if self.front_end_config.cors.allow_origin_regex is not None:
            cors_kwargs["allow_origin_regex"] = self.front_end_config.cors.allow_origin_regex

        if self.front_end_config.cors.allow_methods is not None:
            cors_kwargs["allow_methods"] = self.front_end_config.cors.allow_methods

        if self.front_end_config.cors.allow_headers is not None:
            cors_kwargs["allow_headers"] = self.front_end_config.cors.allow_headers

        if self.front_end_config.cors.allow_credentials is not None:
            cors_kwargs["allow_credentials"] = self.front_end_config.cors.allow_credentials

        if self.front_end_config.cors.expose_headers is not None:
            cors_kwargs["expose_headers"] = self.front_end_config.cors.expose_headers

        if self.front_end_config.cors.max_age is not None:
            cors_kwargs["max_age"] = self.front_end_config.cors.max_age

        aiq_app.add_middleware(
            CORSMiddleware,
            **cors_kwargs,
        )

    async def _suppress_authentication_logs(self, request: Request,
                                            call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """
        Intercepts authentication request and supreses logs that contain sensitive data.
        """
        from aiq.utils.log_utils import LogFilter

        logs_to_suppress: list[str] = []

        if (self.front_end_config.oauth2_callback_path):
            logs_to_suppress.append(self.front_end_config.oauth2_callback_path)

        logging.getLogger("uvicorn.access").addFilter(LogFilter(logs_to_suppress))
        try:
            response = await call_next(request)
        finally:
            logging.getLogger("uvicorn.access").removeFilter(LogFilter(logs_to_suppress))

        return response

    @abstractmethod
    async def configure(self, app: FastAPI, builder: WorkflowBuilder):
        pass

    @abstractmethod
    def get_step_adaptor(self) -> StepAdaptor:
        pass


class RouteInfo(BaseModel):

    function_name: str | None


class FastApiFrontEndPluginWorker(FastApiFrontEndPluginWorkerBase):

    def __init__(self, config: Config):
        super().__init__(config)

        self._outstanding_flows: dict[str, FlowState] = {}
        self._outstanding_flows_lock = asyncio.Lock()

    @staticmethod
    async def _periodic_cleanup(name: str, job_store: JobStore, sleep_time_sec: int = 300):
        while True:
            try:
                job_store.cleanup_expired_jobs()
                logger.debug("Expired %s jobs cleaned up", name)
            except Exception as e:
                logger.error("Error during %s job cleanup: %s", name, e)
            await asyncio.sleep(sleep_time_sec)

    async def create_cleanup_task(self, app: FastAPI, name: str, job_store: JobStore, sleep_time_sec: int = 300):
        # Schedule periodic cleanup of expired jobs on first job creation
        attr_name = f"{name}_cleanup_task"

        # Cheap check, if it doesn't exist, we will need to re-check after we acquire the lock
        if not hasattr(app.state, attr_name):
            async with self._cleanup_tasks_lock:
                if not hasattr(app.state, attr_name):
                    logger.info("Starting %s periodic cleanup task", name)
                    setattr(
                        app.state,
                        attr_name,
                        asyncio.create_task(
                            self._periodic_cleanup(name=name, job_store=job_store, sleep_time_sec=sleep_time_sec)))
                    self._cleanup_tasks.append(attr_name)

    def get_step_adaptor(self) -> StepAdaptor:

        return StepAdaptor(self.front_end_config.step_adaptor)

    async def configure(self, app: FastAPI, builder: WorkflowBuilder):

        # Do things like setting the base URL and global configuration options
        app.root_path = self.front_end_config.root_path

        await self.add_routes(app, builder)

    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder):

        await self.add_default_route(app, SessionManager(builder.build()))
        await self.add_evaluate_route(app, SessionManager(builder.build()))
        await self.add_static_files_route(app, builder)
        await self.add_authorization_route(app)

        for ep in self.front_end_config.endpoints:

            entry_workflow = builder.build(entry_function=ep.function_name)

            await self.add_route(app, endpoint=ep, session_manager=SessionManager(entry_workflow))

    async def add_default_route(self, app: FastAPI, session_manager: SessionManager):

        await self.add_route(app, self.front_end_config.workflow, session_manager)

    async def add_evaluate_route(self, app: FastAPI, session_manager: SessionManager):
        """Add the evaluate endpoint to the FastAPI app."""

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

        # Create job store for tracking evaluation jobs
        job_store = JobStore()
        # Don't run multiple evaluations at the same time
        evaluation_lock = asyncio.Lock()

        async def run_evaluation(job_id: str, config_file: str, reps: int, session_manager: SessionManager):
            """Background task to run the evaluation."""
            async with evaluation_lock:
                try:
                    # Create EvaluationRunConfig using the CLI defaults
                    eval_config = EvaluationRunConfig(config_file=Path(config_file), dataset=None, reps=reps)

                    # Create a new EvaluationRun with the evaluation-specific config
                    job_store.update_status(job_id, "running")
                    eval_runner = EvaluationRun(eval_config)
                    output: EvaluationRunOutput = await eval_runner.run_and_evaluate(session_manager=session_manager,
                                                                                     job_id=job_id)
                    if output.workflow_interrupted:
                        job_store.update_status(job_id, "interrupted")
                    else:
                        parent_dir = os.path.dirname(
                            output.workflow_output_file) if output.workflow_output_file else None

                        job_store.update_status(job_id, "success", output_path=str(parent_dir))
                except Exception as e:
                    logger.error("Error in evaluation job %s: %s", job_id, str(e))
                    job_store.update_status(job_id, "failure", error=str(e))

        async def start_evaluation(request: EvaluateRequest, background_tasks: BackgroundTasks, http_request: Request):
            """Handle evaluation requests."""

            async with session_manager.session(request=http_request):

                # if job_id is present and already exists return the job info
                if request.job_id:
                    job = job_store.get_job(request.job_id)
                    if job:
                        return EvaluateResponse(job_id=job.job_id, status=job.status)

                job_id = job_store.create_job(request.config_file, request.job_id, request.expiry_seconds)
                await self.create_cleanup_task(app=app, name="async_evaluation", job_store=job_store)
                background_tasks.add_task(run_evaluation, job_id, request.config_file, request.reps, session_manager)

                return EvaluateResponse(job_id=job_id, status="submitted")

        def translate_job_to_response(job: JobInfo) -> EvaluateStatusResponse:
            """Translate a JobInfo object to an AIQEvaluateStatusResponse."""
            return EvaluateStatusResponse(job_id=job.job_id,
                                          status=job.status,
                                          config_file=str(job.config_file),
                                          error=job.error,
                                          output_path=str(job.output_path),
                                          created_at=job.created_at,
                                          updated_at=job.updated_at,
                                          expires_at=job_store.get_expires_at(job))

        async def get_job_status(job_id: str, http_request: Request) -> EvaluateStatusResponse:
            """Get the status of an evaluation job."""
            logger.info("Getting status for job %s", job_id)

            async with session_manager.session(request=http_request):

                job = job_store.get_job(job_id)
                if not job:
                    logger.warning("Job %s not found", job_id)
                    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
                logger.info("Found job %s with status %s", job_id, job.status)
                return translate_job_to_response(job)

        async def get_last_job_status(http_request: Request) -> EvaluateStatusResponse:
            """Get the status of the last created evaluation job."""
            logger.info("Getting last job status")

            async with session_manager.session(request=http_request):

                job = job_store.get_last_job()
                if not job:
                    logger.warning("No jobs found when requesting last job status")
                    raise HTTPException(status_code=404, detail="No jobs found")
                logger.info("Found last job %s with status %s", job.job_id, job.status)
                return translate_job_to_response(job)

        async def get_jobs(http_request: Request, status: str | None = None) -> list[EvaluateStatusResponse]:
            """Get all jobs, optionally filtered by status."""

            async with session_manager.session(request=http_request):

                if status is None:
                    logger.info("Getting all jobs")
                    jobs = job_store.get_all_jobs()
                else:
                    logger.info("Getting jobs with status %s", status)
                    jobs = job_store.get_jobs_by_status(status)
                logger.info("Found %d jobs", len(jobs))
                return [translate_job_to_response(job) for job in jobs]

        if self.front_end_config.evaluate.path:
            # Add last job endpoint first (most specific)
            app.add_api_route(
                path=f"{self.front_end_config.evaluate.path}/job/last",
                endpoint=get_last_job_status,
                methods=["GET"],
                response_model=EvaluateStatusResponse,
                description="Get the status of the last created evaluation job",
                responses={
                    404: {
                        "description": "No jobs found"
                    }, 500: response_500
                },
            )

            # Add specific job endpoint (least specific)
            app.add_api_route(
                path=f"{self.front_end_config.evaluate.path}/job/{{job_id}}",
                endpoint=get_job_status,
                methods=["GET"],
                response_model=EvaluateStatusResponse,
                description="Get the status of an evaluation job",
                responses={
                    404: {
                        "description": "Job not found"
                    }, 500: response_500
                },
            )

            # Add jobs endpoint with optional status query parameter
            app.add_api_route(
                path=f"{self.front_end_config.evaluate.path}/jobs",
                endpoint=get_jobs,
                methods=["GET"],
                response_model=list[EvaluateStatusResponse],
                description="Get all jobs, optionally filtered by status",
                responses={500: response_500},
            )

            # Add HTTP endpoint for evaluation
            app.add_api_route(
                path=self.front_end_config.evaluate.path,
                endpoint=start_evaluation,
                methods=[self.front_end_config.evaluate.method],
                response_model=EvaluateResponse,
                description=self.front_end_config.evaluate.description,
                responses={500: response_500},
            )

    async def add_static_files_route(self, app: FastAPI, builder: WorkflowBuilder):

        if not self.front_end_config.object_store:
            logger.debug("No object store configured, skipping static files route")
            return

        object_store_client = await builder.get_object_store_client(self.front_end_config.object_store)

        def sanitize_path(path: str) -> str:
            sanitized_path = os.path.normpath(path.strip("/"))
            if sanitized_path == ".":
                raise HTTPException(status_code=400, detail="Invalid file path.")
            filename = os.path.basename(sanitized_path)
            if not filename:
                raise HTTPException(status_code=400, detail="Filename cannot be empty.")
            return sanitized_path

        # Upload static files to the object store; if key is present, it will fail with 409 Conflict
        async def add_static_file(file_path: str, file: UploadFile):
            sanitized_file_path = sanitize_path(file_path)
            file_data = await file.read()

            try:
                await object_store_client.put_object(sanitized_file_path,
                                                     ObjectStoreItem(data=file_data, content_type=file.content_type))
            except KeyAlreadyExistsError as e:
                raise HTTPException(status_code=409, detail=str(e)) from e

            return {"filename": sanitized_file_path}

        # Upsert static files to the object store; if key is present, it will overwrite the file
        async def upsert_static_file(file_path: str, file: UploadFile):
            sanitized_file_path = sanitize_path(file_path)
            file_data = await file.read()

            await object_store_client.upsert_object(sanitized_file_path,
                                                    ObjectStoreItem(data=file_data, content_type=file.content_type))

            return {"filename": sanitized_file_path}

        # Get static files from the object store
        async def get_static_file(file_path: str):

            try:
                file_data = await object_store_client.get_object(file_path)
            except NoSuchKeyError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e

            filename = file_path.split("/")[-1]

            async def reader():
                yield file_data.data

            return StreamingResponse(reader(),
                                     media_type=file_data.content_type,
                                     headers={"Content-Disposition": f"attachment; filename={filename}"})

        async def delete_static_file(file_path: str):
            try:
                await object_store_client.delete_object(file_path)
            except NoSuchKeyError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e

            return Response(status_code=204)

        # Add the static files route to the FastAPI app
        app.add_api_route(
            path="/static/{file_path:path}",
            endpoint=add_static_file,
            methods=["POST"],
            description="Upload a static file to the object store",
        )

        app.add_api_route(
            path="/static/{file_path:path}",
            endpoint=upsert_static_file,
            methods=["PUT"],
            description="Upsert a static file to the object store",
        )

        app.add_api_route(
            path="/static/{file_path:path}",
            endpoint=get_static_file,
            methods=["GET"],
            description="Get a static file from the object store",
        )

        app.add_api_route(
            path="/static/{file_path:path}",
            endpoint=delete_static_file,
            methods=["DELETE"],
            description="Delete a static file from the object store",
        )

    async def add_route(self,
                        app: FastAPI,
                        endpoint: FastApiFrontEndConfig.EndpointBase,
                        session_manager: SessionManager):

        workflow = session_manager.workflow

        GenerateBodyType = workflow.input_schema  # pylint: disable=invalid-name
        GenerateStreamResponseType = workflow.streaming_output_schema  # pylint: disable=invalid-name
        GenerateSingleResponseType = workflow.single_output_schema  # pylint: disable=invalid-name

        # Append job_id and expiry_seconds to the input schema, this effectively makes these reserved keywords
        # Consider prefixing these with "aiq_" to avoid conflicts
        class AsyncGenerateRequest(GenerateBodyType):
            job_id: str | None = Field(default=None, description="Unique identifier for the evaluation job")
            sync_timeout: int = Field(
                default=0,
                ge=0,
                le=300,
                description="Attempt to perform the job synchronously up until `sync_timeout` sectonds, "
                "if the job hasn't been completed by then a job_id will be returned with a status code of 202.")
            expiry_seconds: int = Field(default=JobStore.DEFAULT_EXPIRY,
                                        ge=JobStore.MIN_EXPIRY,
                                        le=JobStore.MAX_EXPIRY,
                                        description="Optional time (in seconds) before the job expires. "
                                        "Clamped between 600 (10 min) and 86400 (24h).")

        # Ensure that the input is in the body. POD types are treated as query parameters
        if (not issubclass(GenerateBodyType, BaseModel)):
            GenerateBodyType = typing.Annotated[GenerateBodyType, Body()]
        else:
            logger.info("Expecting generate request payloads in the following format: %s",
                        GenerateBodyType.model_fields)

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

        # Create job store for tracking async generation jobs
        job_store = JobStore()

        # Run up to max_running_async_jobs jobs at the same time
        async_job_concurrency = asyncio.Semaphore(self._front_end_config.max_running_async_jobs)

        def get_single_endpoint(result_type: type | None):

            async def get_single(response: Response, request: Request):

                response.headers["Content-Type"] = "application/json"

                async with session_manager.session(request=request,
                                                   user_authentication_callback=self._http_flow_handler.authenticate):

                    return await generate_single_response(None, session_manager, result_type=result_type)

            return get_single

        def get_streaming_endpoint(streaming: bool, result_type: type | None, output_type: type | None):

            async def get_stream(request: Request):

                async with session_manager.session(request=request,
                                                   user_authentication_callback=self._http_flow_handler.authenticate):

                    return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                             content=generate_streaming_response_as_str(
                                                 None,
                                                 session_manager=session_manager,
                                                 streaming=streaming,
                                                 step_adaptor=self.get_step_adaptor(),
                                                 result_type=result_type,
                                                 output_type=output_type))

            return get_stream

        def get_streaming_raw_endpoint(streaming: bool, result_type: type | None, output_type: type | None):

            async def get_stream(filter_steps: str | None = None):

                return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                         content=generate_streaming_response_full_as_str(
                                             None,
                                             session_manager=session_manager,
                                             streaming=streaming,
                                             result_type=result_type,
                                             output_type=output_type,
                                             filter_steps=filter_steps))

            return get_stream

        def post_single_endpoint(request_type: type, result_type: type | None):

            async def post_single(response: Response, request: Request, payload: request_type):

                response.headers["Content-Type"] = "application/json"

                async with session_manager.session(request=request,
                                                   user_authentication_callback=self._http_flow_handler.authenticate):

                    return await generate_single_response(payload, session_manager, result_type=result_type)

            return post_single

        def post_streaming_endpoint(request_type: type,
                                    streaming: bool,
                                    result_type: type | None,
                                    output_type: type | None):

            async def post_stream(request: Request, payload: request_type):

                async with session_manager.session(request=request,
                                                   user_authentication_callback=self._http_flow_handler.authenticate):

                    return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                             content=generate_streaming_response_as_str(
                                                 payload,
                                                 session_manager=session_manager,
                                                 streaming=streaming,
                                                 step_adaptor=self.get_step_adaptor(),
                                                 result_type=result_type,
                                                 output_type=output_type))

            return post_stream

        def post_streaming_raw_endpoint(request_type: type,
                                        streaming: bool,
                                        result_type: type | None,
                                        output_type: type | None):
            """
            Stream raw intermediate steps without any step adaptor translations.
            """

            async def post_stream(payload: request_type, filter_steps: str | None = None):

                return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                         content=generate_streaming_response_full_as_str(
                                             payload,
                                             session_manager=session_manager,
                                             streaming=streaming,
                                             result_type=result_type,
                                             output_type=output_type,
                                             filter_steps=filter_steps))

            return post_stream

        def post_openai_api_compatible_endpoint(request_type: type):
            """
            OpenAI-compatible endpoint that handles both streaming and non-streaming
            based on the 'stream' parameter in the request.
            """

            async def post_openai_api_compatible(response: Response, request: Request, payload: request_type):
                # Check if streaming is requested
                stream_requested = getattr(payload, 'stream', False)

                async with session_manager.session(request=request):
                    if stream_requested:
                        # Return streaming response
                        return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                                 content=generate_streaming_response_as_str(
                                                     payload,
                                                     session_manager=session_manager,
                                                     streaming=True,
                                                     step_adaptor=self.get_step_adaptor(),
                                                     result_type=ChatResponseChunk,
                                                     output_type=ChatResponseChunk))
                    else:
                        # Return single response - check if workflow supports non-streaming
                        try:
                            response.headers["Content-Type"] = "application/json"
                            return await generate_single_response(payload, session_manager, result_type=ChatResponse)
                        except ValueError as e:
                            if "Cannot get a single output value for streaming workflows" in str(e):
                                # Workflow only supports streaming, but client requested non-streaming
                                # Fall back to streaming and collect the result
                                chunks = []
                                async for chunk_str in generate_streaming_response_as_str(
                                        payload,
                                        session_manager=session_manager,
                                        streaming=True,
                                        step_adaptor=self.get_step_adaptor(),
                                        result_type=ChatResponseChunk,
                                        output_type=ChatResponseChunk):
                                    if chunk_str.startswith("data: ") and not chunk_str.startswith("data: [DONE]"):
                                        chunk_data = chunk_str[6:].strip()  # Remove "data: " prefix
                                        if chunk_data:
                                            try:
                                                chunk_json = ChatResponseChunk.model_validate_json(chunk_data)
                                                if (chunk_json.choices and len(chunk_json.choices) > 0
                                                        and chunk_json.choices[0].delta
                                                        and chunk_json.choices[0].delta.content is not None):
                                                    chunks.append(chunk_json.choices[0].delta.content)
                                            except Exception:
                                                continue

                                # Create a single response from collected chunks
                                content = "".join(chunks)
                                single_response = ChatResponse.from_string(content)
                                response.headers["Content-Type"] = "application/json"
                                return single_response
                            else:
                                raise

            return post_openai_api_compatible

        async def run_generation(job_id: str, payload: typing.Any, session_manager: SessionManager, result_type: type):
            """Background task to run the evaluation."""
            async with async_job_concurrency:
                try:
                    result = await generate_single_response(payload=payload,
                                                            session_manager=session_manager,
                                                            result_type=result_type)
                    job_store.update_status(job_id, "success", output=result)
                except Exception as e:
                    logger.error("Error in evaluation job %s: %s", job_id, e)
                    job_store.update_status(job_id, "failure", error=str(e))

        def _job_status_to_response(job: JobInfo) -> AsyncGenerationStatusResponse:
            job_output = job.output
            if job_output is not None:
                job_output = job_output.model_dump()
            return AsyncGenerationStatusResponse(job_id=job.job_id,
                                                 status=job.status,
                                                 error=job.error,
                                                 output=job_output,
                                                 created_at=job.created_at,
                                                 updated_at=job.updated_at,
                                                 expires_at=job_store.get_expires_at(job))

        def post_async_generation(request_type: type, final_result_type: type):

            async def start_async_generation(
                    request: request_type, background_tasks: BackgroundTasks, response: Response,
                    http_request: Request) -> AsyncGenerateResponse | AsyncGenerationStatusResponse:
                """Handle async generation requests."""

                async with session_manager.session(request=http_request):

                    # if job_id is present and already exists return the job info
                    if request.job_id:
                        job = job_store.get_job(request.job_id)
                        if job:
                            return AsyncGenerateResponse(job_id=job.job_id, status=job.status)

                    job_id = job_store.create_job(job_id=request.job_id, expiry_seconds=request.expiry_seconds)
                    await self.create_cleanup_task(app=app, name="async_generation", job_store=job_store)

                    # The fastapi/starlette background tasks won't begin executing until after the response is sent
                    # to the client, so we need to wrap the task in a function, alowing us to start the task now,
                    # and allowing the background task function to await the results.
                    task = asyncio.create_task(
                        run_generation(job_id=job_id,
                                       payload=request,
                                       session_manager=session_manager,
                                       result_type=final_result_type))

                    async def wrapped_task(t: asyncio.Task):
                        return await t

                    background_tasks.add_task(wrapped_task, task)

                    now = time.time()
                    sync_timeout = now + request.sync_timeout
                    while time.time() < sync_timeout:
                        job = job_store.get_job(job_id)
                        if job is not None and job.status not in job_store.ACTIVE_STATUS:
                            # If the job is done, return the result
                            response.status_code = 200
                            return _job_status_to_response(job)

                        # Sleep for a short time before checking again
                        await asyncio.sleep(0.1)

                    response.status_code = 202
                    return AsyncGenerateResponse(job_id=job_id, status="submitted")

            return start_async_generation

        async def get_async_job_status(job_id: str, http_request: Request) -> AsyncGenerationStatusResponse:
            """Get the status of an async job."""
            logger.info("Getting status for job %s", job_id)

            async with session_manager.session(request=http_request):

                job = job_store.get_job(job_id)
                if not job:
                    logger.warning("Job %s not found", job_id)
                    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

                logger.info("Found job %s with status %s", job_id, job.status)
                return _job_status_to_response(job)

        async def websocket_endpoint(websocket: WebSocket):

            # Universal cookie handling: works for both cross-origin and same-origin connections
            session_id = websocket.query_params.get("session")
            if session_id:
                headers = list(websocket.scope.get("headers", []))
                cookie_header = f"aiqtoolkit-session={session_id}"

                # Check if the session cookie already exists to avoid duplicates
                cookie_exists = False
                existing_session_cookie = False

                for i, (name, value) in enumerate(headers):
                    if name == b"cookie":
                        cookie_exists = True
                        cookie_str = value.decode()

                        # Check if aiqtoolkit-session already exists in cookies
                        if "aiqtoolkit-session=" in cookie_str:
                            existing_session_cookie = True
                            logger.info("WebSocket: Session cookie already present in headers (same-origin)")
                        else:
                            # Append to existing cookie header (cross-origin case)
                            headers[i] = (name, f"{cookie_str}; {cookie_header}".encode())
                            logger.info("WebSocket: Added session cookie to existing cookie header: %s",
                                        session_id[:10] + "...")
                        break

                # Add new cookie header only if no cookies exist and no session cookie found
                if not cookie_exists and not existing_session_cookie:
                    headers.append((b"cookie", cookie_header.encode()))
                    logger.info("WebSocket: Added new session cookie header: %s", session_id[:10] + "...")

                # Update the websocket scope with the modified headers
                websocket.scope["headers"] = headers

            async with WebSocketMessageHandler(websocket, session_manager, self.get_step_adaptor()) as handler:

                flow_handler = WebSocketAuthenticationFlowHandler(self._add_flow, self._remove_flow, handler)

                # Ugly hack to set the flow handler on the message handler. Both need eachother to be set.
                handler.set_flow_handler(flow_handler)

                await handler.run()

        if (endpoint.websocket_path):
            app.add_websocket_route(endpoint.websocket_path, websocket_endpoint)

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

                app.add_api_route(
                    path=f"{endpoint.path}/full",
                    endpoint=get_streaming_raw_endpoint(streaming=True,
                                                        result_type=GenerateStreamResponseType,
                                                        output_type=GenerateStreamResponseType),
                    methods=[endpoint.method],
                    description="Stream raw intermediate steps without any step adaptor translations.\n"
                    "Use filter_steps query parameter to filter steps by type (comma-separated list) or\
                        set to 'none' to suppress all intermediate steps.",
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

                app.add_api_route(
                    path=f"{endpoint.path}/full",
                    endpoint=post_streaming_raw_endpoint(request_type=GenerateBodyType,
                                                         streaming=True,
                                                         result_type=GenerateStreamResponseType,
                                                         output_type=GenerateStreamResponseType),
                    methods=[endpoint.method],
                    response_model=GenerateStreamResponseType,
                    description="Stream raw intermediate steps without any step adaptor translations.\n"
                    "Use filter_steps query parameter to filter steps by type (comma-separated list) or \
                        set to 'none' to suppress all intermediate steps.",
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.path}/async",
                    endpoint=post_async_generation(request_type=AsyncGenerateRequest,
                                                   final_result_type=GenerateSingleResponseType),
                    methods=[endpoint.method],
                    response_model=AsyncGenerateResponse | AsyncGenerationStatusResponse,
                    description="Start an async generate job",
                    responses={500: response_500},
                )
            else:
                raise ValueError(f"Unsupported method {endpoint.method}")

            app.add_api_route(
                path=f"{endpoint.path}/async/job/{{job_id}}",
                endpoint=get_async_job_status,
                methods=["GET"],
                response_model=AsyncGenerationStatusResponse,
                description="Get the status of an async job",
                responses={
                    404: {
                        "description": "Job not found"
                    }, 500: response_500
                },
            )

        if (endpoint.openai_api_path):
            if (endpoint.method == "GET"):

                app.add_api_route(
                    path=endpoint.openai_api_path,
                    endpoint=get_single_endpoint(result_type=ChatResponse),
                    methods=[endpoint.method],
                    response_model=ChatResponse,
                    description=endpoint.description,
                    responses={500: response_500},
                )

                app.add_api_route(
                    path=f"{endpoint.openai_api_path}/stream",
                    endpoint=get_streaming_endpoint(streaming=True,
                                                    result_type=ChatResponseChunk,
                                                    output_type=ChatResponseChunk),
                    methods=[endpoint.method],
                    response_model=ChatResponseChunk,
                    description=endpoint.description,
                    responses={500: response_500},
                )

            elif (endpoint.method == "POST"):

                # Check if OpenAI v1 compatible endpoint is configured
                openai_v1_path = getattr(endpoint, 'openai_api_v1_path', None)

                # Always create legacy endpoints for backward compatibility (unless they conflict with v1 path)
                if not openai_v1_path or openai_v1_path != endpoint.openai_api_path:
                    # <openai_api_path> = non-streaming (legacy behavior)
                    app.add_api_route(
                        path=endpoint.openai_api_path,
                        endpoint=post_single_endpoint(request_type=ChatRequest, result_type=ChatResponse),
                        methods=[endpoint.method],
                        response_model=ChatResponse,
                        description=endpoint.description,
                        responses={500: response_500},
                    )

                    # <openai_api_path>/stream = streaming (legacy behavior)
                    app.add_api_route(
                        path=f"{endpoint.openai_api_path}/stream",
                        endpoint=post_streaming_endpoint(request_type=ChatRequest,
                                                         streaming=True,
                                                         result_type=ChatResponseChunk,
                                                         output_type=ChatResponseChunk),
                        methods=[endpoint.method],
                        response_model=ChatResponseChunk | ResponseIntermediateStep,
                        description=endpoint.description,
                        responses={500: response_500},
                    )

                # Create OpenAI v1 compatible endpoint if configured
                if openai_v1_path:
                    # OpenAI v1 Compatible Mode: Create single endpoint that handles both streaming and non-streaming
                    app.add_api_route(
                        path=openai_v1_path,
                        endpoint=post_openai_api_compatible_endpoint(request_type=ChatRequest),
                        methods=[endpoint.method],
                        response_model=ChatResponse | ChatResponseChunk,
                        description=f"{endpoint.description} (OpenAI Chat Completions API compatible)",
                        responses={500: response_500},
                    )

            else:
                raise ValueError(f"Unsupported method {endpoint.method}")

    async def add_authorization_route(self, app: FastAPI):

        from fastapi.responses import HTMLResponse

        from aiq.front_ends.fastapi.html_snippets.auth_code_grant_success import AUTH_REDIRECT_SUCCESS_HTML

        async def redirect_uri(request: Request):
            """
            Handle the redirect URI for OAuth2 authentication.
            Args:
                request: The FastAPI request object containing query parameters.

            Returns:
                HTMLResponse: A response indicating the success of the authentication flow.
            """
            state = request.query_params.get("state")

            async with self._outstanding_flows_lock:
                if not state or state not in self._outstanding_flows:
                    return "Invalid state. Please restart the authentication process."

                flow_state = self._outstanding_flows[state]

            config = flow_state.config
            verifier = flow_state.verifier
            client = flow_state.client

            try:
                res = await client.fetch_token(url=config.token_url,
                                               authorization_response=str(request.url),
                                               code_verifier=verifier,
                                               state=state)
                flow_state.future.set_result(res)
            except Exception as e:
                flow_state.future.set_exception(e)

            return HTMLResponse(content=AUTH_REDIRECT_SUCCESS_HTML,
                                status_code=200,
                                headers={
                                    "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"
                                })

        if (self.front_end_config.oauth2_callback_path):
            # Add the redirect URI route
            app.add_api_route(
                path=self.front_end_config.oauth2_callback_path,
                endpoint=redirect_uri,
                methods=["GET"],
                description="Handles the authorization code and state returned from the Authorization Code Grant Flow.")

    async def _add_flow(self, state: str, flow_state: FlowState):
        async with self._outstanding_flows_lock:
            self._outstanding_flows[state] = flow_state

    async def _remove_flow(self, state: str):
        async with self._outstanding_flows_lock:
            del self._outstanding_flows[state]
