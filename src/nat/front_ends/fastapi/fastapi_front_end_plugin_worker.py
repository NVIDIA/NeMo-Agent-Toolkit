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
import json
import logging
import os
import typing
from abc import ABC
from abc import abstractmethod
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from authlib.common.errors import AuthlibBaseError as OAuthError
from fastapi import Body
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import Field
from starlette.websockets import WebSocket

from nat.builder.eval_builder import WorkflowEvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ResponseIntermediateStep
from nat.data_models.config import Config
from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.eval.config import EvaluationRunOutput
from nat.eval.evaluate import EvaluationRun
from nat.eval.evaluate import EvaluationRunConfig
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.front_ends.fastapi.auth_flow_handlers.http_flow_handler import HTTPAuthenticationFlowHandler
from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import FlowState
from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler
from nat.front_ends.fastapi.fastapi_front_end_config import AsyncGenerateResponse
from nat.front_ends.fastapi.fastapi_front_end_config import AsyncGenerationStatusResponse
from nat.front_ends.fastapi.fastapi_front_end_config import EvaluateItemRequest
from nat.front_ends.fastapi.fastapi_front_end_config import EvaluateItemResponse
from nat.front_ends.fastapi.fastapi_front_end_config import EvaluateRequest
from nat.front_ends.fastapi.fastapi_front_end_config import EvaluateResponse
from nat.front_ends.fastapi.fastapi_front_end_config import EvaluateStatusResponse
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.message_handler import WebSocketMessageHandler
from nat.front_ends.fastapi.response_helpers import generate_single_response
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_as_str
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_full_as_str
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.front_ends.fastapi.utils import get_config_file_path
from nat.object_store.models import ObjectStoreItem
from nat.runtime.loader import load_workflow
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)

_DASK_AVAILABLE = False

try:
    from nat.front_ends.fastapi.job_store import JobInfo
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore
    _DASK_AVAILABLE = True
except ImportError:
    JobInfo = None
    JobStatus = None
    JobStore = None


class FastApiFrontEndPluginWorkerBase(ABC):

    def __init__(self, config: Config):
        self._config = config

        assert isinstance(config.general.front_end,
                          FastApiFrontEndConfig), ("Front end config is not FastApiFrontEndConfig")

        self._front_end_config = config.general.front_end
        self._dask_available = False
        self._job_store = None
        self._http_flow_handler: HTTPAuthenticationFlowHandler | None = HTTPAuthenticationFlowHandler()
        self._scheduler_address = os.environ.get("NAT_DASK_SCHEDULER_ADDRESS")
        self._db_url = os.environ.get("NAT_JOB_STORE_DB_URL")
        self._config_file_path = get_config_file_path()

        if self._scheduler_address is not None:
            if not _DASK_AVAILABLE:
                raise RuntimeError("Dask is not available, please install it to use the FastAPI front end with Dask.")

            if self._db_url is None:
                raise RuntimeError(
                    "NAT_JOB_STORE_DB_URL must be set when using Dask (configure a persistent JobStore database).")

            try:
                self._job_store = JobStore(scheduler_address=self._scheduler_address, db_url=self._db_url)
                self._dask_available = True
                logger.debug("Connected to Dask scheduler at %s", self._scheduler_address)
            except Exception as e:
                raise RuntimeError(f"Failed to connect to Dask scheduler at {self._scheduler_address}: {e}") from e
        else:
            logger.debug("No Dask scheduler address provided, running without Dask support.")

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

            logger.debug("Starting NAT server from process %s", os.getpid())

            async with WorkflowBuilder.from_config(self.config) as builder:

                await self.configure(starting_app, builder)

                yield

            logger.debug("Closing NAT server from process %s", os.getpid())

        nat_app = FastAPI(lifespan=lifespan)

        # Configure app CORS.
        self.set_cors_config(nat_app)

        @nat_app.middleware("http")
        async def authentication_log_filter(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
            return await self._suppress_authentication_logs(request, call_next)

        return nat_app

    def set_cors_config(self, nat_app: FastAPI) -> None:
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

        nat_app.add_middleware(
            CORSMiddleware,
            **cors_kwargs,
        )

    async def _suppress_authentication_logs(self, request: Request,
                                            call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """
        Intercepts authentication request and supreses logs that contain sensitive data.
        """
        from nat.utils.log_utils import LogFilter

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

        # Evaluator storage for single-item evaluation
        self._evaluators: dict[str, EvaluatorInfo] = {}
        self._eval_builder: WorkflowEvalBuilder | None = None

    async def initialize_evaluators(self, config: Config):
        """Initialize and store evaluators from config for single-item evaluation."""
        if not config.eval or not config.eval.evaluators:
            logger.info("No evaluators configured, skipping evaluator initialization")
            return

        try:
            # Build evaluators using WorkflowEvalBuilder (same pattern as nat eval)
            # Start with registry=None and let populate_builder set everything up
            self._eval_builder = WorkflowEvalBuilder(general_config=config.general,
                                                     eval_general_config=config.eval.general,
                                                     registry=None)

            # Enter the async context and keep it alive
            await self._eval_builder.__aenter__()

            # Populate builder with config (this sets up LLMs, functions, etc.)
            # Skip workflow build since we already have it from the main builder
            await self._eval_builder.populate_builder(config, skip_workflow=True)

            # Now evaluators should be populated by populate_builder
            for name in config.eval.evaluators.keys():
                self._evaluators[name] = self._eval_builder.get_evaluator(name)
                logger.info(f"Initialized evaluator: {name}")

            logger.info(f"Successfully initialized {len(self._evaluators)} evaluators")

        except Exception as e:
            logger.error(f"Failed to initialize evaluators: {e}")
            # Don't fail startup, just log the error
            self._evaluators = {}

    async def cleanup_evaluators(self):
        """Clean up evaluator resources on shutdown."""
        if self._eval_builder:
            try:
                await self._eval_builder.__aexit__(None, None, None)
                logger.info("Evaluator builder context cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up evaluator builder: {e}")
            finally:
                self._eval_builder = None
                self._evaluators.clear()

    def get_step_adaptor(self) -> StepAdaptor:

        return StepAdaptor(self.front_end_config.step_adaptor)

    async def configure(self, app: FastAPI, builder: WorkflowBuilder):

        # Do things like setting the base URL and global configuration options
        app.root_path = self.front_end_config.root_path

        # Initialize evaluators for single-item evaluation
        # TODO: we need config control over this as it's not always needed
        await self.initialize_evaluators(self._config)

        # Ensure evaluator resources are cleaned up when the app shuts down
        app.add_event_handler("shutdown", self.cleanup_evaluators)

        await self.add_routes(app, builder)

    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder):

        await self.add_default_route(app, SessionManager(await builder.build()))
        await self.add_evaluate_route(app, SessionManager(await builder.build()))
        await self.add_evaluate_item_route(app, SessionManager(await builder.build()))
        await self.add_static_files_route(app, builder)
        await self.add_authorization_route(app)
        await self.add_mcp_client_tool_list_route(app, builder)

        for ep in self.front_end_config.endpoints:

            entry_workflow = await builder.build(entry_function=ep.function_name)

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

        # TODO: Find another way to limit the number of concurrent evaluations
        async def run_evaluation(scheduler_address: str,
                                 db_url: str,
                                 workflow_config_file_path: str,
                                 job_id: str,
                                 eval_config_file: str,
                                 reps: int):
            """Background task to run the evaluation."""
            job_store = JobStore(scheduler_address=scheduler_address, db_url=db_url)

            try:
                # We have two config files, one for the workflow and one for the evaluation
                # Create EvaluationRunConfig using the CLI defaults
                eval_config = EvaluationRunConfig(config_file=Path(eval_config_file), dataset=None, reps=reps)

                # Create a new EvaluationRun with the evaluation-specific config
                await job_store.update_status(job_id, JobStatus.RUNNING)
                eval_runner = EvaluationRun(eval_config)

                async with load_workflow(workflow_config_file_path) as local_session_manager:
                    output: EvaluationRunOutput = await eval_runner.run_and_evaluate(
                        session_manager=local_session_manager, job_id=job_id)

                if output.workflow_interrupted:
                    await job_store.update_status(job_id, JobStatus.INTERRUPTED)
                else:
                    parent_dir = os.path.dirname(output.workflow_output_file) if output.workflow_output_file else None

                    await job_store.update_status(job_id, JobStatus.SUCCESS, output_path=str(parent_dir))
            except Exception as e:
                logger.exception("Error in evaluation job %s", job_id)
                await job_store.update_status(job_id, JobStatus.FAILURE, error=str(e))

        async def start_evaluation(request: EvaluateRequest, http_request: Request):
            """Handle evaluation requests."""

            async with session_manager.session(http_connection=http_request):

                # if job_id is present and already exists return the job info
                # There is a race condition between this check and the actual job submission, however if the client is
                # supplying their own job_ids, then it is their responsibility to ensure that the job_id is unique.
                if request.job_id:
                    job_status = await self._job_store.get_status(request.job_id)
                    if job_status != JobStatus.NOT_FOUND:
                        return EvaluateResponse(job_id=request.job_id, status=job_status)

                job_id = self._job_store.ensure_job_id(request.job_id)

                await self._job_store.submit_job(job_id=job_id,
                                                 config_file=request.config_file,
                                                 expiry_seconds=request.expiry_seconds,
                                                 job_fn=run_evaluation,
                                                 job_args=[
                                                     self._scheduler_address,
                                                     self._db_url,
                                                     self._config_file_path,
                                                     job_id,
                                                     request.config_file,
                                                     request.reps
                                                 ])

                logger.info("Submitted evaluation job %s with config %s", job_id, request.config_file)

                return EvaluateResponse(job_id=job_id, status=JobStatus.SUBMITTED)

        def translate_job_to_response(job: "JobInfo") -> EvaluateStatusResponse:
            """Translate a JobInfo object to an EvaluateStatusResponse."""
            return EvaluateStatusResponse(job_id=job.job_id,
                                          status=job.status,
                                          config_file=str(job.config_file),
                                          error=job.error,
                                          output_path=str(job.output_path),
                                          created_at=job.created_at,
                                          updated_at=job.updated_at,
                                          expires_at=self._job_store.get_expires_at(job))

        async def get_job_status(job_id: str, http_request: Request) -> EvaluateStatusResponse:
            """Get the status of an evaluation job."""
            logger.info("Getting status for job %s", job_id)

            async with session_manager.session(http_connection=http_request):

                job = await self._job_store.get_job(job_id)
                if not job:
                    logger.warning("Job %s not found", job_id)
                    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
                logger.info("Found job %s with status %s", job_id, job.status)
                return translate_job_to_response(job)

        async def get_last_job_status(http_request: Request) -> EvaluateStatusResponse:
            """Get the status of the last created evaluation job."""
            logger.info("Getting last job status")

            async with session_manager.session(http_connection=http_request):

                job = await self._job_store.get_last_job()
                if not job:
                    logger.warning("No jobs found when requesting last job status")
                    raise HTTPException(status_code=404, detail="No jobs found")
                logger.info("Found last job %s with status %s", job.job_id, job.status)
                return translate_job_to_response(job)

        async def get_jobs(http_request: Request, status: str | None = None) -> list[EvaluateStatusResponse]:
            """Get all jobs, optionally filtered by status."""

            async with session_manager.session(http_connection=http_request):

                if status is None:
                    logger.info("Getting all jobs")
                    jobs = await self._job_store.get_all_jobs()
                else:
                    logger.info("Getting jobs with status %s", status)
                    jobs = await self._job_store.get_jobs_by_status(JobStatus(status))

                logger.info("Found %d jobs", len(jobs))
                return [translate_job_to_response(job) for job in jobs]

        if self.front_end_config.evaluate.path:
            if self._dask_available:
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
            else:
                logger.warning("Dask is not available, evaluation endpoints will not be added.")

    async def add_evaluate_item_route(self, app: FastAPI, session_manager: SessionManager):
        """Add the single-item evaluation endpoint to the FastAPI app."""

        async def evaluate_single_item(request: EvaluateItemRequest, http_request: Request) -> EvaluateItemResponse:
            """Handle single-item evaluation requests."""

            async with session_manager.session(http_connection=http_request):

                # Check if evaluator exists
                if request.evaluator_name not in self._evaluators:
                    raise HTTPException(status_code=404,
                                        detail=f"Evaluator '{request.evaluator_name}' not found. "
                                        f"Available evaluators: {list(self._evaluators.keys())}")

                try:
                    # Get the evaluator
                    evaluator = self._evaluators[request.evaluator_name]

                    # Run evaluation on single item
                    result = await evaluator.evaluate_fn(EvalInput(eval_input_items=[request.item]))

                    # Extract the single output item
                    if result.eval_output_items:
                        output_item = result.eval_output_items[0]
                        return EvaluateItemResponse(success=True, result=output_item, error=None)
                    else:
                        return EvaluateItemResponse(success=False, result=None, error="Evaluator returned no results")

                except Exception as e:
                    logger.exception(f"Error evaluating item with {request.evaluator_name}")
                    return EvaluateItemResponse(success=False, result=None, error=f"Evaluation failed: {str(e)}")

        # Register the route
        if self.front_end_config.evaluate_item.path:
            app.add_api_route(path=self.front_end_config.evaluate_item.path,
                              endpoint=evaluate_single_item,
                              methods=[self.front_end_config.evaluate_item.method],
                              response_model=EvaluateItemResponse,
                              description=self.front_end_config.evaluate_item.description,
                              responses={
                                  404: {
                                      "description": "Evaluator not found",
                                      "content": {
                                          "application/json": {
                                              "example": {
                                                  "detail": "Evaluator 'unknown' not found"
                                              }
                                          }
                                      }
                                  },
                                  500: {
                                      "description": "Internal Server Error",
                                      "content": {
                                          "application/json": {
                                              "example": {
                                                  "detail": "Internal server error occurred"
                                              }
                                          }
                                      }
                                  }
                              })
            logger.info(f"Added evaluate_item route at {self.front_end_config.evaluate_item.path}")

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

        GenerateBodyType = workflow.input_schema
        GenerateStreamResponseType = workflow.streaming_output_schema
        GenerateSingleResponseType = workflow.single_output_schema

        # Skip async generation for custom routes (those with function_name)
        if self._dask_available and not hasattr(endpoint, 'function_name'):
            # Append job_id and expiry_seconds to the input schema, this effectively makes these reserved keywords
            # Consider prefixing these with "nat_" to avoid conflicts

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

                def validate_model(self):
                    # Override to ensure that the parent class validator is not called
                    return self

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

        def get_single_endpoint(result_type: type | None):

            async def get_single(response: Response, request: Request):

                response.headers["Content-Type"] = "application/json"

                async with session_manager.session(http_connection=request,
                                                   user_authentication_callback=self._http_flow_handler.authenticate):

                    return await generate_single_response(None, session_manager, result_type=result_type)

            return get_single

        def get_streaming_endpoint(streaming: bool, result_type: type | None, output_type: type | None):

            async def get_stream(request: Request):

                async with session_manager.session(http_connection=request,
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

                async with session_manager.session(http_connection=request,
                                                   user_authentication_callback=self._http_flow_handler.authenticate):

                    return await generate_single_response(payload, session_manager, result_type=result_type)

            return post_single

        def post_streaming_endpoint(request_type: type,
                                    streaming: bool,
                                    result_type: type | None,
                                    output_type: type | None):

            async def post_stream(request: Request, payload: request_type):

                async with session_manager.session(http_connection=request,
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

                response.headers["Content-Type"] = "application/json"
                stream_requested = getattr(payload, 'stream', False)

                async with session_manager.session(http_connection=request):
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

                    return await generate_single_response(payload, session_manager, result_type=ChatResponse)

            return post_openai_api_compatible

        def _job_status_to_response(job: "JobInfo") -> AsyncGenerationStatusResponse:
            job_output = job.output
            if job_output is not None:
                try:
                    job_output = json.loads(job_output)
                except json.JSONDecodeError:
                    logger.error("Failed to parse job output as JSON: %s", job_output)
                    job_output = {"error": "Output parsing failed"}

            return AsyncGenerationStatusResponse(job_id=job.job_id,
                                                 status=job.status,
                                                 error=job.error,
                                                 output=job_output,
                                                 created_at=job.created_at,
                                                 updated_at=job.updated_at,
                                                 expires_at=self._job_store.get_expires_at(job))

        async def run_generation(scheduler_address: str,
                                 db_url: str,
                                 config_file_path: str,
                                 job_id: str,
                                 payload: typing.Any):
            """Background task to run the workflow."""
            job_store = JobStore(scheduler_address=scheduler_address, db_url=db_url)
            try:
                async with load_workflow(config_file_path) as local_session_manager:
                    result = await generate_single_response(
                        payload, local_session_manager, result_type=local_session_manager.workflow.single_output_schema)

                await job_store.update_status(job_id, JobStatus.SUCCESS, output=result)
            except Exception as e:
                logger.exception("Error in async job %s", job_id)
                await job_store.update_status(job_id, JobStatus.FAILURE, error=str(e))

        def post_async_generation(request_type: type):

            async def start_async_generation(
                    request: request_type, response: Response,
                    http_request: Request) -> AsyncGenerateResponse | AsyncGenerationStatusResponse:
                """Handle async generation requests."""

                async with session_manager.session(http_connection=http_request):

                    # if job_id is present and already exists return the job info
                    if request.job_id:
                        job = await self._job_store.get_job(request.job_id)
                        if job:
                            return AsyncGenerateResponse(job_id=job.job_id, status=job.status)

                    job_id = self._job_store.ensure_job_id(request.job_id)
                    (_, job) = await self._job_store.submit_job(
                        job_id=job_id,
                        expiry_seconds=request.expiry_seconds,
                        job_fn=run_generation,
                        sync_timeout=request.sync_timeout,
                        job_args=[
                            self._scheduler_address,
                            self._db_url,
                            self._config_file_path,
                            job_id,
                            request.model_dump(mode="json", exclude=["job_id", "sync_timeout", "expiry_seconds"])
                        ])

                    if job is not None:
                        response.status_code = 200
                        return _job_status_to_response(job)

                    response.status_code = 202
                    return AsyncGenerateResponse(job_id=job_id, status=JobStatus.SUBMITTED)

            return start_async_generation

        async def get_async_job_status(job_id: str, http_request: Request) -> AsyncGenerationStatusResponse:
            """Get the status of an async job."""
            logger.info("Getting status for job %s", job_id)

            async with session_manager.session(http_connection=http_request):

                job = await self._job_store.get_job(job_id)
                if job is None:
                    logger.warning("Job %s not found", job_id)
                    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

                logger.info("Found job %s with status %s", job_id, job.status)
                return _job_status_to_response(job)

        async def websocket_endpoint(websocket: WebSocket):

            # Universal cookie handling: works for both cross-origin and same-origin connections
            session_id = websocket.query_params.get("session")
            if session_id:
                headers = list(websocket.scope.get("headers", []))
                cookie_header = f"nat-session={session_id}"

                # Check if the session cookie already exists to avoid duplicates
                cookie_exists = False
                existing_session_cookie = False

                for i, (name, value) in enumerate(headers):
                    if name == b"cookie":
                        cookie_exists = True
                        cookie_str = value.decode()

                        # Check if nat-session already exists in cookies
                        if "nat-session=" in cookie_str:
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

                if self._dask_available and not hasattr(endpoint, 'function_name'):
                    app.add_api_route(
                        path=f"{endpoint.path}/async",
                        endpoint=post_async_generation(request_type=AsyncGenerateRequest),
                        methods=[endpoint.method],
                        response_model=AsyncGenerateResponse | AsyncGenerationStatusResponse,
                        description="Start an async generate job",
                        responses={500: response_500},
                    )
                else:
                    logger.warning("Dask is not available, async generation endpoints will not be added.")
            else:
                raise ValueError(f"Unsupported method {endpoint.method}")

            if self._dask_available and not hasattr(endpoint, 'function_name'):
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

        from nat.front_ends.fastapi.html_snippets.auth_code_grant_success import AUTH_REDIRECT_SUCCESS_HTML

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
            except OAuthError as e:
                flow_state.future.set_exception(
                    RuntimeError(f"Authorization server rejected request: {e.error} ({e.description})"))
            except httpx.HTTPError as e:
                flow_state.future.set_exception(RuntimeError(f"Network error during token fetch: {e}"))
            except Exception as e:
                flow_state.future.set_exception(RuntimeError(f"Authentication failed: {e}"))

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

    async def add_mcp_client_tool_list_route(self, app: FastAPI, builder: WorkflowBuilder):
        """Add the MCP client tool list endpoint to the FastAPI app."""
        from typing import Any

        from pydantic import BaseModel

        class MCPToolInfo(BaseModel):
            name: str
            description: str
            server: str
            available: bool

        class MCPClientToolListResponse(BaseModel):
            mcp_clients: list[dict[str, Any]]

        async def get_mcp_client_tool_list() -> MCPClientToolListResponse:
            """
            Get the list of MCP tools from all MCP clients in the workflow configuration.
            Checks session health and compares with workflow function group configuration.
            """
            mcp_clients_info = []

            try:
                # Get all function groups from the builder
                function_groups = builder._function_groups

                # Find MCP client function groups
                for group_name, configured_group in function_groups.items():
                    if configured_group.config.type != "mcp_client":
                        continue

                    from nat.plugins.mcp.client_config import MCPClientConfig

                    config = configured_group.config
                    assert isinstance(config, MCPClientConfig)

                    # Reuse the existing MCP client session stored on the function group instance
                    group_instance = configured_group.instance

                    client = group_instance.mcp_client
                    if client is None:
                        raise RuntimeError(f"MCP client not found for group {group_name}")

                    try:
                        session_healthy = False
                        server_tools: dict[str, Any] = {}

                        try:
                            server_tools = await client.get_tools()
                            session_healthy = True
                        except Exception as e:
                            logger.exception(f"Failed to connect to MCP server {client.server_name}: {e}")
                            session_healthy = False

                        # Get workflow function group configuration (configured client-side tools)
                        configured_short_names: set[str] = set()
                        configured_full_to_fn: dict[str, Function] = {}
                        try:
                            # Pass a no-op filter function to bypass any default filtering that might check
                            # health status, preventing potential infinite recursion during health status checks.
                            async def pass_through_filter(fn):
                                return fn

                            accessible_functions = await group_instance.get_accessible_functions(
                                filter_fn=pass_through_filter)
                            configured_full_to_fn = accessible_functions
                            configured_short_names = {name.split('.', 1)[1] for name in accessible_functions.keys()}
                        except Exception as e:
                            logger.exception(f"Failed to get accessible functions for group {group_name}: {e}")

                        # Build alias->original mapping and override configs from overrides
                        alias_to_original: dict[str, str] = {}
                        override_configs: dict[str, Any] = {}
                        try:
                            if config.tool_overrides is not None:
                                for orig_name, override in config.tool_overrides.items():
                                    if override.alias is not None:
                                        alias_to_original[override.alias] = orig_name
                                        override_configs[override.alias] = override
                                    else:
                                        override_configs[orig_name] = override
                        except Exception:
                            pass

                        # Create tool info list (always return configured tools; mark availability)
                        tools_info: list[dict[str, Any]] = []
                        available_count = 0
                        for wf_fn, fn_short in zip(configured_full_to_fn.values(), configured_short_names):
                            orig_name = alias_to_original.get(fn_short, fn_short)
                            available = session_healthy and (orig_name in server_tools)
                            if available:
                                available_count += 1

                            # Prefer tool override description, then workflow function description,
                            # then server description
                            description = ""
                            if fn_short in override_configs and override_configs[fn_short].description:
                                description = override_configs[fn_short].description
                            elif wf_fn.description:
                                description = wf_fn.description
                            elif available and orig_name in server_tools:
                                description = server_tools[orig_name].description or ""

                            tools_info.append(
                                MCPToolInfo(name=fn_short,
                                            description=description or "",
                                            server=client.server_name,
                                            available=available).model_dump())

                        # Sort tools_info by name to maintain consistent ordering
                        tools_info.sort(key=lambda x: x['name'])

                        mcp_clients_info.append({
                            "function_group": group_name,
                            "server": client.server_name,
                            "transport": config.server.transport,
                            "session_healthy": session_healthy,
                            "protected": True if config.server.auth_provider is not None else False,
                            "tools": tools_info,
                            "total_tools": len(configured_short_names),
                            "available_tools": available_count
                        })

                    except Exception as e:
                        logger.error(f"Error processing MCP client {group_name}: {e}")
                        mcp_clients_info.append({
                            "function_group": group_name,
                            "server": "unknown",
                            "transport": config.server.transport if config.server else "unknown",
                            "session_healthy": False,
                            "protected": False,
                            "error": str(e),
                            "tools": [],
                            "total_tools": 0,
                            "workflow_tools": 0
                        })

                return MCPClientToolListResponse(mcp_clients=mcp_clients_info)

            except Exception as e:
                logger.error(f"Error in MCP client tool list endpoint: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve MCP client information: {str(e)}")

        # Add the route to the FastAPI app
        app.add_api_route(
            path="/mcp/client/tool/list",
            endpoint=get_mcp_client_tool_list,
            methods=["GET"],
            response_model=MCPClientToolListResponse,
            description="Get list of MCP client tools with session health and workflow configuration comparison",
            responses={
                200: {
                    "description": "Successfully retrieved MCP client tool information",
                    "content": {
                        "application/json": {
                            "example": {
                                "mcp_clients": [{
                                    "function_group": "mcp_tools",
                                    "server": "streamable-http:http://localhost:9901/mcp",
                                    "transport": "streamable-http",
                                    "session_healthy": True,
                                    "protected": False,
                                    "tools": [{
                                        "name": "tool_a",
                                        "description": "Tool A description",
                                        "server": "streamable-http:http://localhost:9901/mcp",
                                        "available": True
                                    }],
                                    "total_tools": 1,
                                    "available_tools": 1
                                }]
                            }
                        }
                    }
                },
                500: {
                    "description": "Internal Server Error"
                }
            })

    async def _add_flow(self, state: str, flow_state: FlowState):
        async with self._outstanding_flows_lock:
            self._outstanding_flows[state] = flow_state

    async def _remove_flow(self, state: str):
        async with self._outstanding_flows_lock:
            del self._outstanding_flows[state]


# Prevent Sphinx from documenting items not a part of the public API
__all__ = ["FastApiFrontEndPluginWorkerBase", "FastApiFrontEndPluginWorker", "RouteInfo"]
