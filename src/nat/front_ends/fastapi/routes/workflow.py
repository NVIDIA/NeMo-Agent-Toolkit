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

import json
import logging
import typing
from typing import TYPE_CHECKING

from fastapi import Body
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import Field

from nat.builder.context import Context
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.front_ends.fastapi.async_job import run_generation
from nat.front_ends.fastapi.auth_flow_handlers.http_flow_handler import HTTPAuthenticationFlowHandler
from nat.front_ends.fastapi.fastapi_front_end_config import AsyncGenerateResponse
from nat.front_ends.fastapi.fastapi_front_end_config import AsyncGenerationStatusResponse
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.response_helpers import generate_single_response
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_as_str
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_full_as_str
from nat.runtime.session import SessionManager

try:
    from nat.front_ends.fastapi.job_store import JobInfo
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore
except ImportError:
    JobInfo = typing.cast(typing.Any, None)
    JobStatus = typing.cast(typing.Any, None)
    JobStore = typing.cast(typing.Any, None)

DEFAULT_EXPIRY = JobStore.DEFAULT_EXPIRY if JobStore else 86400
MIN_EXPIRY = JobStore.MIN_EXPIRY if JobStore else 600
MAX_EXPIRY = JobStore.MAX_EXPIRY if JobStore else 86400

if TYPE_CHECKING:
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

logger = logging.getLogger(__name__)


async def register_workflow_route(worker: "FastApiFrontEndPluginWorker",
                                  app: FastAPI,
                                  endpoint: FastApiFrontEndConfig.EndpointBase,
                                  session_manager: SessionManager) -> None:
    """Register workflow and OpenAI-compatible HTTP endpoints."""

    RequestBodyType: typing.Any = session_manager.get_workflow_input_schema()
    StreamResponseType: typing.Any = session_manager.get_workflow_streaming_output_schema()
    SingleResponseType: typing.Any = session_manager.get_workflow_single_output_schema()
    StreamResponseModel: typing.Any = StreamResponseType
    SingleResponseModel: typing.Any = SingleResponseType
    auth_handler = worker._http_flow_handler or HTTPAuthenticationFlowHandler()
    job_store = typing.cast(JobStore | None, worker._job_store)
    async_generate_request_type: typing.Any = None

    def add_context_headers_to_response(response: Response) -> None:
        """Add context-based headers to response if available."""
        observability_trace_id = Context.get().observability_trace_id
        if observability_trace_id:
            response.headers["Observability-Trace-Id"] = observability_trace_id

    # Skip async generation for custom routes (those with function_name)
    if worker._dask_available and not hasattr(endpoint, 'function_name'):

        class AsyncGenerateRequest(RequestBodyType):
            job_id: str | None = Field(default=None, description="Unique identifier for the evaluation job")
            sync_timeout: int = Field(
                default=0,
                ge=0,
                le=300,
                description="Attempt to perform the job synchronously up until `sync_timeout` sectonds, "
                "if the job hasn't been completed by then a job_id will be returned with a status code of 202.")
            expiry_seconds: int = Field(default=DEFAULT_EXPIRY,
                                        ge=MIN_EXPIRY,
                                        le=MAX_EXPIRY,
                                        description="Optional time (in seconds) before the job expires. "
                                        "Clamped between 600 (10 min) and 86400 (24h).")

            def validate_model(self):
                # Override to ensure that the parent class validator is not called
                return self

        async_generate_request_type = AsyncGenerateRequest

    # Ensure that the input is in the body. POD types are treated as query parameters
    try:
        if (not issubclass(RequestBodyType, BaseModel)):
            RequestBodyType = typing.Annotated[RequestBodyType, Body()]
        else:
            logger.info("Expecting generate request payloads in the following format: %s", RequestBodyType.model_fields)
            # Explicitly mark BaseModel payloads as body parameters to avoid query parsing.
            RequestBodyType = typing.Annotated[RequestBodyType, Body()]
    except TypeError:
        # If RequestBodyType is not a class, leave as-is
        pass

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

    def _add_api_route(path: str, **kwargs):
        worker._register_api_route(app, path=path, **kwargs)

    def get_single_endpoint(result_type: type | None):

        async def get_single(response: Response, request: Request):

            response.headers["Content-Type"] = "application/json"

            async with session_manager.session(
                    http_connection=request,
                    user_authentication_callback=auth_handler.authenticate,
                    user_input_callback=worker._get_user_input_callback(request)) as session:  # type: ignore[arg-type]

                result = await generate_single_response(None, session, result_type=result_type)
                add_context_headers_to_response(response)
                return result

        return get_single

    def get_streaming_endpoint(streaming: bool, result_type: type | None, output_type: type | None):

        async def get_stream(request: Request):

            async with session_manager.session(
                    http_connection=request,
                    user_authentication_callback=auth_handler.authenticate,
                    user_input_callback=worker._get_user_input_callback(request)) as session:  # type: ignore[arg-type]

                return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                         content=generate_streaming_response_as_str(
                                             None,
                                             session=session,
                                             streaming=streaming,
                                             step_adaptor=worker.get_step_adaptor(),
                                             result_type=result_type,
                                             output_type=output_type,
                                             send_done=True,
                                             include_usage=False))

        return get_stream

    def get_streaming_raw_endpoint(streaming: bool, result_type: type | None, output_type: type | None):

        async def get_stream(filter_steps: str | None = None):

            async with session_manager.session(http_connection=None) as session:
                return StreamingResponse(
                    headers={"Content-Type": "text/event-stream; charset=utf-8"},
                    content=generate_streaming_response_full_as_str(
                        None,
                        session=session,
                        streaming=streaming,
                        result_type=result_type,
                        output_type=output_type,
                        filter_steps=filter_steps,
                        send_done=True,
                        include_usage=False,
                        include_trace=worker.front_end_config.observability.embed_trace_in_response))

        return get_stream

    def post_single_endpoint(request_type: type, result_type: type | None):

        async def post_single(response: Response, request: Request, payload: request_type):

            response.headers["Content-Type"] = "application/json"

            async with session_manager.session(
                    http_connection=request,
                    user_authentication_callback=auth_handler.authenticate,
                    user_input_callback=worker._get_user_input_callback(request)) as session:  # type: ignore[arg-type]

                result = await generate_single_response(payload, session, result_type=result_type)
                worker._ensure_usage(result)
                add_context_headers_to_response(response)
                return result

        return post_single

    def post_streaming_endpoint(request_type: type, streaming: bool, result_type: type | None,
                                output_type: type | None):

        async def post_stream(request: Request, payload: request_type):

            async with session_manager.session(
                    http_connection=request,
                    user_authentication_callback=auth_handler.authenticate,
                    user_input_callback=worker._get_user_input_callback(request)) as session:  # type: ignore[arg-type]

                return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                         content=generate_streaming_response_as_str(
                                             payload,
                                             session=session,
                                             streaming=streaming,
                                             step_adaptor=worker.get_step_adaptor(),
                                             result_type=result_type,
                                             output_type=output_type,
                                             send_done=True,
                                             include_usage=False))

        return post_stream

    def post_streaming_raw_endpoint(request_type: type,
                                    streaming: bool,
                                    result_type: type | None,
                                    output_type: type | None):
        """
        Stream raw intermediate steps without any step adaptor translations.
        """

        async def post_stream(payload: request_type, filter_steps: str | None = None):

            async with session_manager.session(http_connection=None) as session:
                return StreamingResponse(
                    headers={"Content-Type": "text/event-stream; charset=utf-8"},
                    content=generate_streaming_response_full_as_str(
                        payload,
                        session=session,
                        streaming=streaming,
                        result_type=result_type,
                        output_type=output_type,
                        filter_steps=filter_steps,
                        send_done=True,
                        include_usage=False,
                        include_trace=worker.front_end_config.observability.embed_trace_in_response))

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

            async with session_manager.session(
                    http_connection=request,
                    user_input_callback=worker._get_user_input_callback(request)) as session:  # type: ignore[arg-type]
                try:
                    if stream_requested:

                        # Return streaming response
                        include_usage = False
                        stream_options = getattr(payload, "stream_options", None)
                        if stream_options and isinstance(stream_options, dict):
                            include_usage = bool(stream_options.get("include_usage", False))

                        return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                                 content=generate_streaming_response_as_str(
                                                     payload,
                                                     session=session,
                                                     streaming=True,
                                                     step_adaptor=worker.get_step_adaptor(),
                                                     result_type=ChatResponseChunk,
                                                     output_type=ChatResponseChunk,
                                                     send_done=True,
                                                     include_usage=include_usage))

                    result = await generate_single_response(payload, session, result_type=ChatResponse)
                    add_context_headers_to_response(response)
                    worker._ensure_usage(result)
                    return result
                except HTTPException:
                    raise
                except Exception as exc:  # OpenAI-compatible error shape
                    return worker._openai_error_response(exc)

        return post_openai_api_compatible

    def _job_status_to_response(job: JobInfo) -> AsyncGenerationStatusResponse:
        if job_store is None:
            raise HTTPException(status_code=503, detail="Async generation unavailable (job store not configured).")
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
                                             expires_at=job_store.get_expires_at(job))

    def post_async_generation(request_type: type):

        async def start_async_generation(
                request: request_type, response: Response,
                http_request: Request) -> AsyncGenerateResponse | AsyncGenerationStatusResponse:
            """Handle async generation requests."""

            if job_store is None:
                raise HTTPException(status_code=503, detail="Async generation unavailable (job store not configured).")

            async with session_manager.session(http_connection=http_request):

                # if job_id is present and already exists return the job info
                if request.job_id:
                    job = await job_store.get_job(request.job_id)
                    if job:
                        return AsyncGenerateResponse(job_id=job.job_id, status=job.status)

                job_id = job_store.ensure_job_id(request.job_id)
                (_, job) = await job_store.submit_job(job_id=job_id,
                                                      expiry_seconds=request.expiry_seconds,
                                                      job_fn=run_generation,
                                                      sync_timeout=request.sync_timeout,
                                                      job_args=[
                                                          not worker._use_dask_threads,
                                                          worker._log_level,
                                                          worker._scheduler_address,
                                                          worker._db_url,
                                                          worker._config_file_path,
                                                          job_id,
                                                          request.model_dump(
                                                              mode="json",
                                                              exclude=["job_id", "sync_timeout", "expiry_seconds"])
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

        if job_store is None:
            raise HTTPException(status_code=503, detail="Async generation unavailable (job store not configured).")

        async with session_manager.session(http_connection=http_request):

            job = await job_store.get_job(job_id)
            if job is None:
                logger.warning("Job %s not found", job_id)
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            logger.info("Found job %s with status %s", job_id, job.status)
            return _job_status_to_response(job)

    if (endpoint.path):

        if (endpoint.method == "GET"):

            _add_api_route(
                path=endpoint.path,
                endpoint=get_single_endpoint(result_type=SingleResponseType),
                methods=[endpoint.method],
                response_model=SingleResponseModel,
                description=endpoint.description,
                responses={500: response_500},
            )

            _add_api_route(
                path=f"{endpoint.path}/stream",
                endpoint=get_streaming_endpoint(streaming=True,
                                                result_type=StreamResponseType,
                                                output_type=StreamResponseType),
                methods=[endpoint.method],
                response_model=StreamResponseModel,
                description=endpoint.description,
                responses={500: response_500},
            )

            _add_api_route(
                path=f"{endpoint.path}/full",
                endpoint=get_streaming_raw_endpoint(streaming=True,
                                                    result_type=StreamResponseType,
                                                    output_type=StreamResponseType),
                methods=[endpoint.method],
                description="Stream raw intermediate steps without any step adaptor translations.\n"
                "Use filter_steps query parameter to filter steps by type (comma-separated list) or"
                "                        set to 'none' to suppress all intermediate steps.",
            )

        elif (endpoint.method == "POST"):

            _add_api_route(
                path=endpoint.path,
                endpoint=post_single_endpoint(request_type=RequestBodyType, result_type=SingleResponseType),
                methods=[endpoint.method],
                response_model=SingleResponseModel,
                description=endpoint.description,
                responses={500: response_500},
            )

            _add_api_route(
                path=f"{endpoint.path}/stream",
                endpoint=post_streaming_endpoint(request_type=RequestBodyType,
                                                 streaming=True,
                                                 result_type=StreamResponseType,
                                                 output_type=StreamResponseType),
                methods=[endpoint.method],
                response_model=StreamResponseModel,
                description=endpoint.description,
                responses={500: response_500},
            )

            _add_api_route(
                path=f"{endpoint.path}/full",
                endpoint=post_streaming_raw_endpoint(request_type=RequestBodyType,
                                                     streaming=True,
                                                     result_type=StreamResponseType,
                                                     output_type=StreamResponseType),
                methods=[endpoint.method],
                response_model=StreamResponseModel,
                description="Stream raw intermediate steps without any step adaptor translations.\n"
                "Use filter_steps query parameter to filter steps by type (comma-separated list) or "
                "                        set to 'none' to suppress all intermediate steps.",
                responses={500: response_500},
            )

            if worker._dask_available and not hasattr(endpoint, 'function_name'):
                assert async_generate_request_type is not None
                _add_api_route(
                    path=f"{endpoint.path}/async",
                    endpoint=post_async_generation(request_type=async_generate_request_type),
                    methods=[endpoint.method],
                    response_model=AsyncGenerateResponse | AsyncGenerationStatusResponse,
                    description="Start an async generate job",
                    responses={500: response_500},
                )
            else:
                logger.warning("Dask is not available, async generation endpoints will not be added.")
        else:
            raise ValueError(f"Unsupported method {endpoint.method}")

        if worker._dask_available and not hasattr(endpoint, 'function_name'):
            _add_api_route(
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

    if endpoint.openai_api_v1_path:
        if endpoint.method == "POST":
            _add_api_route(
                path=endpoint.openai_api_v1_path,
                endpoint=post_openai_api_compatible_endpoint(request_type=ChatRequest),
                methods=[endpoint.method],
                response_model=ChatResponse | ChatResponseChunk,
                description=f"{endpoint.description} (OpenAI Chat Completions API compatible)",
                responses={500: response_500},
            )
        else:
            raise ValueError(f"Unsupported method {endpoint.method}")

    legacy_enabled = worker.front_end_config.versioning.enable_legacy_routes
    is_default_endpoint = not hasattr(endpoint, "function_name")
    if legacy_enabled and is_default_endpoint and endpoint.method == "POST":

        def _add_legacy_api_route(path: str, **kwargs):
            app.add_api_route(path=path, **kwargs)

        _add_legacy_api_route(
            path="/generate",
            endpoint=post_single_endpoint(request_type=RequestBodyType, result_type=SingleResponseType),
            methods=[endpoint.method],
            response_model=SingleResponseModel,
            description=f"{endpoint.description} (legacy)",
            responses={500: response_500},
        )

        _add_legacy_api_route(
            path="/generate/stream",
            endpoint=post_streaming_endpoint(request_type=RequestBodyType,
                                             streaming=True,
                                             result_type=StreamResponseType,
                                             output_type=StreamResponseType),
            methods=[endpoint.method],
            response_model=StreamResponseModel,
            description=f"{endpoint.description} (legacy stream)",
            responses={500: response_500},
        )

        _add_legacy_api_route(
            path="/generate/full",
            endpoint=post_streaming_raw_endpoint(request_type=RequestBodyType,
                                                 streaming=True,
                                                 result_type=StreamResponseType,
                                                 output_type=StreamResponseType),
            methods=[endpoint.method],
            response_model=StreamResponseModel,
            description="Stream raw intermediate steps without any step adaptor translations.\n"
            "Use filter_steps query parameter to filter steps by type (comma-separated list) or "
            "                        set to 'none' to suppress all intermediate steps.",
            responses={500: response_500},
        )
