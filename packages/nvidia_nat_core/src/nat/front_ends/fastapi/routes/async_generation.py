# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Async generation route helpers."""

import json
import logging
from typing import Any
from typing import cast

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from pydantic import BaseModel
from pydantic import Field

from nat.front_ends.fastapi.async_jobs.async_job import run_generation
from nat.front_ends.fastapi.fastapi_front_end_config import AsyncGenerateResponse
from nat.front_ends.fastapi.fastapi_front_end_config import AsyncGenerationStatusResponse
from nat.front_ends.fastapi.routes.common_utils import _serialize_request
from nat.front_ends.fastapi.routes.common_utils import _with_annotation
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


def _job_status_to_response(worker: Any, job):
    job_output = job.output
    if job_output is not None:
        try:
            job_output = json.loads(job_output)
        except json.JSONDecodeError:
            logger.exception("Failed to parse job output as JSON: %s", job_output)
            job_output = {"error": "Output parsing failed"}

    return AsyncGenerationStatusResponse(job_id=job.job_id,
                                         status=job.status,
                                         error=job.error,
                                         output=job_output,
                                         created_at=job.created_at,
                                         updated_at=job.updated_at,
                                         expires_at=worker._job_store.get_expires_at(job))


def post_async_generation(*, worker: Any, session_manager: SessionManager, request_type: Any):
    """Build async generation POST handler."""
    from nat.front_ends.fastapi.async_jobs.job_store import JobStatus

    async def start_async_generation(request: Any, response: Response, http_request: Request):
        async with session_manager.session(http_connection=http_request):
            if request.job_id:
                job = await worker._job_store.get_job(request.job_id)
                if job:
                    return AsyncGenerateResponse(job_id=job.job_id, status=job.status)

            job_id = worker._job_store.ensure_job_id(request.job_id)
            (_, job) = await worker._job_store.submit_job(
                job_id=job_id,
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
                    request.model_dump(mode="json", exclude=["job_id", "sync_timeout", "expiry_seconds"]),
                    _serialize_request(http_request),
                ],
            )

            if job is not None:
                response.status_code = 200
                return _job_status_to_response(worker, job)

            response.status_code = 202
            return AsyncGenerateResponse(job_id=job_id, status=JobStatus.SUBMITTED)

    return _with_annotation(start_async_generation, "request", request_type)


def get_async_job_status(*, worker: Any, session_manager: SessionManager):
    """Build async generation status GET handler."""

    async def _get_async_job_status(job_id: str, http_request: Request):
        logger.info("Getting status for job %s", job_id)
        async with session_manager.session(http_connection=http_request):
            job = await worker._job_store.get_job(job_id)
            if job is None:
                logger.warning("Job %s not found", job_id)
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            logger.info("Found job %s with status %s", job_id, job.status)
            return _job_status_to_response(worker, job)

    return _get_async_job_status


async def add_async_generation_routes(
    *,
    worker: Any,
    app: FastAPI,
    endpoint: Any,
    session_manager: SessionManager,
    generate_body_type: Any,
    response_500: dict[str, Any],
    disable_legacy_routes: bool = False,
) -> None:
    """Register async generation submission and status routes."""

    if not (worker._dask_available and not hasattr(endpoint, "function_name")):
        logger.warning("Dask is not available, async generation endpoints will not be added.")
        return

    from nat.front_ends.fastapi.async_jobs.job_store import JobStore

    if not (isinstance(generate_body_type, type) and issubclass(generate_body_type, BaseModel)):
        logger.warning("Async generation requires a BaseModel request schema; skipping async route.")
        return

    base_request_model = cast(type[BaseModel], generate_body_type)

    class AsyncGenerateRequest(base_request_model):
        job_id: str | None = Field(default=None, description="Unique identifier for the evaluation job")
        sync_timeout: int = Field(
            default=0,
            ge=0,
            le=300,
            description="Attempt to perform the job synchronously up until `sync_timeout` seconds, "
            "if the job hasn't been completed by then a job_id will be returned with a status code of 202.",
        )
        expiry_seconds: int = Field(default=JobStore.DEFAULT_EXPIRY,
                                    ge=JobStore.MIN_EXPIRY,
                                    le=JobStore.MAX_EXPIRY,
                                    description="Optional time (in seconds) before the job expires. "
                                    "Clamped between 600 (10 min) and 86400 (24h).")

        def validate_model(self):
            return self

    app.add_api_route(
        path=f"{endpoint.path}/async",
        endpoint=post_async_generation(worker=worker,
                                       session_manager=session_manager,
                                       request_type=AsyncGenerateRequest),
        methods=[endpoint.method],
        response_model=AsyncGenerateResponse | AsyncGenerationStatusResponse,
        description="Start an async generate job",
        responses={500: response_500},
    )

    app.add_api_route(
        path=f"{endpoint.path}/async/job/{{job_id}}",
        endpoint=get_async_job_status(worker=worker, session_manager=session_manager),
        methods=["GET"],
        response_model=AsyncGenerationStatusResponse,
        description="Get the status of an async job",
        responses={
            404: {
                "description": "Job not found"
            }, 500: response_500
        },
    )

    if not disable_legacy_routes and getattr(endpoint, "legacy_path", None):
        app.add_api_route(
            path=f"{endpoint.legacy_path}/async",
            endpoint=post_async_generation(worker=worker,
                                           session_manager=session_manager,
                                           request_type=AsyncGenerateRequest),
            methods=[endpoint.method],
            response_model=AsyncGenerateResponse | AsyncGenerationStatusResponse,
            description="Start an async generate job (legacy path)",
            responses={500: response_500},
        )

        app.add_api_route(
            path=f"{endpoint.legacy_path}/async/job/{{job_id}}",
            endpoint=get_async_job_status(worker=worker, session_manager=session_manager),
            methods=["GET"],
            response_model=AsyncGenerationStatusResponse,
            description="Get the status of an async job (legacy path)",
            responses={
                404: {
                    "description": "Job not found"
                }, 500: response_500
            },
        )
