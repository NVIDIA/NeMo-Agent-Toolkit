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
#
# Evaluation endpoints (/evaluate, /evaluate/item).

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request

from nat.eval.config import EvaluationRunOutput
from nat.eval.evaluate import EvaluationRun
from nat.eval.evaluate import EvaluationRunConfig
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.front_ends.fastapi.fastapi_front_end_config import EvaluateItemRequest
from nat.front_ends.fastapi.fastapi_front_end_config import EvaluateItemResponse
from nat.front_ends.fastapi.fastapi_front_end_config import EvaluateRequest
from nat.front_ends.fastapi.fastapi_front_end_config import EvaluateResponse
from nat.front_ends.fastapi.fastapi_front_end_config import EvaluateStatusResponse
from nat.runtime.loader import load_workflow
from nat.runtime.session import SessionManager

try:
    from nat.front_ends.fastapi.job_store import JobInfo
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore
except ImportError:
    JobInfo = None
    JobStatus = None
    JobStore = None

if TYPE_CHECKING:
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

logger = logging.getLogger(__name__)


async def register_evaluate_route(worker: FastApiFrontEndPluginWorker, app: FastAPI,
                                  session_manager: SessionManager) -> None:
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
                output: EvaluationRunOutput = await eval_runner.run_and_evaluate(session_manager=local_session_manager,
                                                                                 job_id=job_id)

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
                job_status = await worker._job_store.get_status(request.job_id)
                if job_status != JobStatus.NOT_FOUND:
                    return EvaluateResponse(job_id=request.job_id, status=job_status)

            job_id = worker._job_store.ensure_job_id(request.job_id)

            await worker._job_store.submit_job(job_id=job_id,
                                               config_file=request.config_file,
                                               expiry_seconds=request.expiry_seconds,
                                               job_fn=run_evaluation,
                                               job_args=[
                                                   worker._scheduler_address,
                                                   worker._db_url,
                                                   worker._config_file_path,
                                                   job_id,
                                                   request.config_file,
                                                   request.reps
                                               ])

            logger.info("Submitted evaluation job %s with config %s", job_id, request.config_file)

            return EvaluateResponse(job_id=job_id, status=JobStatus.SUBMITTED)

    def translate_job_to_response(job: JobInfo) -> EvaluateStatusResponse:
        """Translate a JobInfo object to an EvaluateStatusResponse."""
        return EvaluateStatusResponse(job_id=job.job_id,
                                      status=job.status,
                                      config_file=str(job.config_file),
                                      error=job.error,
                                      output_path=str(job.output_path),
                                      created_at=job.created_at,
                                      updated_at=job.updated_at,
                                      expires_at=worker._job_store.get_expires_at(job))

    async def get_job_status(job_id: str, http_request: Request) -> EvaluateStatusResponse:
        """Get the status of an evaluation job."""
        logger.info("Getting status for job %s", job_id)

        async with session_manager.session(http_connection=http_request):

            job = await worker._job_store.get_job(job_id)
            if not job:
                logger.warning("Job %s not found", job_id)
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            logger.info("Found job %s with status %s", job_id, job.status)
            return translate_job_to_response(job)

    async def get_last_job_status(http_request: Request) -> EvaluateStatusResponse:
        """Get the status of the last created evaluation job."""
        logger.info("Getting last job status")

        async with session_manager.session(http_connection=http_request):

            job = await worker._job_store.get_last_job()
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
                jobs = await worker._job_store.get_all_jobs()
            else:
                logger.info("Getting jobs with status %s", status)
                jobs = await worker._job_store.get_jobs_by_status(JobStatus(status))

            logger.info("Found %d jobs", len(jobs))
            return [translate_job_to_response(job) for job in jobs]

    if worker.front_end_config.evaluate.path:
        if worker._dask_available:
            # Add last job endpoint first (most specific)
            worker._register_api_route(
                app,
                path=f"{worker.front_end_config.evaluate.path}/job/last",
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
            worker._register_api_route(
                app,
                path=f"{worker.front_end_config.evaluate.path}/job/{{job_id}}",
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
            worker._register_api_route(
                app,
                path=f"{worker.front_end_config.evaluate.path}/jobs",
                endpoint=get_jobs,
                methods=["GET"],
                response_model=list[EvaluateStatusResponse],
                description="Get all jobs, optionally filtered by status",
                responses={500: response_500},
            )

            # Add HTTP endpoint for evaluation
            worker._register_api_route(
                app,
                path=worker.front_end_config.evaluate.path,
                endpoint=start_evaluation,
                methods=[worker.front_end_config.evaluate.method],
                response_model=EvaluateResponse,
                description=worker.front_end_config.evaluate.description,
                responses={500: response_500},
            )
        else:
            logger.warning("Dask is not available, evaluation endpoints will not be added.")


async def register_evaluate_item_route(worker: FastApiFrontEndPluginWorker,
                                       app: FastAPI,
                                       session_manager: SessionManager) -> None:
    """Add the single-item evaluation endpoint to the FastAPI app."""

    async def evaluate_single_item(request: EvaluateItemRequest, http_request: Request) -> EvaluateItemResponse:
        """Handle single-item evaluation requests."""

        async with session_manager.session(http_connection=http_request):

            # Check if evaluator exists
            if request.evaluator_name not in worker._evaluators:
                raise HTTPException(status_code=404,
                                    detail=f"Evaluator '{request.evaluator_name}' not found. "
                                    f"Available evaluators: {list(worker._evaluators.keys())}")

            try:
                # Get the evaluator
                evaluator = worker._evaluators[request.evaluator_name]

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
    if worker.front_end_config.evaluate_item.path:
        worker._register_api_route(path=worker.front_end_config.evaluate_item.path,
                                   app=app,
                                   endpoint=evaluate_single_item,
                                   methods=[worker.front_end_config.evaluate_item.method],
                                   response_model=EvaluateItemResponse,
                                   description=worker.front_end_config.evaluate_item.description,
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
        logger.info("Added evaluate_item route at %s", worker.front_end_config.evaluate_item.path)
