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

import json
import logging
import os
import shutil
import typing
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
from operator import attrgetter
from uuid import uuid4

import dask.config
from dask.distributed import Client
from dask.distributed import Future
from dask.distributed import Lock
from dask.distributed import Variable
from dask.distributed import fire_and_forget
from dask.distributed.protocol import dask_deserialize
from dask.distributed.protocol import dask_serialize
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"
    NOT_FOUND = "not_found"


# pydantic model for the job status
class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    config_file: str | None
    error: str | None
    output_path: str | None
    created_at: datetime
    updated_at: datetime
    expiry_seconds: int
    output: BaseModel | None = None


# Register custom serializers for JobInfo with Dask if available.
@dask_serialize.register(JobInfo)
def serialize_job_info(job: JobInfo) -> tuple[dict, list[bytes]]:
    """Custom serialization for JobInfo."""
    header = {}

    job_dict = job.model_dump(mode="json", round_trip=True)
    json_str = json.dumps(job_dict)
    frames = [json_str.encode('utf-8')]

    return (header, frames)


@dask_deserialize.register(JobInfo)
def deserialize_job_info(header: dict, frames: list[bytes]) -> JobInfo:
    """Custom deserialization for JobInfo."""
    # model_validate_json accepts a bytes string, no need to decode first
    return JobInfo.model_validate_json(frames[0])


class JobStore:

    MIN_EXPIRY = 600  # 10 minutes
    MAX_EXPIRY = 86400  # 24 hours
    DEFAULT_EXPIRY = 3600  # 1 hour

    # active jobs are exempt from expiry
    ACTIVE_STATUS = {"running", "submitted"}

    def __init__(self, scheduler_address: str):
        self._scheduler_address = scheduler_address
        self._client: Client | None = None
        self._lock: Lock | None = None

    @asynccontextmanager
    async def client(self):
        if self._client is None:
            self._client = await Client(address=self._scheduler_address, asynchronous=True)

        yield self._client

    @asynccontextmanager
    async def lock(self):
        # refer to https://distributed.dask.org/en/stable/api.html?highlight=future#distributed.Lock
        with dask.config.set({"distributed.scheduler.locks.lease-timeout": "inf"}):
            if self._lock is None:
                async with self.client() as client:
                    self._lock = Lock(name="job_store_lock", client=client)

            await self._lock.acquire()
            yield self._lock
            await self._lock.release()

    async def close(self):
        """Close the Dask client."""
        if self._client is not None:
            await self._client.close()

    def ensure_job_id(self, job_id: str | None) -> str:
        """Ensure a job ID is provided, generating a new one if necessary."""
        if job_id is None:
            job_id = str(uuid4())
            logger.info("Generated new job ID: %s", job_id)

        return job_id

    async def create_job(self,
                         config_file: str | None = None,
                         job_id: str | None = None,
                         expiry_seconds: int = DEFAULT_EXPIRY) -> str:
        job_id = self.ensure_job_id(job_id)

        clamped_expiry = max(self.MIN_EXPIRY, min(expiry_seconds, self.MAX_EXPIRY))
        if expiry_seconds != clamped_expiry:
            logger.info("Clamped expiry_seconds from %d to %d for job %s", expiry_seconds, clamped_expiry, job_id)

        job = JobInfo(job_id=job_id,
                      status=JobStatus.SUBMITTED,
                      config_file=config_file,
                      created_at=datetime.now(UTC),
                      updated_at=datetime.now(UTC),
                      error=None,
                      output_path=None,
                      expiry_seconds=clamped_expiry)

        async with (self.client() as client, self.lock()):
            await client.set_metadata(["jobs", job_id], job)

        logger.info("Created new job %s with config %s", job_id, config_file)
        return job_id

    async def submit_job(self,
                         *,
                         job_id: str | None = None,
                         config_file: str | None = None,
                         expiry_seconds: int = DEFAULT_EXPIRY,
                         job_fn: Callable[..., typing.Any],
                         job_args: list[typing.Any],
                         **job_kwargs) -> (str, Future):
        job_id = await self.create_job(job_id=job_id, config_file=config_file, expiry_seconds=expiry_seconds)

        # We are intentionally not using job_id as the key, since Dask will clear the associated metadata once
        # the job has completed, and we want the metadata to persist until the job expires.
        async with self.client() as client:
            print("\n***************\nsubmitting job\n***************\n", flush=True)
            print(f"job_args: {job_args}, job_kwargs: {job_kwargs}\n***************\n", flush=True)
            future = client.submit(job_fn, *job_args, key=f"{job_id}-job", **job_kwargs)

            print(f"\n***************\nconstructing future for job future={future}\n***************\n", flush=True)
            future_var = Variable(name=job_id, client=self._client)

            print("\n***************\nsetting future for job\n***************\n", flush=True)
            await future_var.set(future)
            print(f"\n***************\ndone - setting future for job future={future}\n***************\n", flush=True)
            fire_and_forget(future)

        return (job_id, future)

    async def update_status(self,
                            job_id: str,
                            status: str,
                            error: str | None = None,
                            output_path: str | None = None,
                            output: BaseModel | None = None):
        async with (self.client() as client, self.lock()):
            try:
                job: JobInfo = await client.get_metadata(["jobs", job_id])

                job.status = status
                job.error = error
                job.output_path = output_path
                job.updated_at = datetime.now(UTC)
                job.output = output

                await client.set_metadata(["jobs", job_id], job)
            except KeyError as e:
                raise ValueError(f"Job {job_id} not found") from e

    async def get_all_jobs(self) -> list[JobInfo]:
        async with (self.client() as client, self.lock()):
            job_dict: dict[str, JobInfo] = await client.get_metadata(["jobs"], default={})
            return list(job_dict.values())

    async def get_job(self, job_id: str) -> JobInfo | None:
        """Get a job by its ID."""
        async with (self.client() as client, self.lock()):
            return await client.get_metadata(["jobs", job_id], default=None)

    async def get_status(self, job_id: str) -> JobStatus:
        job = await self.get_job(job_id)
        if job is not None:
            return job.status
        else:
            return JobStatus.NOT_FOUND

    async def get_last_job(self) -> JobInfo | None:
        """Get the last created job."""
        jobs = await self.get_all_jobs()
        if len(jobs) == 0:
            logger.info("No jobs found in job store")
            return None

        last_job = max(jobs, key=attrgetter('created_at'))
        logger.info("Retrieved last job %s created at %s", last_job.job_id, last_job.created_at)
        return last_job

    async def get_jobs_by_status(self, status: JobStatus) -> list[JobInfo]:
        """Get all jobs with the specified status."""
        jobs = await self.get_all_jobs()
        return [job for job in jobs if job.status == status]

    def get_expires_at(self, job: JobInfo) -> datetime | None:
        """Get the time for a job to expire."""
        if job.status in self.ACTIVE_STATUS:
            return None
        return job.updated_at + timedelta(seconds=job.expiry_seconds)

    async def cleanup_expired_jobs(self):
        """
        Cleanup expired jobs, keeping the most recent one.
        Updated_at is used instead of created_at to determine the most recent job.
        This is because jobs may not be processed in the order they are created.
        """
        now = datetime.now(UTC)

        # Filter out active jobs
        async with (self.client() as client, self.lock()):
            jobs: dict[str, JobInfo] = await client.get_metadata(["jobs"], default={})
            finished_jobs = {job_id: job for job_id, job in jobs.items() if job.status not in self.ACTIVE_STATUS}

            # Sort finished jobs by updated_at descending
            sorted_finished = sorted(finished_jobs.items(), key=lambda item: item[1].updated_at, reverse=True)

            # Always keep the most recent finished job
            jobs_to_check = sorted_finished[1:]

            expired_ids = []
            for job_id, job in jobs_to_check:
                expires_at = self.get_expires_at(job)
                if expires_at and now > expires_at:
                    expired_ids.append(job_id)
                    # cleanup output dir if present
                    if job.output_path:
                        logger.info("Cleaning up output directory for job %s at %s", job_id, job.output_path)
                        # If it is a file remove it
                        if os.path.isfile(job.output_path):
                            os.remove(job.output_path)
                        # If it is a directory remove it
                        elif os.path.isdir(job.output_path):
                            shutil.rmtree(job.output_path)

            if len(expired_ids) > 0:
                var = Variable(name=job_id, client=self._client)
                try:
                    future = var.get(timeout=0)
                    if isinstance(future, Future):
                        await client.cancel([future], force=True, reason="Expired job cleanup")

                except TimeoutError:
                    pass

                var.delete()
                for job_id in expired_ids:
                    del jobs[job_id]

                await client.set_metadata(["jobs"], jobs)
