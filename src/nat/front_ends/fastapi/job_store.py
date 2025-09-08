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
from asyncio import current_task
from collections.abc import AsyncGenerator
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
from uuid import uuid4

from dask.distributed import Client as DaskClient
from dask.distributed import Future
from dask.distributed import Variable
from dask.distributed import fire_and_forget
from pydantic import BaseModel
from sqlalchemy import DateTime
from sqlalchemy import String
from sqlalchemy import and_
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_scoped_session
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.sql import expression as sa_expr

if typing.TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.ext.asyncio import AsyncEngine
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"
    NOT_FOUND = "not_found"


class Base(DeclarativeBase):
    pass


# pydantic model for the job status
class JobInfo(Base):
    __tablename__ = "job_info"

    job_id: Mapped[str] = mapped_column(primary_key=True)
    status: Mapped[JobStatus] = mapped_column(String(11))
    config_file: Mapped[str] = mapped_column(nullable=True)
    error: Mapped[str] = mapped_column(nullable=True)
    output_path: Mapped[str] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=datetime.now(UTC),
                                                 onupdate=datetime.now(UTC))
    expiry_seconds: Mapped[int]
    output: Mapped[str] = mapped_column(nullable=True)
    is_expired: Mapped[bool] = mapped_column(default=False, index=True)

    def __repr__(self):
        return f"JobInfo(job_id={self.job_id}, status={self.status})"


class JobStore:
    """
    Tracks and manages jobs submitted to the Dask scheduler, along with persisting job metadata (JobInfo objects) into
    a database.

    Parameters
    ----------
    scheduler_address: str
        The address of the Dask scheduler.
    db_engine: AsyncEngine
        The database engine for the job store.
    """

    MIN_EXPIRY = 600  # 10 minutes
    MAX_EXPIRY = 86400  # 24 hours
    DEFAULT_EXPIRY = 3600  # 1 hour

    # active jobs are exempt from expiry
    ACTIVE_STATUS = {"running", "submitted"}

    def __init__(self, scheduler_address: str, db_engine: "AsyncEngine | None" = None, db_url: str | None = None):
        self._scheduler_address = scheduler_address

        if db_engine is None:
            if db_url is None:
                raise ValueError("Either db_engine or db_url must be provided")

            db_engine = get_db_engine(db_url, use_async=True)

        # Disabling expire_on_commit allows us to detach (expunge) job instances from the session
        session_maker = async_sessionmaker(db_engine, expire_on_commit=False)

        # The async_scoped_session ensures that the same session is used within the same task, and that no two tasks
        # share the same session.
        self._session = async_scoped_session(session_maker, scopefunc=current_task)

    @asynccontextmanager
    async def client(self) -> AsyncGenerator[DaskClient]:
        """
        Get the Dask client.
        """
        client = await DaskClient(address=self._scheduler_address, asynchronous=True)

        yield client
        await client.close()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator["AsyncSession"]:
        """
        Async context manager for a SQLAlchemy session which explicitly begins a transaction.
        """
        async with self._session() as session:
            async with session.begin():
                yield session

        # Removes the current task key from the session registry, preventing potential memory leaks
        await self._session.remove()

    def ensure_job_id(self, job_id: str | None) -> str:
        """Ensure a job ID is provided, generating a new one if necessary."""
        if job_id is None:
            job_id = str(uuid4())
            logger.info("Generated new job ID: %s", job_id)

        return job_id

    async def _create_job(self,
                          config_file: str | None = None,
                          job_id: str | None = None,
                          expiry_seconds: int = DEFAULT_EXPIRY) -> str:
        """
        Create a job and add it to the job store. This should not be called directly, but instead be called by
        `submit_job`
        """
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

        async with self.session() as session:
            session.add(job)

        logger.info("Created new job %s with config %s", job_id, config_file)
        return job_id

    async def submit_job(self,
                         *,
                         job_id: str | None = None,
                         config_file: str | None = None,
                         expiry_seconds: int = DEFAULT_EXPIRY,
                         sync_timeout: int = 0,
                         job_fn: Callable[..., typing.Any],
                         job_args: list[typing.Any],
                         **job_kwargs) -> tuple[str, JobInfo | None]:
        job_id = await self._create_job(job_id=job_id, config_file=config_file, expiry_seconds=expiry_seconds)

        # We are intentionally not using job_id as the key, since Dask will clear the associated metadata once
        # the job has completed, and we want the metadata to persist until the job expires.
        async with self.client() as client:
            logger.debug("Submitting job with job_args: %s, job_kwargs: %s", job_args, job_kwargs)
            future = client.submit(job_fn, *job_args, key=f"{job_id}-job", **job_kwargs)

            # Store the future in a variable, this allows us to potentially cancel the future later if needed
            future_var = Variable(name=job_id, client=client)
            await future_var.set(future)

            if sync_timeout > 0:
                try:
                    _ = await future.result(timeout=sync_timeout)
                    job = await self.get_job(job_id)
                    assert job is not None, "Job should exist after future result"
                    return (job_id, job)
                except TimeoutError:
                    pass

            fire_and_forget(future)

        return (job_id, None)

    async def update_status(self,
                            job_id: str,
                            status: str,
                            error: str | None = None,
                            output_path: str | None = None,
                            output: BaseModel | None = None):

        async with self.session() as session:
            job: JobInfo = await session.get(JobInfo, job_id)
            if job is None:
                raise ValueError(f"Job {job_id} not found in job store")

            job.status = status
            job.error = error
            job.output_path = output_path
            job.updated_at = datetime.now(UTC)

            if isinstance(output, BaseModel):
                # Convert BaseModel to JSON string for storage
                output = output.model_dump_json(round_trip=True)

            if isinstance(output, (dict, list)):
                # Convert dict or list to JSON string for storage
                output = json.dumps(output)

            job.output = output

    async def get_all_jobs(self) -> list[JobInfo]:
        """
        Get all jobs, potentially costly if there are many jobs.
        """
        async with self.session() as session:
            return (await session.scalars(select(JobInfo))).all()

    async def get_job(self, job_id: str) -> JobInfo | None:
        """Get a job by its ID."""
        async with self.session() as session:
            return await session.get(JobInfo, job_id)

    async def get_status(self, job_id: str) -> JobStatus:
        job = await self.get_job(job_id)
        if job is not None:
            return JobStatus(job.status)
        else:
            return JobStatus.NOT_FOUND

    async def get_last_job(self) -> JobInfo | None:
        """Get the last created job."""
        stmt = select(JobInfo).order_by(JobInfo.created_at.desc())
        async with self.session() as session:
            last_job = (await session.scalars(stmt)).first()

        if last_job is None:
            logger.info("No jobs found in job store")
        else:
            logger.info("Retrieved last job %s created at %s", last_job.job_id, last_job.created_at)

        return last_job

    async def get_jobs_by_status(self, status: JobStatus) -> list[JobInfo]:
        """Get all jobs with the specified status."""
        stmt = select(JobInfo).where(JobInfo.status == status)
        async with self.session() as session:
            return (await session.scalars(stmt)).all()

    def get_expires_at(self, job: JobInfo) -> datetime | None:
        """Get the time for a job to expire."""
        if job.status in self.ACTIVE_STATUS:
            return None

        updated_at = job.updated_at
        if updated_at.tzinfo is None:
            # Not all DB backends support timezone aware datetimes
            updated_at = updated_at.replace(tzinfo=UTC)

        return updated_at + timedelta(seconds=job.expiry_seconds)

    async def cleanup_expired_jobs(self):
        """
        Cleanup expired jobs, keeping the most recent one.
        Updated_at is used instead of created_at to determine the most recent job.
        This is because jobs may not be processed in the order they are created.
        """
        now = datetime.now(UTC)

        stmt = select(JobInfo).where(
            and_(JobInfo.is_expired == sa_expr.false(),
                 JobInfo.status.not_in(self.ACTIVE_STATUS))).order_by(JobInfo.updated_at.desc())
        # Filter out active jobs
        async with (self.client() as client, self.session() as session):
            finished_jobs = (await session.execute(stmt)).scalars().all()

            # Always keep the most recent finished job
            jobs_to_check = finished_jobs[1:]

            expired_ids = []
            for job in jobs_to_check:
                expires_at = self.get_expires_at(job)
                if expires_at and now > expires_at:
                    expired_ids.append(job.job_id)
                    # cleanup output dir if present
                    if job.output_path:
                        logger.info("Cleaning up output directory for job %s at %s", job.job_id, job.output_path)
                        # If it is a file remove it
                        if os.path.isfile(job.output_path):
                            os.remove(job.output_path)
                        # If it is a directory remove it
                        elif os.path.isdir(job.output_path):
                            shutil.rmtree(job.output_path)

            if len(expired_ids) > 0:
                successfully_expired = []
                for job_id in expired_ids:
                    try:
                        var = Variable(name=job_id, client=client)
                        try:
                            future = await var.get(timeout=0)
                            if isinstance(future, Future):
                                await client.cancel([future], force=True, reason="Expired job cleanup")

                        except TimeoutError:
                            pass

                        var.delete()
                        successfully_expired.append(job_id)
                    except Exception as e:
                        logger.error("Failed to expire %s: %s", job_id, e)

                await session.execute(
                    (update(JobInfo).where(JobInfo.job_id.in_(successfully_expired)).values(is_expired=True)))


def get_db_engine(db_url: str | None = None, echo: bool = False, use_async: bool = True) -> "Engine | AsyncEngine":
    """
    Create a SQLAlchemy database engine, this should only be run once per process

    Parameters
    ----------
    db_url: str | None
        The database URL to connect to. Refer to https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls
    echo: bool
        If True, SQLAlchemy will log all SQL statements. Useful for debugging.
    use_async: bool
        If True, use the async database engine.
        The JobStore class requires an async database engine, setting `use_async` to False is only usefull for testing.
    """
    if db_url is None:
        db_url = os.environ.get("NAT_JOB_STORE_DB_URL")
        if db_url is None:
            dot_tmp_dir = os.path.join(os.getcwd(), ".tmp")
            os.makedirs(dot_tmp_dir, exist_ok=True)
            db_file = os.path.join(dot_tmp_dir, "job_store.db")
            if os.path.exists(db_file):
                logger.warning("Database file %s already exists, it will be overwritten.", db_file)
                os.remove(db_file)

            if use_async:
                driver = "+aiosqlite"
            else:
                driver = ""

            db_url = f"sqlite{driver}:///{db_file}"

    if use_async:
        # This is actually a blocking call, it just returns an AsyncEngine
        from sqlalchemy.ext.asyncio import create_async_engine as create_engine_fn
    else:
        from sqlalchemy import create_engine as create_engine_fn

    return create_engine_fn(db_url, echo=echo)
