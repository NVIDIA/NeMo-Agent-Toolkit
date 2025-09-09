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
import os
import typing
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

if typing.TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class _TestModel(BaseModel):
    value: str


async def simple_job_function(x: int, y: int = 10) -> int:
    """Simple function for testing job execution."""
    await asyncio.sleep(0.1)  # Simulate some work
    return x + y


async def failing_job_function() -> None:
    """Function that raises an exception for testing error handling."""
    raise ValueError("This job is designed to fail")


@pytest.mark.asyncio
async def test_job_store_init_with_engine(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test JobStore initialization with provided database engine."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    assert job_store._scheduler_address == dask_scheduler_address
    assert job_store._session is not None


@pytest.mark.asyncio
async def test_job_store_init_with_db_url(db_url: str, dask_scheduler_address: str):
    """Test JobStore initialization with database URL."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_url=db_url)

    assert job_store._scheduler_address == dask_scheduler_address
    assert job_store._session is not None


def test_job_store_init_missing_db_params(dask_scheduler_address: str):
    """Test JobStore fails when both db_engine and db_url are missing."""
    from nat.front_ends.fastapi.job_store import JobStore

    with pytest.raises(ValueError, match="Either db_engine or db_url must be provided"):
        JobStore(scheduler_address=dask_scheduler_address)


def test_ensure_job_id_with_existing_id(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test ensure_job_id returns the same ID when one is provided."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    existing_id = "test-job-123"
    result = job_store.ensure_job_id(existing_id)
    assert result == existing_id


def test_ensure_job_id_generates_new_id(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test ensure_job_id generates a new ID when None is provided."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    result = job_store.ensure_job_id(None)
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_create_job_default_params(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test job creation with default parameters."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job_id = await job_store._create_job()

    job = await job_store.get_job(job_id)
    assert job is not None
    assert job.job_id == job_id
    assert job.status == JobStatus.SUBMITTED
    assert job.config_file is None
    assert job.error is None
    assert job.output_path is None
    assert job.expiry_seconds == JobStore.DEFAULT_EXPIRY
    assert job.is_expired is False


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_create_job_with_params(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test job creation with custom parameters."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    config_file = "/path/to/config.yaml"
    custom_job_id = "custom-job-id"
    custom_expiry = 7200  # 2 hours

    job_id = await job_store._create_job(config_file=config_file, job_id=custom_job_id, expiry_seconds=custom_expiry)

    assert job_id == custom_job_id

    job = await job_store.get_job(job_id)
    assert job is not None
    assert job.config_file == config_file
    assert job.expiry_seconds == custom_expiry


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_create_job_clamps_expiry(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test job creation clamps expiry seconds to valid range."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    # Test too small expiry gets clamped to minimum
    job_id_small = await job_store._create_job(expiry_seconds=100)
    job_small = await job_store.get_job(job_id_small)
    assert job_small is not None
    assert job_small.expiry_seconds == JobStore.MIN_EXPIRY

    # Test too large expiry gets clamped to maximum
    job_id_large = await job_store._create_job(expiry_seconds=100000)
    job_large = await job_store.get_job(job_id_large)
    assert job_large is not None
    assert job_large.expiry_seconds == JobStore.MAX_EXPIRY


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_submit_job_success(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test successful job submission."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job_id, job_info = await job_store.submit_job(job_fn=simple_job_function, job_args=[5, 3])

    assert job_id is not None
    assert job_info is None  # sync_timeout is 0, so no immediate result

    # Verify job was created in database
    job = await job_store.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.SUBMITTED


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_submit_job_with_sync_timeout(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test job submission with sync timeout to get immediate result."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job_id, job_info = await job_store.submit_job(
        job_fn=simple_job_function,
        job_args=[5, 3],
        sync_timeout=5  # Wait up to 5 seconds
    )

    assert job_id is not None
    assert job_info is not None
    assert job_info.job_id == job_id


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_submit_job_with_kwargs(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test job submission with keyword arguments."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job_id, _ = await job_store.submit_job(
        job_fn=simple_job_function,
        job_args=[5],
        y=15  # keyword argument
    )

    assert job_id is not None

    # Verify job was created
    job = await job_store.get_job(job_id)
    assert job is not None


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_update_status_basic(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test basic status update."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    # Create a job first
    job_id = await job_store._create_job()

    # Update the status
    await job_store.update_status(job_id=job_id,
                                  status=JobStatus.RUNNING.value,
                                  error=None,
                                  output_path="/path/to/output")

    # Verify the update
    job = await job_store.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.RUNNING
    assert job.output_path == "/path/to/output"
    assert job.error is None


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_update_status_with_error(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test status update with error message."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job_id = await job_store._create_job()

    error_msg = "Something went wrong"
    await job_store.update_status(job_id=job_id, status=JobStatus.FAILURE.value, error=error_msg)

    job = await job_store.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.FAILURE
    assert job.error == error_msg


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_update_status_with_pydantic_output(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test status update with Pydantic model output."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job_id = await job_store._create_job()

    test_output = _TestModel(value="test result")
    await job_store.update_status(job_id=job_id, status=JobStatus.SUCCESS.value, output=test_output)

    job = await job_store.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.SUCCESS
    assert job.output is not None

    # Verify output was serialized to JSON
    output_data = json.loads(job.output)
    assert output_data["value"] == "test result"


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_update_status_with_dict_output(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test status update with dictionary output."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job_id = await job_store._create_job()

    test_output = {"result": "success", "count": 42}
    await job_store.update_status(job_id=job_id, status=JobStatus.SUCCESS.value, output=test_output)

    job = await job_store.get_job(job_id)
    assert job is not None
    assert job.output is not None

    # Verify output was serialized to JSON
    output_data = json.loads(job.output)
    assert output_data["result"] == "success"
    assert output_data["count"] == 42


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_update_status_nonexistent_job(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test updating status of non-existent job raises error."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    with pytest.raises(ValueError, match="Job nonexistent-job not found"):
        await job_store.update_status(job_id="nonexistent-job", status=JobStatus.SUCCESS.value)


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_get_job_existing(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test getting an existing job."""
    from nat.front_ends.fastapi.job_store import JobInfo
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job_id = await job_store._create_job()

    job = await job_store.get_job(job_id)
    assert job is not None
    assert job.job_id == job_id
    assert isinstance(job, JobInfo)


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_get_job_nonexistent(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test getting a non-existent job returns None."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job = await job_store.get_job("nonexistent-job")
    assert job is None


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_get_status_existing(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test getting status of an existing job."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job_id = await job_store._create_job()

    status = await job_store.get_status(job_id)
    assert status == JobStatus.SUBMITTED


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_get_status_nonexistent(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test getting status of non-existent job returns NOT_FOUND."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    status = await job_store.get_status("nonexistent-job")
    assert status == JobStatus.NOT_FOUND


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_get_all_jobs_empty(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test getting all jobs when database is empty."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    jobs = await job_store.get_all_jobs()
    assert jobs == []


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_get_all_jobs_multiple(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test getting all jobs with multiple jobs in database."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    # Create multiple jobs
    job_id1 = await job_store._create_job()
    job_id2 = await job_store._create_job()
    job_id3 = await job_store._create_job()

    jobs = await job_store.get_all_jobs()
    assert len(jobs) == 3

    job_ids = {job.job_id for job in jobs}
    assert job_ids == {job_id1, job_id2, job_id3}


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_get_last_job_empty(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test getting last job when database is empty."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job = await job_store.get_last_job()
    assert job is None


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_get_last_job_multiple(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test getting last job with multiple jobs in database."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    # Create jobs with small delay to ensure different timestamps
    await job_store._create_job()
    await asyncio.sleep(0.01)
    await job_store._create_job()
    await asyncio.sleep(0.01)
    job_id3 = await job_store._create_job()

    last_job = await job_store.get_last_job()
    assert last_job is not None
    assert last_job.job_id == job_id3


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_get_jobs_by_status(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test filtering jobs by status."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    # Create jobs with different statuses
    job_id1 = await job_store._create_job()
    job_id2 = await job_store._create_job()
    job_id3 = await job_store._create_job()

    # Update some statuses
    await job_store.update_status(job_id2, JobStatus.RUNNING.value)
    await job_store.update_status(job_id3, JobStatus.SUCCESS.value)

    # Test filtering
    submitted_jobs = await job_store.get_jobs_by_status(JobStatus.SUBMITTED)
    assert len(submitted_jobs) == 1
    assert submitted_jobs[0].job_id == job_id1

    running_jobs = await job_store.get_jobs_by_status(JobStatus.RUNNING)
    assert len(running_jobs) == 1
    assert running_jobs[0].job_id == job_id2

    success_jobs = await job_store.get_jobs_by_status(JobStatus.SUCCESS)
    assert len(success_jobs) == 1
    assert success_jobs[0].job_id == job_id3

    failure_jobs = await job_store.get_jobs_by_status(JobStatus.FAILURE)
    assert len(failure_jobs) == 0


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_get_expires_at_active_job(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test get_expires_at for active jobs returns None."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job_id = await job_store._create_job()
    job = await job_store.get_job(job_id)

    # Active jobs (submitted, running) should not expire
    assert job is not None
    assert job.status in job_store.ACTIVE_STATUS
    expires_at = job_store.get_expires_at(job)
    assert expires_at is None

    # Test with running status too
    await job_store.update_status(job_id, JobStatus.RUNNING.value)
    job = await job_store.get_job(job_id)
    assert job is not None
    expires_at = job_store.get_expires_at(job)
    assert expires_at is None


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_get_expires_at_finished_job(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test get_expires_at for finished jobs returns correct expiry time."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    job_id = await job_store._create_job(expiry_seconds=3600)

    # Update to finished status
    await job_store.update_status(job_id, JobStatus.SUCCESS.value)
    job = await job_store.get_job(job_id)

    expires_at = job_store.get_expires_at(job)
    assert expires_at is not None

    # Should expire 1 hour after updated_at
    assert job is not None
    updated_at = job.updated_at
    if updated_at.tzinfo is None:
        # Handle timezone-naive datetime from database
        from datetime import UTC
        updated_at = updated_at.replace(tzinfo=UTC)
    expected_expiry = updated_at + timedelta(seconds=3600)
    assert abs((expires_at - expected_expiry).total_seconds()) < 1


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_cleanup_expired_jobs_no_expired(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test cleanup when no jobs are expired."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    # Create some recent finished jobs
    job_id1 = await job_store._create_job()
    job_id2 = await job_store._create_job()

    await job_store.update_status(job_id1, JobStatus.SUCCESS.value)
    await job_store.update_status(job_id2, JobStatus.SUCCESS.value)

    # Run cleanup
    await job_store.cleanup_expired_jobs()

    # Verify jobs are still there and not marked as expired
    job1 = await job_store.get_job(job_id1)
    job2 = await job_store.get_job(job_id2)

    assert job1 is not None
    assert job2 is not None
    assert job1.is_expired is False
    assert job2.is_expired is False


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_cleanup_expired_jobs_with_output_files(db_engine: "AsyncEngine",
                                                      dask_scheduler_address: str,
                                                      tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch):
    """Test cleanup removes output files for expired jobs."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    with monkeypatch.context() as monkey_context:
        # Lower minimum expiry for testing
        monkey_context.setattr(JobStore, "MIN_EXPIRY", 1, raising=True)

        job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

        output_dir1 = tmp_path / "output_dir1"
        output_dir1.mkdir()

        output_dir2 = tmp_path / "output_dir2"
        output_dir2.mkdir()

        # Create jobs with very short expiry
        job_id1 = await job_store._create_job(expiry_seconds=1)
        job_id2 = await job_store._create_job(expiry_seconds=1)

        # Update to finished status with output paths
        await job_store.update_status(job_id1, JobStatus.SUCCESS, output_path=str(output_dir1))
        await job_store.update_status(job_id2, JobStatus.SUCCESS, output_path=str(output_dir2))

        # Verify files exist before cleanup
        assert output_dir1.exists()
        assert output_dir2.exists()

        # Wait for jobs to expire
        await asyncio.sleep(3)

        # Run cleanup
        await job_store.cleanup_expired_jobs()

        # Check that cleanup attempted to process the jobs
        job1 = await job_store.get_job(job_id1)
        job2 = await job_store.get_job(job_id2)

        assert job1 is not None
        assert job2 is not None

        assert job1.is_expired is True
        assert job2.is_expired is False  # Most recent job is kept

        assert not output_dir1.exists()
        assert output_dir2.exists()


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_cleanup_expired_jobs_keeps_active(db_engine: "AsyncEngine",
                                                 dask_scheduler_address: str,
                                                 monkeypatch: pytest.MonkeyPatch):
    """Test cleanup never expires active (running/submitted) jobs."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore

    with monkeypatch.context() as monkey_context:
        # Lower minimum expiry for testing
        monkey_context.setattr(JobStore, "MIN_EXPIRY", 1, raising=True)

        job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

        # Create jobs with very short expiry
        job_id1 = await job_store._create_job(expiry_seconds=1)
        job_id2 = await job_store._create_job(expiry_seconds=1)
        job_id3 = await job_store._create_job(expiry_seconds=1)

        # Keep one as submitted (active), update other to finished
        await job_store.update_status(job_id2, JobStatus.SUCCESS)
        await job_store.update_status(job_id3, JobStatus.SUCCESS)

        # Wait for expiry time to pass
        await asyncio.sleep(2)

        # Run cleanup
        await job_store.cleanup_expired_jobs()

        # Active job should never be expired
        job1 = await job_store.get_job(job_id1)
        job2 = await job_store.get_job(job_id2)
        job3 = await job_store.get_job(job_id3)

        assert job1 is not None
        assert job2 is not None
        assert job3 is not None
        assert job1.is_expired is False  # Active job should not be expired
        assert job2.is_expired  # Completed job should be expired
        assert job3.is_expired is False  # last job is not expired


def test_get_db_engine_with_url():
    """Test get_db_engine with provided URL."""
    from nat.front_ends.fastapi.job_store import get_db_engine

    db_url = "sqlite:///test.db"
    engine = get_db_engine(db_url, use_async=False)

    assert engine is not None
    assert str(engine.url) == db_url


def test_get_db_engine_async():
    """Test get_db_engine creates async engine."""
    from nat.front_ends.fastapi.job_store import get_db_engine

    db_url = "sqlite+aiosqlite:///test.db"
    engine = get_db_engine(db_url, use_async=True)

    assert engine is not None
    # AsyncEngine should have the async interface
    assert hasattr(engine, 'begin')


def test_get_db_engine_from_env_var(set_nat_job_store_db_url_env_var: str):
    """Test get_db_engine uses environment variable when no URL provided."""
    from nat.front_ends.fastapi.job_store import get_db_engine

    engine = get_db_engine(use_async=True)

    assert engine is not None
    # Should use the URL from environment variable
    assert str(engine.url) == set_nat_job_store_db_url_env_var


def test_get_db_engine_creates_default_sqlite():
    """Test get_db_engine creates default SQLite when no URL provided."""
    from nat.front_ends.fastapi.job_store import get_db_engine

    # Temporarily clear the environment variable
    original_url = os.environ.get("NAT_JOB_STORE_DB_URL")
    if original_url:
        del os.environ["NAT_JOB_STORE_DB_URL"]

    try:
        engine = get_db_engine(use_async=True)
        assert engine is not None

        # Should create a SQLite database in .tmp directory
        assert "sqlite" in str(engine.url)
        assert ".tmp/job_store.db" in str(engine.url)

    finally:
        # Restore environment variable
        if original_url:
            os.environ["NAT_JOB_STORE_DB_URL"] = original_url


@pytest.mark.asyncio
async def test_client_context_manager(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test the client context manager works correctly."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    async with job_store.client() as client:
        assert client is not None
        # Should be connected to the correct scheduler
        assert client.scheduler.address == dask_scheduler_address


@pytest.mark.usefixtures("setup_db")
@pytest.mark.asyncio
async def test_session_context_manager(db_engine: "AsyncEngine", dask_scheduler_address: str):
    """Test the session context manager works correctly."""
    from nat.front_ends.fastapi.job_store import JobStore

    job_store = JobStore(scheduler_address=dask_scheduler_address, db_engine=db_engine)

    async with job_store.session() as session:
        assert session is not None
        # Should be able to execute queries
        from sqlalchemy import text
        result = await session.execute(text("SELECT 1"))
        assert result is not None
