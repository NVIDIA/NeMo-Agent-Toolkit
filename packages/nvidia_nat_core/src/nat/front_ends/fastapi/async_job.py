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
"""
The functions in this module are intentionally written to be submitted as Dask tasks, as such they are self-contained.
"""

import asyncio
import logging
import typing


def _configure_logging(configure_logging: bool, log_level: int) -> logging.Logger:
    from nat.utils.log_utils import setup_logging
    if configure_logging:
        setup_logging(log_level)

    return logging.getLogger(__name__)


async def run_generation(configure_logging: bool,
                         log_level: int,
                         scheduler_address: str,
                         db_url: str,
                         config_file_path: str,
                         job_id: str,
                         payload: typing.Any):
    """
    Background async task to run the workflow.

    Parameters
    ----------
    configure_logging : bool
        Whether to configure logging.
    log_level : int
        The log level to use when `configure_logging` is `True`, ignored otherwise.
    scheduler_address : str
        The Dask scheduler address.
    db_url : str
        The database URL for the job store.
    config_file_path : str
        The path to the workflow configuration file.
    job_id : str
        The job ID.
    payload : typing.Any
        The input payload for the workflow.
    """
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore
    from nat.front_ends.fastapi.response_helpers import generate_single_response
    from nat.runtime.loader import load_workflow

    logger = _configure_logging(configure_logging, log_level)

    job_store = None
    try:
        job_store = JobStore(scheduler_address=scheduler_address, db_url=db_url)
        await job_store.update_status(job_id, JobStatus.RUNNING)
        async with load_workflow(config_file_path) as local_session_manager:
            async with local_session_manager.session() as session:
                result = await generate_single_response(payload,
                                                        session,
                                                        result_type=session.workflow.single_output_schema)

        await job_store.update_status(job_id, JobStatus.SUCCESS, output=result)
    except asyncio.CancelledError:
        logger.info("Async job %s cancelled", job_id)
        if job_store is not None:
            await job_store.update_status(job_id, JobStatus.INTERRUPTED, error="cancelled")
    except Exception as e:
        logger.exception("Error in async job %s", job_id)
        if job_store is not None:
            await job_store.update_status(job_id, JobStatus.FAILURE, error=str(e))


async def periodic_cleanup(*,
                           scheduler_address: str,
                           db_url: str,
                           sleep_time_sec: int = 300,
                           configure_logging: bool = True,
                           log_level: int = logging.INFO):
    """
    Dask task to periodically clean up expired jobs from the job store. This task is intended to be submitted only
    once to the Dask cluster and run indefinitely.

    Parameters
    ----------
    scheduler_address : str
        The Dask scheduler address.
    db_url : str
        The database URL for the job store.
    sleep_time_sec : int
        The sleep time between cleanup operations in seconds.
    configure_logging : bool
        Whether to configure logging.
    log_level : int
        The log level to use when `configure_logging` is `True`, ignored otherwise.
    """
    from nat.front_ends.fastapi.job_store import JobStore

    logger = _configure_logging(configure_logging, log_level)

    job_store = None

    logger.info("Starting periodic cleanup of expired jobs every %d seconds", sleep_time_sec)
    while True:
        await asyncio.sleep(sleep_time_sec)

        try:
            if job_store is None:
                job_store = JobStore(scheduler_address=scheduler_address, db_url=db_url)

            num_expired = await job_store.cleanup_expired_jobs()
            logger.info("Expired jobs cleaned up: %d", num_expired)
        except:  # noqa: E722
            logger.exception("Error during job cleanup")
            job_store = None  # Reset job store to attempt re-creation on next iteration
