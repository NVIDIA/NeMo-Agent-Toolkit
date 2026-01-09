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
    if configure_logging:
        logging.basicConfig(level=log_level)
    
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
    """
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore
    from nat.front_ends.fastapi.response_helpers import generate_single_response
    from nat.runtime.loader import load_workflow

    logger = _configure_logging(configure_logging, log_level)

    job_store = JobStore(scheduler_address=scheduler_address, db_url=db_url)
    try:
        async with load_workflow(config_file_path) as local_session_manager:
            async with local_session_manager.session() as session:
                result = await generate_single_response(payload,
                                                        session,
                                                        result_type=session.workflow.single_output_schema)

        del session
        del local_session_manager
        await job_store.update_status(job_id, JobStatus.SUCCESS, output=result)
    except Exception as e:
        logger.exception("Error in async job %s", job_id)
        await job_store.update_status(job_id, JobStatus.FAILURE, error=str(e))

    # Explicitly release the resources held by the job store
    del job_store


async def periodic_cleanup(*, scheduler_address: str,
                           db_url: str,
                           sleep_time_sec: int = 300,
                           configure_logging: bool = True,
                           log_level: int = logging.INFO):
    from nat.front_ends.fastapi.job_store import JobStore

    logger = _configure_logging(configure_logging, log_level)

    job_store = JobStore(scheduler_address=scheduler_address, db_url=db_url)

    logger.info("Starting periodic cleanup of expired jobs every %d seconds", sleep_time_sec)
    while True:
        await asyncio.sleep(sleep_time_sec)

        try:
            num_expired = await job_store.cleanup_expired_jobs()
            logger.info("Expired jobs cleaned up: %d", num_expired)
        except:  # noqa: E722
            logger.exception("Error during job cleanup")
