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

import logging
import typing

async def run_generation(scheduler_address: str,
                         db_url: str,
                         config_file_path: str,
                         job_id: str,
                         payload: typing.Any):
    """
    Background async task to run the workflow.
    
    Note: this function is intentionally written in it's own module such that it is packaged easily in Dask
    """
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore
    from nat.front_ends.fastapi.response_helpers import generate_single_response
    from nat.runtime.loader import load_workflow

    logger = logging.getLogger(__name__)

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
