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

from datetime import datetime
from uuid import uuid4


class JobStatus(str):
    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"
    NOT_FOUND = "not_found"


class JobStore:

    def __init__(self):
        self._jobs = {}

    def create_job(self, config_file: str) -> str:
        job_id = str(uuid4())
        self._jobs[job_id] = {
            "status": JobStatus.SUBMITTED,
            "config_file": config_file,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "error": None,
            "output_path": None,
        }
        return job_id

    def update_status(self, job_id: str, status: str, error: str | None = None, output_path: str | None = None):
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = status
            self._jobs[job_id]["updated_at"] = datetime.utcnow()
            if error:
                self._jobs[job_id]["error"] = error
            if output_path:
                self._jobs[job_id]["output_path"] = output_path

    def get_status(self, job_id: str):
        return self._jobs.get(job_id, {"status": JobStatus.NOT_FOUND})

    def list_jobs(self):
        return self._jobs
