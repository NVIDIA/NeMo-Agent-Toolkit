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
"""Async job execution utilities for FastAPI front end."""

from nat.front_ends.fastapi.async_jobs.async_job import periodic_cleanup
from nat.front_ends.fastapi.async_jobs.async_job import run_generation
from nat.front_ends.fastapi.async_jobs.async_job import setup_worker
from nat.front_ends.fastapi.async_jobs.dask_client_mixin import DaskClientMixin
from nat.front_ends.fastapi.async_jobs.job_store import JobInfo
from nat.front_ends.fastapi.async_jobs.job_store import JobStatus
from nat.front_ends.fastapi.async_jobs.job_store import JobStore
from nat.front_ends.fastapi.async_jobs.job_store import get_db_engine

__all__ = [
    "setup_worker",
    "periodic_cleanup",
    "run_generation",
    "DaskClientMixin",
    "JobInfo",
    "JobStatus",
    "JobStore",
    "get_db_engine",
]
