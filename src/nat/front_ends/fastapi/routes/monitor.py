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
# Monitoring endpoints (/monitor/users).

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import FastAPI

from nat.runtime.metrics import PerUserMetricsCollector
from nat.runtime.metrics import PerUserMonitorResponse
from nat.runtime.metrics import PerUserResourceUsage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker


async def register_monitor_routes(worker: FastApiFrontEndPluginWorker, app: FastAPI) -> None:
    """Add the per-user monitoring endpoint to the FastAPI app."""
    # Check if monitoring is enabled in config
    if not worker._config.general.enable_per_user_monitoring:
        logger.debug("Per-user monitoring disabled, skipping /monitor/users endpoint")
        return

    async def get_per_user_metrics(user_id: str | None = None) -> PerUserMonitorResponse:
        """
        Get resource usage metrics for per-user workflows.

        Args:
            user_id: Optional user ID to filter metrics for a specific user

        Returns:
            PerUserMonitorResponse with metrics for all or specified users
        """
        # Collect metrics from all session managers that have per-user workflows
        all_users: list[PerUserResourceUsage] = []

        for session_manager in worker._session_managers:
            if not session_manager.is_workflow_per_user:
                continue

            collector = PerUserMetricsCollector(session_manager)

            if user_id is not None:
                # Filter for specific user
                user_metrics = await collector.collect_user_metrics(user_id)
                if user_metrics:
                    all_users.append(user_metrics)
            else:
                # Get all users
                response = await collector.collect_all_metrics()
                all_users.extend(response.users)

        return PerUserMonitorResponse(
            timestamp=datetime.now(),
            total_active_users=len(all_users),
            users=all_users,
        )

    # Register the monitoring endpoint
    worker._register_api_route(path="/monitor/users",
                               app=app,
                               endpoint=get_per_user_metrics,
                               methods=["GET"],
                               response_model=PerUserMonitorResponse,
                               description="Get resource usage metrics for per-user workflows",
                               tags=["Monitoring"],
                               responses={
                                   200: {
                                       "description": "Successfully retrieved per-user metrics",
                                       "content": {
                                           "application/json": {
                                               "example": {
                                                   "timestamp":
                                                       "2025-12-16T10:30:00Z",
                                                   "total_active_users":
                                                       2,
                                                   "users": [{
                                                       "user_id": "alice",
                                                       "session": {
                                                           "created_at": "2025-12-16T09:00:00Z",
                                                           "last_activity": "2025-12-16T10:29:55Z",
                                                           "ref_count": 1,
                                                           "is_active": True
                                                       },
                                                       "requests": {
                                                           "total_requests": 42,
                                                           "active_requests": 1,
                                                           "avg_latency_ms": 1250.5,
                                                           "error_count": 2
                                                       },
                                                       "memory": {
                                                           "per_user_functions_count": 2,
                                                           "per_user_function_groups_count": 1,
                                                           "exit_stack_size": 3
                                                       }
                                                   }]
                                               }
                                           }
                                       }
                                   },
                                   500: {
                                       "description": "Internal Server Error"
                                   }
                               })

    logger.info("Added per-user monitoring endpoint at /monitor/users")
