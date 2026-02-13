# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monitoring route registration."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import FastAPI

from nat.runtime.metrics import PerUserMetricsCollector
from nat.runtime.metrics import PerUserMonitorResponse
from nat.runtime.metrics import PerUserResourceUsage

if TYPE_CHECKING:
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

logger = logging.getLogger(__name__)


async def add_monitor_route(worker: "FastApiFrontEndPluginWorker", app: FastAPI):
    """Add per-user monitoring endpoint when enabled."""
    if not worker._config.general.enable_per_user_monitoring:
        logger.debug("Per-user monitoring disabled, skipping /monitor/users endpoint")
        return

    async def get_per_user_metrics(user_id: str | None = None) -> PerUserMonitorResponse:
        """Get resource usage metrics for per-user workflows."""
        all_users: list[PerUserResourceUsage] = []

        for session_manager in worker._session_managers:
            if not session_manager.is_workflow_per_user:
                continue

            collector = PerUserMetricsCollector(session_manager)
            if user_id is not None:
                user_metrics = await collector.collect_user_metrics(user_id)
                if user_metrics:
                    all_users.append(user_metrics)
            else:
                response = await collector.collect_all_metrics()
                all_users.extend(response.users)

        return PerUserMonitorResponse(
            timestamp=datetime.now(),
            total_active_users=len(all_users),
            users=all_users,
        )

    app.add_api_route(path="/monitor/users",
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

