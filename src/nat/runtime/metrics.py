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
"""Per-user workflow resource usage monitoring models and collector."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

if TYPE_CHECKING:
    from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class PerUserSessionMetrics(BaseModel):
    """Session lifecycle metrics for a per-user workflow."""

    created_at: datetime = Field(description="When the per-user workflow was created")
    last_activity: datetime = Field(description="Last time the workflow was accessed")
    ref_count: int = Field(ge=0, description="Current number of active references (in-flight requests)")
    is_active: bool = Field(description="Whether the workflow is currently being used")


class PerUserRequestMetrics(BaseModel):
    """Request-level metrics for a per-user workflow."""

    total_requests: int = Field(ge=0, default=0, description="Total number of requests processed")
    active_requests: int = Field(ge=0, default=0, description="Number of currently active requests")
    avg_latency_ms: float = Field(ge=0, default=0.0, description="Average request latency in milliseconds")
    error_count: int = Field(ge=0, default=0, description="Total number of failed requests")


class PerUserLLMUsageMetrics(BaseModel):
    """LLM token usage metrics for a per-user workflow."""

    total_tokens: int = Field(ge=0, default=0, description="Total tokens used (prompt + completion)")
    prompt_tokens: int = Field(ge=0, default=0, description="Total prompt tokens used")
    completion_tokens: int = Field(ge=0, default=0, description="Total completion tokens used")
    llm_calls: int = Field(ge=0, default=0, description="Total number of LLM API calls")


class PerUserMemoryMetrics(BaseModel):
    """Memory/resource count metrics for a per-user workflow."""

    per_user_functions_count: int = Field(ge=0, default=0, description="Number of per-user functions built")
    per_user_function_groups_count: int = Field(ge=0, default=0, description="Number of per-user function groups built")
    exit_stack_size: int = Field(ge=0, default=0, description="Number of resources in the async exit stack")


class PerUserResourceUsage(BaseModel):
    """Combined resource usage metrics for a single per-user workflow."""

    user_id: str = Field(description="The user identifier")
    session: PerUserSessionMetrics = Field(description="Session lifecycle metrics")
    requests: PerUserRequestMetrics = Field(description="Request-level metrics")
    llm_usage: PerUserLLMUsageMetrics = Field(description="LLM token usage metrics")
    memory: PerUserMemoryMetrics = Field(description="Memory/resource count metrics")


class PerUserMonitorResponse(BaseModel):
    """Response model for the /monitor/users endpoint."""

    timestamp: datetime = Field(default_factory=datetime.now, description="When the metrics were collected")
    total_active_users: int = Field(ge=0, description="Number of users with active per-user workflows")
    users: list[PerUserResourceUsage] = Field(default_factory=list, description="Per-user resource usage details")


class PerUserMetricsCollector:
    """Collector for per-user workflow metrics.

    This class aggregates metrics from SessionManager's per-user builders
    and provides methods to collect metrics for individual users or all users.
    """

    def __init__(self, session_manager: SessionManager):
        """Initialize the collector with a SessionManager reference.

        Args:
            session_manager: The SessionManager instance to collect metrics from
        """
        self._session_manager = session_manager

    async def collect_user_metrics(self, user_id: str) -> PerUserResourceUsage | None:
        """Collect metrics for a specific user.

        Args:
            user_id: The user identifier to collect metrics for

        Returns:
            PerUserResourceUsage if user exists, None otherwise
        """
        async with self._session_manager._per_user_builders_lock:
            if user_id not in self._session_manager._per_user_builders:
                return None

            builder_info = self._session_manager._per_user_builders[user_id]
            return self._build_user_metrics(user_id, builder_info)

    async def collect_all_metrics(self) -> PerUserMonitorResponse:
        """Collect metrics for all active per-user workflows.

        Returns:
            PerUserMonitorResponse with all user metrics
        """
        users: list[PerUserResourceUsage] = []

        async with self._session_manager._per_user_builders_lock:
            for user_id, builder_info in self._session_manager._per_user_builders.items():
                try:
                    user_metrics = self._build_user_metrics(user_id, builder_info)
                    users.append(user_metrics)
                except Exception as e:
                    logger.warning("Failed to collect metrics for user %s: %s", user_id, e)

        return PerUserMonitorResponse(
            timestamp=datetime.now(),
            total_active_users=len(users),
            users=users,
        )

    def _build_user_metrics(self, user_id: str, builder_info) -> PerUserResourceUsage:
        """Build metrics for a single user from builder info.

        Args:
            user_id: The user identifier
            builder_info: The PerUserBuilderInfo instance

        Returns:
            PerUserResourceUsage with all metrics
        """
        # Session metrics
        session_metrics = PerUserSessionMetrics(
            created_at=getattr(builder_info, 'created_at', builder_info.last_activity),
            last_activity=builder_info.last_activity,
            ref_count=builder_info.ref_count,
            is_active=builder_info.ref_count > 0,
        )

        # Request metrics
        total_requests = getattr(builder_info, 'total_requests', 0)
        error_count = getattr(builder_info, 'error_count', 0)
        total_latency_ms = getattr(builder_info, 'total_latency_ms', 0.0)
        avg_latency = total_latency_ms / total_requests if total_requests > 0 else 0.0

        request_metrics = PerUserRequestMetrics(
            total_requests=total_requests,
            active_requests=builder_info.ref_count,
            avg_latency_ms=round(avg_latency, 2),
            error_count=error_count,
        )

        # LLM usage metrics
        llm_metrics = PerUserLLMUsageMetrics(
            total_tokens=getattr(builder_info, 'total_tokens', 0),
            prompt_tokens=getattr(builder_info, 'prompt_tokens', 0),
            completion_tokens=getattr(builder_info, 'completion_tokens', 0),
            llm_calls=getattr(builder_info, 'llm_calls', 0),
        )

        # Memory/resource count metrics from the builder
        builder = builder_info.builder
        per_user_functions_count = len(getattr(builder, '_per_user_functions', {}))
        per_user_function_groups_count = len(getattr(builder, '_per_user_function_groups', {}))

        # Count resources in exit stack (if accessible)
        exit_stack = getattr(builder, '_exit_stack', None)
        if exit_stack and hasattr(exit_stack, '_exit_callbacks'):
            exit_stack_size = len(exit_stack._exit_callbacks)
        else:
            exit_stack_size = 0

        memory_metrics = PerUserMemoryMetrics(
            per_user_functions_count=per_user_functions_count,
            per_user_function_groups_count=per_user_function_groups_count,
            exit_stack_size=exit_stack_size,
        )

        return PerUserResourceUsage(
            user_id=user_id,
            session=session_metrics,
            requests=request_metrics,
            llm_usage=llm_metrics,
            memory=memory_metrics,
        )
