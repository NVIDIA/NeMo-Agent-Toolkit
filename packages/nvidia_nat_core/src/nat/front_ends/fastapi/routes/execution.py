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
"""Interactive execution route registration."""

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response

from nat.data_models.interactive_http import ExecutionAcceptedInteraction
from nat.data_models.interactive_http import ExecutionAcceptedOAuth
from nat.data_models.interactive_http import ExecutionCompletedStatus
from nat.data_models.interactive_http import ExecutionFailedStatus
from nat.data_models.interactive_http import ExecutionInteractionRequiredStatus
from nat.data_models.interactive_http import ExecutionOAuthRequiredStatus
from nat.data_models.interactive_http import ExecutionRunningStatus
from nat.data_models.interactive_http import ExecutionStatus
from nat.data_models.interactive_http import ExecutionStatusResponse
from nat.data_models.interactive_http import InteractionResponseRequest

if TYPE_CHECKING:
    from nat.front_ends.fastapi.execution_store import ExecutionRecord
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

logger = logging.getLogger(__name__)


def build_accepted_response(record: "ExecutionRecord") -> ExecutionAcceptedInteraction | ExecutionAcceptedOAuth:
    """Build a 202 accepted response from an interactive execution record."""
    status_url = f"/executions/{record.execution_id}"

    if record.status == ExecutionStatus.INTERACTION_REQUIRED and record.pending_interaction is not None:
        return ExecutionAcceptedInteraction(
            execution_id=record.execution_id,
            status_url=status_url,
            interaction_id=record.pending_interaction.interaction_id,
            prompt=record.pending_interaction.prompt,
            response_url=(f"/executions/{record.execution_id}"
                          f"/interactions/{record.pending_interaction.interaction_id}/response"),
        )
    if record.status == ExecutionStatus.OAUTH_REQUIRED and record.pending_oauth is not None:
        return ExecutionAcceptedOAuth(
            execution_id=record.execution_id,
            status_url=status_url,
            auth_url=record.pending_oauth.auth_url,
            oauth_state=record.pending_oauth.oauth_state,
        )

    raise ValueError(f"Cannot build 202 response for execution status: {record.status}")


async def add_execution_routes(worker: "FastApiFrontEndPluginWorker", app: FastAPI):
    """Add HTTP interactive execution endpoints (HITL + OAuth polling)."""
    execution_store = worker._execution_store
    execution_oauth_required_status_model = cast(Any, ExecutionOAuthRequiredStatus)

    async def get_execution_status(execution_id: str):
        """Get the status of an interactive execution."""
        record = await execution_store.get(execution_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

        if record.status == ExecutionStatus.COMPLETED:
            return ExecutionCompletedStatus(
                execution_id=record.execution_id,
                result=record.result,
            )
        if record.status == ExecutionStatus.FAILED:
            return ExecutionFailedStatus(
                execution_id=record.execution_id,
                error=record.error or "Unknown error",
            )
        if record.status == ExecutionStatus.INTERACTION_REQUIRED and record.pending_interaction is not None:
            return ExecutionInteractionRequiredStatus(
                execution_id=record.execution_id,
                interaction_id=record.pending_interaction.interaction_id,
                prompt=record.pending_interaction.prompt,
                response_url=(f"/executions/{execution_id}"
                              f"/interactions/{record.pending_interaction.interaction_id}/response"),
            )
        if record.status == ExecutionStatus.OAUTH_REQUIRED and record.pending_oauth is not None:
            return execution_oauth_required_status_model(
                execution_id=record.execution_id,
                auth_url=record.pending_oauth.auth_url,
                oauth_state=record.pending_oauth.oauth_state,
            )
        if record.status == ExecutionStatus.RUNNING:
            return ExecutionRunningStatus(execution_id=record.execution_id)

        raise ValueError(f"Cannot build status response for execution status: {record.status}")

    async def post_interaction_response(
        execution_id: str,
        interaction_id: str,
        body: InteractionResponseRequest,
    ):
        """Submit a human response to a pending interaction."""
        try:
            await execution_store.resolve_interaction(
                execution_id=execution_id,
                interaction_id=interaction_id,
                response=body.response,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return Response(status_code=204)

    app.add_api_route(
        path="/executions/{execution_id}",
        endpoint=get_execution_status,
        methods=["GET"],
        response_model=ExecutionStatusResponse,
        description="Get the status of an interactive execution (HTTP HITL / OAuth).",
        responses={
            404: {
                "description": "Execution not found"
            },
        },
    )
    app.add_api_route(
        path="/executions/{execution_id}/interactions/{interaction_id}/response",
        endpoint=post_interaction_response,
        methods=["POST"],
        description="Submit a human response to a pending interaction prompt.",
        responses={
            204: {
                "description": "Response accepted"
            },
            400: {
                "description": "Interaction already resolved"
            },
            404: {
                "description": "Execution or interaction not found"
            },
        },
    )

    logger.info("Added HTTP interactive execution endpoints at /executions/...")
