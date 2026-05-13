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
"""
Data models for HTTP Human-in-the-Loop (HITL) and OAuth support.

These types power the execution + polling model that enables interactive
workflows over plain HTTP (no WebSocket required).
"""

import typing
from enum import StrEnum

from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import Field

from nat.data_models.api_server import ResponseSerializable
from nat.data_models.interactive import HumanPrompt
from nat.data_models.interactive import HumanResponse

# ---------------------------------------------------------------------------
# Execution status enum
# ---------------------------------------------------------------------------


class ExecutionStatus(StrEnum):
    """Status of an HTTP interactive execution."""
    RUNNING = "running"
    INTERACTION_REQUIRED = "interaction_required"
    OAUTH_REQUIRED = "oauth_required"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Execution status response – discriminated union (GET /executions/{id})
# ---------------------------------------------------------------------------


class _ExecutionStatusBase(BaseModel):
    """Common fields for every execution status variant."""
    execution_id: str = Field(description="Unique identifier for this execution.")


class ExecutionRunningStatus(_ExecutionStatusBase):
    """Execution is in progress, no interaction or result yet."""
    status: typing.Literal[ExecutionStatus.RUNNING] = ExecutionStatus.RUNNING


class ExecutionInteractionRequiredStatus(_ExecutionStatusBase):
    """Execution is paused waiting for a human interaction response."""
    status: typing.Literal[ExecutionStatus.INTERACTION_REQUIRED] = ExecutionStatus.INTERACTION_REQUIRED
    interaction_id: str = Field(description="Unique identifier for the pending interaction.")
    prompt: HumanPrompt = Field(description="The human prompt awaiting a response.")
    response_url: str = Field(description="URL to POST the HumanResponse to.")


class ExecutionOAuthRequiredStatus(_ExecutionStatusBase):
    """Execution is paused waiting for an OAuth consent flow to complete."""
    status: typing.Literal[ExecutionStatus.OAUTH_REQUIRED] = ExecutionStatus.OAUTH_REQUIRED
    auth_url: str = Field(description="OAuth authorization URL the client should open.")
    oauth_state: str = Field(description="OAuth state parameter associated with the flow.")


class ExecutionCompletedStatus(_ExecutionStatusBase):
    """Execution finished successfully."""
    status: typing.Literal[ExecutionStatus.COMPLETED] = ExecutionStatus.COMPLETED
    result: typing.Any = Field(description="Workflow result.")


class ExecutionFailedStatus(_ExecutionStatusBase):
    """Execution finished with an error."""
    status: typing.Literal[ExecutionStatus.FAILED] = ExecutionStatus.FAILED
    error: str = Field(description="Error message.")


ExecutionStatusResponse = typing.Annotated[
    ExecutionRunningStatus
    | ExecutionInteractionRequiredStatus
    | ExecutionOAuthRequiredStatus
    | ExecutionCompletedStatus
    | ExecutionFailedStatus,
    Discriminator("status"),
]

# ---------------------------------------------------------------------------
# 202 Accepted response body – discriminated union
# ---------------------------------------------------------------------------


class _ExecutionAcceptedBase(_ExecutionStatusBase):
    """Common fields for every 202 Accepted variant."""
    status_url: str = Field(description="URL to poll for execution status.")


class ExecutionAcceptedInteraction(_ExecutionAcceptedBase):
    """202 response when the execution requires human interaction."""
    status: typing.Literal[ExecutionStatus.INTERACTION_REQUIRED] = ExecutionStatus.INTERACTION_REQUIRED
    interaction_id: str = Field(description="Pending interaction id.")
    prompt: HumanPrompt = Field(description="The human prompt awaiting a response.")
    response_url: str = Field(description="URL to POST the HumanResponse to.")


class ExecutionAcceptedOAuth(_ExecutionAcceptedBase):
    """202 response when the execution requires OAuth consent."""
    status: typing.Literal[ExecutionStatus.OAUTH_REQUIRED] = ExecutionStatus.OAUTH_REQUIRED
    auth_url: str = Field(description="OAuth authorization URL.")
    oauth_state: str = Field(description="OAuth state parameter.")


ExecutionAcceptedResponse = typing.Annotated[
    ExecutionAcceptedInteraction | ExecutionAcceptedOAuth,
    Discriminator("status"),
]

# ---------------------------------------------------------------------------
# Interaction response request body
# ---------------------------------------------------------------------------


class InteractionResponseRequest(BaseModel):
    """
    Body for ``POST /executions/{execution_id}/interactions/{interaction_id}/response``.

    Uses the existing ``HumanResponse`` discriminated union so that all
    interaction types (text, binary, radio, checkbox, dropdown, notification)
    are supported without new types.
    """
    response: HumanResponse = Field(description="The human response to the interaction prompt.")


# ---------------------------------------------------------------------------
# SSE stream event types for streaming endpoints
# ---------------------------------------------------------------------------


class StreamInteractionEvent(BaseModel, ResponseSerializable):
    """
    SSE event emitted in a streaming response when the workflow requires
    human interaction (HITL).
    """
    event_type: typing.Literal["interaction_required"] = "interaction_required"
    execution_id: str = Field(description="Execution identifier.")
    interaction_id: str = Field(description="Interaction identifier.")
    prompt: HumanPrompt = Field(description="The human prompt awaiting a response.")
    response_url: str = Field(description="URL to POST the HumanResponse to.")

    def get_stream_data(self) -> str:
        return f"event: interaction_required\ndata: {self.model_dump_json()}\n\n"


class StreamOAuthEvent(BaseModel, ResponseSerializable):
    """
    SSE event emitted in a streaming response when the workflow requires
    OAuth authentication.
    """
    event_type: typing.Literal["oauth_required"] = "oauth_required"
    execution_id: str = Field(description="Execution identifier.")
    auth_url: str = Field(description="OAuth authorization URL.")
    oauth_state: str = Field(description="OAuth state parameter.")

    def get_stream_data(self) -> str:
        return f"event: oauth_required\ndata: {self.model_dump_json()}\n\n"
