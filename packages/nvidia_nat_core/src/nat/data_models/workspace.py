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

import datetime
import typing
import uuid
from abc import ABC
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from nat.data_models.common import BaseModelRegistryTag
from nat.data_models.common import TypedBaseModel
from nat.data_models.component_ref import WorkspaceGuardrailRef


@dataclass(frozen=True)
class ActionContext:
    """Context provided to workspace actions."""

    session_id: str
    root_path: Path


class WorkspaceBaseConfig(TypedBaseModel, BaseModelRegistryTag, ABC):
    """Base configuration for workspaces."""
    working_directory: Path = Field(description="The working directory for the workspace.")
    initial_commands: list[str] | Path | str = Field(
        description="The initial commands to run in the workspace. Assumes the existence of a shell workspace tool")
    workspace_guardrails: list[WorkspaceGuardrailRef] = Field(
        default_factory=list,
        description="Workspace guardrails to apply to workspace actions.",
    )
    endpoint: str | None = Field(
        default=None,
        description=
        "endpoint override to use. Should likely not be used in end-user configurations " + \
        "due to this only being needed for workspace propagation across agents."
    )
    session_id: str | None = Field(
        default=None,
        description="Session identifier to reuse an existing workspace session.",
    )

    @property
    def skills_directory(self) -> Path:
        return self.working_directory / "skills"


WorkspaceBaseConfigT = typing.TypeVar("WorkspaceBaseConfigT", bound=WorkspaceBaseConfig)


class ActionStatus(StrEnum):
    """Status values for workspace action execution."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    BLOCKED_BY_GUARDRAIL = "BLOCKED_BY_GUARDRAIL"


class ActionRequest(BaseModel):
    """Structured request for a workspace action.

    Args:
        action_id: Unique identifier for this action instance.
        action_name: Name of the action to execute.
        arguments: Key-value arguments for the action.
        timestamp: Time the action request was generated.
        model_metadata: Optional metadata provided by the model layer.
    """

    model_config = ConfigDict(extra="forbid")

    action_id: uuid.UUID = Field(description="Unique identifier for this action instance.")
    action_name: str = Field(description="Name of the action to execute.")
    arguments: dict[str, typing.Any] = Field(default_factory=dict, description="Key-value action arguments.")
    timestamp: datetime.datetime = Field(description="Time the action request was generated.")
    model_metadata: dict[str, typing.Any] | None = Field(
        default=None,
        description="Optional metadata provided by the model layer.",
    )


class ActionResult(BaseModel):
    """Structured result for a workspace action.

    Args:
        action_id: Identifier of the corresponding action.
        status: Execution status for the action.
        output: Execution output or returned data.
        error_message: Error details for failed or blocked actions.
        execution_time: Time taken for action execution in seconds.
    """

    model_config = ConfigDict(extra="forbid")

    action_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Identifier of the corresponding action.")
    status: ActionStatus = Field(description="Execution status for the action.")
    output: typing.Any | None = Field(default=None, description="Execution output or returned data.")
    error_message: str | None = Field(
        default=None,
        description="Error details for failed or blocked actions.",
    )
    execution_time: float | None = Field(default=None, description="Time taken for action execution in seconds.")

    @model_validator(mode="after")
    def _validate_error_message(self) -> "ActionResult":
        if self.status in {ActionStatus.FAILURE, ActionStatus.BLOCKED_BY_GUARDRAIL} and not self.error_message:
            raise ValueError("error_message must be set for failed or blocked actions.")
        return self
