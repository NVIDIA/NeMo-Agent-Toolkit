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
"""Event schemas for NAT CLI telemetry.

The base :class:`TelemetryEvent` enforces that every concrete subclass declares
an ``_event_name`` ClassVar (validated at subclass-creation time so mistakes
fail before the first instance is built). All event payloads use camelCase
wire aliases while keeping snake_case Python attribute names.
"""
from __future__ import annotations

from enum import Enum
from typing import Any
from typing import ClassVar

from pydantic import BaseModel
from pydantic import Field

from nat.utils.telemetry.config import DEPLOYMENT_TYPE
from nat.utils.telemetry.config import DeploymentTypeEnum


class TaskStatusEnum(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"
    UNDEFINED = "undefined"


class TelemetryEvent(BaseModel):
    """Base class for all NAT telemetry events.

    Subclasses must set ``_event_name`` as a ClassVar. Attempting to define a
    subclass without it raises ``TypeError`` at class-creation time.
    """

    _event_name: ClassVar[str]
    _schema_version: ClassVar[str] = "1.0"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "_event_name" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define '_event_name' class variable")


class CliCommandEvent(TelemetryEvent):
    """Single invocation of a top-level NAT CLI command (e.g. ``nat run``).

    Privacy: this schema is deliberately minimal. It must never carry command
    arguments, option values, file paths, config contents, workflow/function
    names, hostnames, usernames, or any other user-supplied strings. The only
    free-form string is ``error_class``, which is the exception class name on
    failure (never the message).
    """

    _event_name: ClassVar[str] = "nat_cli_command"

    command: str = Field(
        ...,
        description="Top-level CLI command name (e.g. 'run', 'serve', 'evaluate').",
    )
    subcommand: str | None = Field(
        default=None,
        description="Second-level command name when applicable (e.g. 'list-components' for 'nat info list-components').",
    )
    task_status: TaskStatusEnum = Field(
        ...,
        alias="taskStatus",
        description="Outcome of the invocation.",
    )
    deployment_type: DeploymentTypeEnum = Field(
        default=DEPLOYMENT_TYPE,
        alias="deploymentType",
        description="Deployment context the event came from.",
    )
    duration_ms: int = Field(
        default=-1,
        alias="durationMs",
        description="Wall-clock duration of the invocation in milliseconds. -1 if unknown.",
        ge=-1,
    )
    exit_code: int = Field(
        default=-1,
        alias="exitCode",
        description="Process exit code (0 on success, 130 on interrupt, non-zero on failure). -1 if unknown.",
    )
    error_class: str | None = Field(
        default=None,
        alias="errorClass",
        description="Exception class name on failure (no message). Null on success or interrupt.",
    )
    python_version: str = Field(
        ...,
        alias="pythonVersion",
        description="The runtime Python version, e.g. '3.11.7'.",
    )

    model_config = {"populate_by_name": True}
