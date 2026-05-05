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

The wire shape mirrors the NeMo Usage Telemetry project schema (clientId
``184482118588404``). Fields use sentinel values for unknowns (``-1`` for ints,
``"undefined"`` for strings) rather than nullable types, matching the schema's
``additionalProperties: false`` + all-required-fields convention.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any
from typing import ClassVar

from pydantic import BaseModel
from pydantic import Field


class NemoSourceEnum(StrEnum):
    """The NeMo product that emitted the event. Discriminator across NeMo products
    sharing the NeMo Usage Telemetry schema.

    Values mirror the schema-1.5 ``NemoSourceEnum`` definition in
    ``nemo-telemetry/schemas/anonymous_events.json``. NAT itself only emits
    ``AGENT_TOOLKIT``; the other values exist so this enum is a faithful
    mirror of the published schema (e.g. tests can deserialize foreign
    payloads, and future upstream additions surface as diff conflicts here).
    """

    INFERENCE = "inference"
    AUDITOR = "auditor"
    DATADESIGNER = "datadesigner"
    EVALUATOR = "evaluator"
    GUARDRAILS = "guardrails"
    SAFE_SYNTHESIZER = "safe-synthesizer"
    ANONYMIZER = "anonymizer"
    AGENT_TOOLKIT = "agent_toolkit"
    UNDEFINED = "undefined"


class TaskStatusEnum(StrEnum):
    """Outcome of the task being reported.

    Values mirror the schema-1.5 ``TaskStatusEnum`` definition. NAT's
    ``CliCommandEvent`` only emits ``SUCCESS`` / ``FAILURE`` / ``INTERRUPTED``;
    the other values exist for schema-mirror parity with other NeMo products.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELED = "canceled"
    INTERRUPTED = "interrupted"
    UNDEFINED = "undefined"


class TelemetryEvent(BaseModel):
    """Base class for all NAT telemetry events.

    Subclasses must set ``_event_name`` as a ClassVar. Attempting to define a
    subclass without it raises ``TypeError`` at class-creation time.
    """

    _event_name: ClassVar[str]
    _schema_version: ClassVar[str] = "1.5"

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

    nemo_source: NemoSourceEnum = Field(
        default=NemoSourceEnum.AGENT_TOOLKIT,
        alias="nemoSource",
        description="The NeMo product that created the event. Always 'agent_toolkit' for this event type.",
    )
    command: str = Field(
        ...,
        description="Top-level CLI command name (e.g. 'run', 'serve', 'evaluate').",
    )
    subcommand: str = Field(
        default="undefined",
        description=("Second-level command name when applicable, "
                     "such as 'list-components' for 'nat info list-components'. "
                     "'undefined' when no subcommand was invoked."),
    )
    task_status: TaskStatusEnum = Field(
        ...,
        alias="taskStatus",
        description="Outcome of the invocation: 'success', 'failure', or 'interrupted' (Ctrl-C / KeyboardInterrupt).",
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
    error_class: str = Field(
        default="undefined",
        alias="errorClass",
        description="Exception class name on failure (no message). 'undefined' on success or interrupt.",
    )
    python_version: str = Field(
        ...,
        alias="pythonVersion",
        description="The runtime Python version, e.g. '3.11.7'.",
    )

    model_config = {"populate_by_name": True}
