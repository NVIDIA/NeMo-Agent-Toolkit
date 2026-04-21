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
"""Typed models for NAT metadata inside ATIF ``Step.extra``."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


class AtifAncestry(BaseModel):
    """Validated ancestry metadata embedded in ATIF ``Step.extra``."""

    model_config = ConfigDict(extra="forbid")

    function_id: str = Field(
        ...,
        description="Unique identifier for the callable node.",
    )
    function_name: str = Field(
        ...,
        description="Name of the callable node.",
    )
    parent_id: str | None = Field(
        default=None,
        description="Optional parent callable identifier.",
    )
    parent_name: str | None = Field(
        default=None,
        description="Optional parent callable name.",
    )


class AtifInvocationInfo(BaseModel):
    """Invocation timing metadata embedded in ATIF ``Step.extra``."""

    model_config = ConfigDict(extra="forbid")

    start_timestamp: float | None = Field(
        default=None,
        description="Invocation start timestamp in epoch seconds.",
    )
    end_timestamp: float | None = Field(
        default=None,
        description="Invocation end timestamp in epoch seconds.",
    )
    invocation_id: str | None = Field(
        default=None,
        description=("Optional stable invocation identifier for correlation (for example, "
                     "`tool_call_id` for tool invocations)."),
    )
    status: str | None = Field(
        default=None,
        description="Optional terminal status for the invocation (for example, `completed`, `error`).",
    )
    framework: str | None = Field(
        default=None,
        description="Optional LLM framework identifier (for example, `langchain`).",
    )

    @model_validator(mode="after")
    def validate_timestamp_pairing(self) -> AtifInvocationInfo:
        has_start = self.start_timestamp is not None
        has_end = self.end_timestamp is not None
        if has_start != has_end:
            raise ValueError("start_timestamp and end_timestamp must both be set, or both be null")
        return self


class AtifStepExtra(BaseModel):
    """Validated structure for NAT-owned ATIF ``Step.extra`` payload."""

    model_config = ConfigDict(extra="allow")

    ancestry: AtifAncestry = Field(
        ...,
        description="Step-level ancestry metadata for ATIF step context.",
    )
    invocation: AtifInvocationInfo | None = Field(
        default=None,
        description="Optional step-level invocation timing metadata.",
    )
    tool_ancestry: list[AtifAncestry] = Field(
        default_factory=list,
        description=("Per-tool ancestry metadata aligned by index with `tool_calls` for observed invocation "
                     "lineage reconstruction."),
    )
    tool_invocations: list[AtifInvocationInfo] | None = Field(
        default=None,
        description=("Optional per-tool invocation timing metadata aligned by index with `tool_calls` when "
                     "present."),
    )
