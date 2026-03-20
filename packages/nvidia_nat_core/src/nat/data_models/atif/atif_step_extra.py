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

from nat.data_models.invocation_node import InvocationNode


class AtifAncestry(BaseModel):
    """Validated ancestry metadata embedded in ATIF ``Step.extra``."""

    model_config = ConfigDict(extra="forbid")

    function_ancestry: InvocationNode = Field(
        ...,
        description="Function ancestry for the event represented by this metadata entry.",
    )
    framework: str | None = Field(
        default=None,
        description="Optional LLM framework identifier (for example, `langchain`).",
    )


class AtifInvocationInfo(BaseModel):
    """Invocation timing metadata embedded in ATIF ``Step.extra``."""

    model_config = ConfigDict(extra="forbid")

    start_timestamp: float = Field(
        ...,
        description="Invocation start timestamp in epoch seconds.",
    )
    end_timestamp: float = Field(
        ...,
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

    # Experimental helper fields for migration, will be removed so don't use for any purpose other than troubleshooting
    step_ancestry_path: list[InvocationNode] | None = Field(
        default=None,
        min_length=1,
        description=("Optional derived root-to-leaf ancestry path for step context. "
                     "Provided as a convenience helper for consumers."),
    )
    tool_ancestry_paths: list[list[InvocationNode]] | None = Field(
        default=None,
        description=("Optional derived root-to-leaf ancestry path per tool call, aligned by index with "
                     "`tool_calls` when present. Provided as a convenience helper for consumers."),
    )
