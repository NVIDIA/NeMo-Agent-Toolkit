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
    span_event_timestamp: float | None = Field(
        default=None,
        description=("Start timestamp of the span for an END event. For step-level ancestry this is the step span "
                     "start; for tool ancestry entries this is the tool span start."),
    )
    framework: str | None = Field(
        default=None,
        description="Optional LLM framework identifier (for example, `langchain`).",
    )


class AtifStepExtra(BaseModel):
    """Validated structure for NAT-owned ATIF ``Step.extra`` payload."""

    model_config = ConfigDict(extra="allow")

    ancestry: AtifAncestry = Field(
        ...,
        description="Required step-level ancestry metadata for ATIF processing.",
    )
    tool_ancestry: list[AtifAncestry] = Field(
        default_factory=list,
        description=("Optional per-tool ancestry metadata aligned by index with `tool_calls` when a single agent "
                     "step contains multiple tool calls."),
    )
