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
"""ATIF v1.6 data models for trajectory interchange."""

from __future__ import annotations

import uuid
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

ATIF_VERSION = "ATIF-v1.6"


class ImageSource(BaseModel):
    """Reference to an image stored alongside the trajectory."""

    model_config = ConfigDict(extra="forbid")

    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    path: str


class ContentPart(BaseModel):
    """A part of multimodal content — either text or an image."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image"]
    text: str | None = None
    source: ImageSource | None = None


class ATIFToolCall(BaseModel):
    """A single tool/function invocation by the agent."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    function_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class SubagentTrajectoryRef(BaseModel):
    """Reference to a delegated subagent trajectory."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    trajectory_path: str | None = None
    extra: dict[str, Any] | None = None


class ATIFObservationResult(BaseModel):
    """Result from a single tool call or action."""

    model_config = ConfigDict(extra="forbid")

    source_call_id: str | None = None
    content: str | list[ContentPart] | None = None
    subagent_trajectory_ref: list[SubagentTrajectoryRef] | None = None


class ATIFObservation(BaseModel):
    """Environment feedback after tool calls or system events."""

    model_config = ConfigDict(extra="forbid")

    results: list[ATIFObservationResult] = Field(default_factory=list)


class ATIFStepMetrics(BaseModel):
    """LLM operational metrics for a single step."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None


class ATIFFinalMetrics(BaseModel):
    """Aggregate metrics for an entire trajectory."""

    model_config = ConfigDict(extra="forbid")

    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int | None = None
    extra: dict[str, Any] | None = None


class ATIFAgentConfig(BaseModel):
    """Agent system identification and configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class ATIFStep(BaseModel):
    """A single step in an ATIF trajectory."""

    model_config = ConfigDict(extra="forbid")

    step_id: int
    source: Literal["system", "user", "agent"]
    message: str | list[ContentPart] = ""
    timestamp: str | None = None
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    reasoning_content: str | None = None
    tool_calls: list[ATIFToolCall] | None = None
    observation: ATIFObservation | None = None
    metrics: ATIFStepMetrics | None = None
    extra: dict[str, Any] | None = None


class ATIFTrajectory(BaseModel):
    """ATIF v1.6 trajectory — the complete interaction history of an agent run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = ATIF_VERSION
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent: ATIFAgentConfig
    steps: list[ATIFStep] = Field(default_factory=list)
    notes: str | None = None
    final_metrics: ATIFFinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    extra: dict[str, Any] | None = None
