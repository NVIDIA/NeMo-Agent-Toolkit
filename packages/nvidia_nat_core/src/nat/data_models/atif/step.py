# SPDX-FileCopyrightText: Copyright (c) 2025, Harbor Framework Contributors (https://github.com/harbor-framework/harbor)
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
"""Step model for ATIF trajectories."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from nat.data_models.atif.content import ContentPart
from nat.data_models.atif.metrics import Metrics
from nat.data_models.atif.observation import Observation
from nat.data_models.atif.tool_call import ToolCall


class Step(BaseModel):
    """A single step in an ATIF trajectory."""

    step_id: int = Field(
        ...,
        ge=1,
        description="Ordinal index of the turn (starting from 1)",
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO 8601 timestamp indicating when this step occurred",
    )
    source: Literal["system", "user", "agent"] = Field(
        ...,
        description="The originator of this step",
    )
    model_name: str | None = Field(
        default=None,
        description=("The specific LLM model used for this turn. Omission implies the model "
                     "defined in the root-level agent config."),
    )
    reasoning_effort: str | float | None = Field(
        default=None,
        description="Qualitative or quantitative measure of effort",
    )
    # NAT deviation: defaults to "" so the converter can create steps without
    # explicitly passing a message.  Harbor upstream requires this field.
    message: str | list[ContentPart] = Field(
        default="",
        description=("The dialogue message. String for text-only content, or array of "
                     "ContentPart for multimodal content (ATIF v1.6+)."),
    )
    reasoning_content: str | None = Field(
        default=None,
        description="The agent's explicit internal reasoning",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="Array of structured objects for the agent's actions",
    )
    observation: Observation | None = Field(
        default=None,
        description="Environment feedback/result after actions or system events",
    )
    metrics: Metrics | None = Field(
        default=None,
        description="LLM operational and confidence data for this step",
    )
    is_copied_context: bool | None = Field(
        default=None,
        description=("Indicates whether this step was copied from a previous trajectory "
                     "for context (e.g., during continuation after summarization). "
                     "Steps marked as copied context should not be included in training data."),
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom step-level metadata",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str | None) -> str | None:
        """Validate that timestamp is a valid ISO 8601 string."""
        if v is not None:
            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError as e:
                raise ValueError(f"Invalid ISO 8601 timestamp: {e}") from e
        return v

    @model_validator(mode="after")
    def validate_agent_only_fields(self) -> Step:
        """Validate that certain fields are only present for agent steps."""
        if self.source != "agent":
            agent_only_fields = [
                "model_name",
                "reasoning_effort",
                "reasoning_content",
                "tool_calls",
                "metrics",
            ]
            for field in agent_only_fields:
                if getattr(self, field) is not None:
                    raise ValueError(f"Field '{field}' is only applicable when source is 'agent', "
                                     f"but source is '{self.source}'")
        return self
