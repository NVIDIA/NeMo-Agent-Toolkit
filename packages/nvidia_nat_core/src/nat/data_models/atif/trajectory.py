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
"""Trajectory (root) model for ATIF (Agent Trajectory Interchange Format)."""

from __future__ import annotations

import uuid
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from nat.data_models.atif.agent import Agent
from nat.data_models.atif.final_metrics import FinalMetrics
from nat.data_models.atif.step import Step

ATIF_VERSION = "ATIF-v1.6"


class Trajectory(BaseModel):
    """ATIF trajectory — the complete interaction history of an agent run."""

    schema_version: Literal[
        "ATIF-v1.0",
        "ATIF-v1.1",
        "ATIF-v1.2",
        "ATIF-v1.3",
        "ATIF-v1.4",
        "ATIF-v1.5",
        "ATIF-v1.6",
    ] = Field(
        default=ATIF_VERSION,
        description="String defining ATIF compatibility",
    )
    # NAT deviation: defaults to a generated UUID so the converter can create
    # trajectories without an explicit session_id.  Harbor upstream requires it.
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the entire agent run",
    )
    agent: Agent = Field(
        ...,
        description="Object specifying the agent configuration",
    )
    # NAT deviation: allows an empty steps list (the batch converter returns an
    # empty trajectory for empty input).  Harbor upstream requires min_length=1.
    steps: list[Step] = Field(
        default_factory=list,
        description="Array of step objects representing the complete interaction history",
    )
    notes: str | None = Field(
        default=None,
        description="Custom information, design notes, or explanations",
    )
    final_metrics: FinalMetrics | None = Field(
        default=None,
        description="Summary metrics for the entire trajectory",
    )
    continued_trajectory_ref: str | None = Field(
        default=None,
        description="Reference to the continuation trajectory file if this trajectory is continued in another file",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom root-level metadata",
    )

    model_config = ConfigDict(extra="forbid")

    def to_json_dict(self, exclude_none: bool = True) -> dict[str, Any]:
        """Export trajectory to a dictionary suitable for JSON serialization."""
        return self.model_dump(exclude_none=exclude_none, mode="json")

    @model_validator(mode="after")
    def validate_step_ids(self) -> Trajectory:
        """Validate that step_ids are sequential starting from 1."""
        for i, step in enumerate(self.steps):
            expected_step_id = i + 1
            if step.step_id != expected_step_id:
                raise ValueError(f"steps[{i}].step_id: expected {expected_step_id} "
                                 f"(sequential from 1), got {step.step_id}")
        return self

    @model_validator(mode="after")
    def validate_tool_call_references(self) -> Trajectory:
        """Validate that observation source_call_ids reference valid tool_call_ids."""
        for step in self.steps:
            if step.observation is None:
                continue
            tool_call_ids = set()
            if step.tool_calls:
                tool_call_ids = {tc.tool_call_id for tc in step.tool_calls}
            for result in step.observation.results:
                if result.source_call_id is not None and result.source_call_id not in tool_call_ids:
                    raise ValueError(f"Observation result references source_call_id "
                                     f"'{result.source_call_id}' which is not found in "
                                     f"step {step.step_id}'s tool_calls")
        return self

    def has_multimodal_content(self) -> bool:
        """Check if any step contains multimodal content (images)."""
        for step in self.steps:
            if isinstance(step.message, list):
                for part in step.message:
                    if part.type == "image":
                        return True
            if step.observation:
                for result in step.observation.results:
                    if isinstance(result.content, list):
                        for part in result.content:
                            if part.type == "image":
                                return True
        return False
