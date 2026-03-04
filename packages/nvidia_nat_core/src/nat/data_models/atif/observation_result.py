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
"""Observation result model for ATIF trajectories."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from nat.data_models.atif.content import ContentPart
from nat.data_models.atif.subagent_trajectory_ref import SubagentTrajectoryRef


class ObservationResult(BaseModel):
    """A single result within an observation."""

    source_call_id: str | None = Field(
        default=None,
        description=("The tool_call_id from the tool_calls array that this result corresponds to. "
                     "If null or omitted, the result comes from an action that doesn't use the "
                     "standard tool calling format."),
    )
    content: str | list[ContentPart] | None = Field(
        default=None,
        description=("The output or result from the tool execution. String for text-only "
                     "content, or array of ContentPart for multimodal content (ATIF v1.6+)."),
    )
    subagent_trajectory_ref: list[SubagentTrajectoryRef] | None = Field(
        default=None,
        description="Array of references to delegated subagent trajectories",
    )

    model_config = ConfigDict(extra="forbid")
