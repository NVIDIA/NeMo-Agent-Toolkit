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
"""Subagent trajectory reference model for ATIF trajectories."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class SubagentTrajectoryRef(BaseModel):
    """Reference to a delegated subagent trajectory."""

    session_id: str = Field(
        ...,
        description="The session ID of the delegated subagent trajectory",
    )
    trajectory_path: str | None = Field(
        default=None,
        description="Reference to the complete subagent trajectory file",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom metadata about the subagent execution",
    )

    model_config = ConfigDict(extra="forbid")
