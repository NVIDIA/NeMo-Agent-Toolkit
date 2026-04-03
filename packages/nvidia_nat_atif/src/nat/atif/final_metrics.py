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
"""Final (trajectory-level) metrics model for ATIF trajectories."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class FinalMetrics(BaseModel):
    """Aggregate statistics for the entire trajectory."""

    total_prompt_tokens: int | None = Field(
        default=None,
        description="Sum of all prompt tokens across all steps, including cached tokens",
    )
    total_completion_tokens: int | None = Field(
        default=None,
        description="Sum of all completion tokens across all steps",
    )
    total_cached_tokens: int | None = Field(
        default=None,
        description="Sum of all cached tokens across all steps",
    )
    total_cost_usd: float | None = Field(
        default=None,
        description="Total real monetary cost for the entire trajectory, including cost for subagents, if any",
    )
    total_steps: int | None = Field(
        default=None,
        ge=0,
        description=("Total number of steps. If not equivalent to the number of steps in the "
                     "trajectory, must be documented in the root-level notes field."),
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom aggregate metrics",
    )

    model_config = ConfigDict(extra="forbid")
