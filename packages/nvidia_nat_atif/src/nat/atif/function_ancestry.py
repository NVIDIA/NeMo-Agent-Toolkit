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
"""FunctionAncestry model for ATIF trajectories (v1.7)."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class FunctionAncestry(BaseModel):
    """Typed ancestry node in the workflow call graph (ATIF v1.7).

    Records a callable node's identity and its parent. Reused by
    ``Step.function_ancestry`` (step-level callable) and
    ``ToolCall.tool_ancestry`` (per-tool-call callable).
    """

    function_id: str = Field(
        ...,
        description="Unique identifier for the callable node, stable across invocations",
    )
    function_name: str = Field(
        ...,
        description="Human-readable callable name",
    )
    parent_id: str | None = Field(
        default=None,
        description="Unique identifier of the parent callable node; null for root-level",
    )
    parent_name: str | None = Field(
        default=None,
        description="Human-readable name of the parent callable; null when parent_id is null",
    )

    model_config = ConfigDict(extra="forbid")
