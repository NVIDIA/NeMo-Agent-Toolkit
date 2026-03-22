# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Pydantic models for the tool failure evaluator reasoning output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field


class _ToolCall(BaseModel):
    """A single invocation of a tool."""

    input: Any = Field(default=None, description="Arguments passed to the tool.")
    output: Any | None = Field(default=None, description="Return value from the tool.")
    error: str | None = Field(default=None, description="Error string if failed, None if succeeded.")


class _ToolSummary(BaseModel):
    """Complete health and attempt data for a single tool."""

    tool_name: str = Field(description="Name of the tool.")
    total_calls: int = Field(default=0, description="Total number of calls to this tool.")
    failed_calls: int = Field(default=0, description="Number of calls that returned an error.")
    failed_attempts: list[_ToolCall] = Field(
        default_factory=list,
        description="Ordered list of failed calls to this tool.",
    )


class _ToolFailureReasoning(BaseModel):
    """Complete reasoning payload returned by the tool failure evaluator."""

    total_tool_calls: int = Field(default=0, description="Total tool calls in the trajectory.")
    failed_tool_calls: int = Field(default=0, description="Total tool calls that errored.")
    failed_tools: list[str] | None = Field(default=None, description="Names of tools that had at least one failure.")
    score: float = Field(default=1.0, description="Overall success rate (0.0-1.0).")
    per_tool_summary: list[_ToolSummary] | None = Field(
        default=None,
        description="Per-tool health summary with attempt history.",
    )
