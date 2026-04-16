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
"""Shared MCP tool result models used by NAT server and client paths."""

from typing import Any

from pydantic import BaseModel

from nat.atif import ATIFTrajectory  # type: ignore[reportMissingImports]


class StructuredToolContent(BaseModel):
    """Stable structured output contract for NAT FastMCP tools.

    Intent:
    - Keep a deterministic object shape for MCP `structuredContent`.
    - Preserve backward-compatible text in `result_text`.
    - Include native JSON payload in `result_json` only when available.
    """
    result_text: str
    result_json: dict[str, Any] | list[Any] | None = None


class AtifToolMeta(BaseModel):
    """ATIF runtime metadata attached to MCP ToolResult meta."""
    run_id: str
    schema_version: str
    trajectory: ATIFTrajectory | dict[str, Any]


class ToolRuntimeMeta(BaseModel):
    """Top-level runtime metadata envelope for MCP ToolResult."""
    atif: AtifToolMeta | None = None
