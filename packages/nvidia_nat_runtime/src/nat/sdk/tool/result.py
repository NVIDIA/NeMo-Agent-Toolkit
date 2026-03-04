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

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from nat.data_models.workspace import ActionResult


@dataclass
class ToolResult:
    """The outcome of a tool execution."""

    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_action_result(result: ActionResult) -> ToolResult:
        """Convert a core :class:`~nat.data_models.workspace.ActionResult` to a :class:`ToolResult`."""
        from nat.data_models.workspace import ActionStatus

        if result.status != ActionStatus.SUCCESS:
            return ToolResult(error=result.error_message or f"Action failed with status {result.status}")
        return ToolResult(output=result.output)

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def __str__(self) -> str:
        if self.is_error:
            return f"Error: {self.error}"
        return str(self.output) if self.output is not None else ""
