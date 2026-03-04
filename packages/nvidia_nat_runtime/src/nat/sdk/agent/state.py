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

from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class AgentStatus(StrEnum):
    """Runtime status of an agent within a conversation."""

    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    MAX_ITERATIONS = "max_iterations"


class AgentState(BaseModel):
    """Mutable runtime state for a single agent invocation.

    This is intentionally separated from :class:`~nat.sdk.agent.agent.Agent`
    (which is frozen/immutable configuration) so that the same agent
    definition can be reused across multiple conversations.
    """

    model_config = ConfigDict(extra="allow")

    iteration: int = Field(default=0)
    status: AgentStatus = Field(default=AgentStatus.IDLE)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def reset(self) -> None:
        """Reset state for a new run."""
        self.iteration = 0
        self.status = AgentStatus.IDLE
        self.metadata.clear()
