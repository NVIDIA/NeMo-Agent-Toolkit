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

import uuid
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum

from nat.sdk.agent.state import AgentState
from nat.sdk.event.log import EventLog
from nat.sdk.llm.message import TokenUsage


class ConversationStatus(StrEnum):
    """Lifecycle status of a conversation."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class UsageStats:
    """Aggregate usage statistics for a conversation."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0

    def record(self, usage: TokenUsage | None) -> None:
        """Record usage from a single LLM call."""
        self.llm_calls += 1
        if usage:
            self.total_tokens += usage.total_tokens
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens


@dataclass
class ConversationState:
    """Mutable runtime state for a conversation.

    Owns the event log, agent state, and usage statistics.  Intentionally
    separated from :class:`~nat.sdk.conversation.conversation.Conversation`
    so that state can be inspected, serialised, or transferred independently.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: EventLog = field(default_factory=EventLog)
    agent_state: AgentState = field(default_factory=AgentState)
    status: ConversationStatus = ConversationStatus.ACTIVE
    stats: UsageStats = field(default_factory=UsageStats)
    activated_skills: list[str] = field(default_factory=list)
