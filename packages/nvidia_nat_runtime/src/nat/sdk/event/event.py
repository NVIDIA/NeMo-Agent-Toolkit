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
from datetime import UTC
from datetime import datetime
from enum import StrEnum
from typing import Any


class EventSource(StrEnum):
    """Origin of an event in the conversation."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


def _make_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Event:
    """Base event in a conversation timeline.

    All events are immutable and carry a unique id plus a UTC timestamp.
    """

    id: str = field(default_factory=_make_id)
    timestamp: datetime = field(default_factory=_now)
    source: EventSource = EventSource.SYSTEM


@dataclass(frozen=True)
class SystemPromptEvent(Event):
    """The system prompt injected at the start of a conversation."""

    source: EventSource = field(default=EventSource.SYSTEM, init=False)
    content: str = ""


@dataclass(frozen=True)
class MessageEvent(Event):
    """A text message from the user or the agent."""

    content: str = ""
    role: str = "user"  # "user" or "assistant"


@dataclass(frozen=True)
class ActionEvent(Event):
    """An agent's request to invoke a tool."""

    source: EventSource = field(default=EventSource.AGENT, init=False)
    tool_name: str = ""
    tool_call_id: str = field(default_factory=_make_id)
    arguments: dict[str, Any] = field(default_factory=dict)
    thought: str | None = None
    llm_response_id: str | None = None  # groups parallel tool calls from a single LLM response


@dataclass(frozen=True)
class ObservationEvent(Event):
    """The result of executing a tool call."""

    source: EventSource = field(default=EventSource.SYSTEM, init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    output: Any = None
    error: str | None = None
    is_error: bool = False


@dataclass(frozen=True)
class ErrorEvent(Event):
    """An error that occurred during conversation execution."""

    source: EventSource = field(default=EventSource.SYSTEM, init=False)
    error: str = ""
    recoverable: bool = True
