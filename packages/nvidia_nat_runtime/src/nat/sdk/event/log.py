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

from collections.abc import Iterator

from nat.sdk.event.event import ActionEvent
from nat.sdk.event.event import Event
from nat.sdk.event.event import MessageEvent
from nat.sdk.event.event import ObservationEvent
from nat.sdk.event.event import SystemPromptEvent
from nat.sdk.llm.message import Message
from nat.sdk.llm.message import ToolCall


class EventLog:
    """Ordered, append-only event store for a conversation.

    Provides conversion of the event timeline into LLM message format
    and detection of pending (unmatched) tool calls.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []

    # -- Mutation --

    def append(self, event: Event) -> None:
        """Append an event to the log."""
        self._events.append(event)

    def extend(self, events: list[Event]) -> None:
        """Append multiple events to the log."""
        self._events.extend(events)

    # -- Query --

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __getitem__(self, index: int) -> Event:
        return self._events[index]

    @property
    def events(self) -> list[Event]:
        """Return a shallow copy of all events."""
        return list(self._events)

    def get_unmatched_actions(self) -> list[ActionEvent]:
        """Return ActionEvents that have no corresponding ObservationEvent."""
        matched_ids: set[str] = set()
        for event in self._events:
            if isinstance(event, ObservationEvent):
                matched_ids.add(event.tool_call_id)

        return [
            event for event in self._events if isinstance(event, ActionEvent) and event.tool_call_id not in matched_ids
        ]

    def to_messages(self) -> list[Message]:
        """Convert the event log into a list of LLM-compatible messages.

        Groups parallel tool calls (same llm_response_id) into a single
        assistant message with multiple tool_calls.
        """

        messages: list[Message] = []

        # Buffer for grouping parallel ActionEvents by llm_response_id
        pending_actions: list[ActionEvent] = []

        def _flush_actions() -> None:
            """Convert buffered ActionEvents into an assistant message + tool results."""
            if not pending_actions:
                return

            tool_calls = [
                ToolCall(
                    id=action.tool_call_id,
                    name=action.tool_name,
                    arguments=action.arguments,
                ) for action in pending_actions
            ]

            # The thought from the first action serves as the assistant text
            thought = pending_actions[0].thought or ""
            messages.append(Message(role="assistant", content=thought, tool_calls=tool_calls))
            pending_actions.clear()

        for event in self._events:
            if isinstance(event, SystemPromptEvent):
                messages.append(Message(role="system", content=event.content))

            elif isinstance(event, MessageEvent):
                _flush_actions()
                messages.append(Message(role=event.role, content=event.content))

            elif isinstance(event, ActionEvent):
                # Buffer actions; flush when the group changes
                if pending_actions and (event.llm_response_id is None
                                        or event.llm_response_id != pending_actions[0].llm_response_id):
                    _flush_actions()
                pending_actions.append(event)

            elif isinstance(event, ObservationEvent):
                _flush_actions()
                content = event.error if event.is_error else str(event.output)
                messages.append(Message(
                    role="tool",
                    content=content or "",
                    tool_call_id=event.tool_call_id,
                ))

        _flush_actions()
        return messages
