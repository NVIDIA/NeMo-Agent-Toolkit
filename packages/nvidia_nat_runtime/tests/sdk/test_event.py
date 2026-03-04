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
"""Tests for nat.sdk.event.event — Event base and concrete types."""

from datetime import UTC

from nat.sdk.event.event import ActionEvent
from nat.sdk.event.event import ErrorEvent
from nat.sdk.event.event import Event
from nat.sdk.event.event import EventSource
from nat.sdk.event.event import MessageEvent
from nat.sdk.event.event import ObservationEvent
from nat.sdk.event.event import SystemPromptEvent

# ---------------------------------------------------------------------------
# EventSource enum
# ---------------------------------------------------------------------------


class TestEventSource:

    def test_values(self) -> None:
        assert EventSource.USER == "user"
        assert EventSource.AGENT == "agent"
        assert EventSource.SYSTEM == "system"

    def test_membership(self) -> None:
        assert "user" in EventSource.__members__.values()


# ---------------------------------------------------------------------------
# Event base
# ---------------------------------------------------------------------------


class TestEvent:

    def test_default_id_is_uuid(self) -> None:
        e = Event()
        assert len(e.id) == 36  # UUID format
        assert "-" in e.id

    def test_unique_ids(self) -> None:
        e1 = Event()
        e2 = Event()
        assert e1.id != e2.id

    def test_default_timestamp_is_utc(self) -> None:
        e = Event()
        assert e.timestamp.tzinfo == UTC

    def test_default_source_is_system(self) -> None:
        e = Event()
        assert e.source == EventSource.SYSTEM

    def test_custom_source(self) -> None:
        e = Event(source=EventSource.USER)
        assert e.source == EventSource.USER

    def test_frozen(self) -> None:
        e = Event()
        try:
            e.id = "new-id"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# SystemPromptEvent
# ---------------------------------------------------------------------------


class TestSystemPromptEvent:

    def test_defaults(self) -> None:
        e = SystemPromptEvent()
        assert e.source == EventSource.SYSTEM
        assert e.content == ""

    def test_with_content(self) -> None:
        e = SystemPromptEvent(content="You are a helpful assistant.")
        assert e.content == "You are a helpful assistant."

    def test_source_always_system(self) -> None:
        e = SystemPromptEvent()
        assert e.source == EventSource.SYSTEM


# ---------------------------------------------------------------------------
# MessageEvent
# ---------------------------------------------------------------------------


class TestMessageEvent:

    def test_user_message(self) -> None:
        e = MessageEvent(source=EventSource.USER, content="Hello", role="user")
        assert e.source == EventSource.USER
        assert e.content == "Hello"
        assert e.role == "user"

    def test_assistant_message(self) -> None:
        e = MessageEvent(
            source=EventSource.AGENT,
            content="Hi there!",
            role="assistant",
        )
        assert e.source == EventSource.AGENT
        assert e.role == "assistant"

    def test_default_role(self) -> None:
        e = MessageEvent()
        assert e.role == "user"

    def test_empty_content(self) -> None:
        e = MessageEvent()
        assert e.content == ""


# ---------------------------------------------------------------------------
# ActionEvent
# ---------------------------------------------------------------------------


class TestActionEvent:

    def test_defaults(self) -> None:
        e = ActionEvent()
        assert e.source == EventSource.AGENT
        assert e.tool_name == ""
        assert e.arguments == {}
        assert e.thought is None
        assert e.llm_response_id is None

    def test_with_fields(self) -> None:
        e = ActionEvent(
            tool_name="search",
            tool_call_id="tc-1",
            arguments={"query": "test"},
            thought="I should search",
            llm_response_id="resp-1",
        )
        assert e.tool_name == "search"
        assert e.tool_call_id == "tc-1"
        assert e.arguments == {"query": "test"}
        assert e.thought == "I should search"
        assert e.llm_response_id == "resp-1"

    def test_source_always_agent(self) -> None:
        e = ActionEvent()
        assert e.source == EventSource.AGENT

    def test_tool_call_id_auto_generated(self) -> None:
        e1 = ActionEvent()
        e2 = ActionEvent()
        assert e1.tool_call_id != e2.tool_call_id


# ---------------------------------------------------------------------------
# ObservationEvent
# ---------------------------------------------------------------------------


class TestObservationEvent:

    def test_defaults(self) -> None:
        e = ObservationEvent()
        assert e.source == EventSource.SYSTEM
        assert e.tool_call_id == ""
        assert e.tool_name == ""
        assert e.output is None
        assert e.error is None
        assert e.is_error is False

    def test_success(self) -> None:
        e = ObservationEvent(
            tool_call_id="tc-1",
            tool_name="search",
            output="found it",
        )
        assert e.output == "found it"
        assert e.is_error is False

    def test_error(self) -> None:
        e = ObservationEvent(
            tool_call_id="tc-1",
            tool_name="search",
            error="not found",
            is_error=True,
        )
        assert e.error == "not found"
        assert e.is_error is True


# ---------------------------------------------------------------------------
# ErrorEvent
# ---------------------------------------------------------------------------


class TestErrorEvent:

    def test_defaults(self) -> None:
        e = ErrorEvent()
        assert e.source == EventSource.SYSTEM
        assert e.error == ""
        assert e.recoverable is True

    def test_non_recoverable(self) -> None:
        e = ErrorEvent(error="fatal", recoverable=False)
        assert e.error == "fatal"
        assert e.recoverable is False
