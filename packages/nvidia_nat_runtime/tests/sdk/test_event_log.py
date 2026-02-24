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
"""Tests for nat.sdk.event.log — EventLog."""

from nat.sdk.event.event import ActionEvent
from nat.sdk.event.event import EventSource
from nat.sdk.event.event import MessageEvent
from nat.sdk.event.event import ObservationEvent
from nat.sdk.event.event import SystemPromptEvent
from nat.sdk.event.log import EventLog


class TestEventLogBasics:

    def test_empty(self) -> None:
        log = EventLog()
        assert len(log) == 0
        assert list(log) == []

    def test_append_and_len(self) -> None:
        log = EventLog()
        log.append(MessageEvent(content="hi"))
        assert len(log) == 1

    def test_extend(self) -> None:
        log = EventLog()
        log.extend([MessageEvent(content="a"), MessageEvent(content="b")])
        assert len(log) == 2

    def test_iteration(self) -> None:
        log = EventLog()
        e1 = MessageEvent(content="first")
        e2 = MessageEvent(content="second")
        log.append(e1)
        log.append(e2)
        assert list(log) == [e1, e2]

    def test_getitem(self) -> None:
        log = EventLog()
        e = MessageEvent(content="test")
        log.append(e)
        assert log[0] is e

    def test_events_returns_copy(self) -> None:
        log = EventLog()
        log.append(MessageEvent(content="test"))
        events = log.events
        events.clear()
        assert len(log) == 1


class TestEventLogUnmatchedActions:

    def test_no_actions(self) -> None:
        log = EventLog()
        log.append(MessageEvent(content="hi"))
        assert log.get_unmatched_actions() == []

    def test_unmatched_action(self) -> None:
        log = EventLog()
        action = ActionEvent(tool_name="search", tool_call_id="tc-1")
        log.append(action)
        unmatched = log.get_unmatched_actions()
        assert len(unmatched) == 1
        assert unmatched[0].tool_call_id == "tc-1"

    def test_matched_action(self) -> None:
        log = EventLog()
        log.append(ActionEvent(tool_name="search", tool_call_id="tc-1"))
        log.append(ObservationEvent(tool_call_id="tc-1", tool_name="search", output="ok"))
        assert log.get_unmatched_actions() == []

    def test_partial_match(self) -> None:
        log = EventLog()
        log.append(ActionEvent(tool_name="a", tool_call_id="tc-1"))
        log.append(ActionEvent(tool_name="b", tool_call_id="tc-2"))
        log.append(ObservationEvent(tool_call_id="tc-1", tool_name="a", output="ok"))
        unmatched = log.get_unmatched_actions()
        assert len(unmatched) == 1
        assert unmatched[0].tool_call_id == "tc-2"


class TestEventLogToMessages:

    def test_system_prompt(self) -> None:
        log = EventLog()
        log.append(SystemPromptEvent(content="Be helpful"))
        messages = log.to_messages()
        assert len(messages) == 1
        assert messages[0].role == "system"
        assert messages[0].content == "Be helpful"

    def test_user_message(self) -> None:
        log = EventLog()
        log.append(MessageEvent(source=EventSource.USER, content="Hi", role="user"))
        messages = log.to_messages()
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hi"

    def test_assistant_message(self) -> None:
        log = EventLog()
        log.append(MessageEvent(
            source=EventSource.AGENT,
            content="Hello!",
            role="assistant",
        ))
        messages = log.to_messages()
        assert len(messages) == 1
        assert messages[0].role == "assistant"

    def test_action_and_observation(self) -> None:
        log = EventLog()
        log.append(
            ActionEvent(
                tool_name="add",
                tool_call_id="tc-1",
                arguments={
                    "a": 1, "b": 2
                },
                thought="Let me add",
                llm_response_id="resp-1",
            ))
        log.append(ObservationEvent(tool_call_id="tc-1", tool_name="add", output="3"))
        messages = log.to_messages()

        # Should produce: assistant message with tool_calls, then tool result
        assert len(messages) == 2
        assert messages[0].role == "assistant"
        assert len(messages[0].tool_calls) == 1
        assert messages[0].tool_calls[0].name == "add"
        assert messages[0].content == "Let me add"

        assert messages[1].role == "tool"
        assert messages[1].tool_call_id == "tc-1"
        assert messages[1].content == "3"

    def test_parallel_tool_calls_grouped(self) -> None:
        """Multiple ActionEvents with the same llm_response_id become one assistant message."""
        log = EventLog()
        log.append(
            ActionEvent(
                tool_name="a",
                tool_call_id="tc-1",
                arguments={},
                thought="Doing both",
                llm_response_id="resp-1",
            ))
        log.append(ActionEvent(
            tool_name="b",
            tool_call_id="tc-2",
            arguments={},
            llm_response_id="resp-1",
        ))
        log.append(ObservationEvent(tool_call_id="tc-1", tool_name="a", output="r1"))
        log.append(ObservationEvent(tool_call_id="tc-2", tool_name="b", output="r2"))

        messages = log.to_messages()
        # 1 assistant message with 2 tool_calls + 2 tool results
        assert len(messages) == 3
        assert messages[0].role == "assistant"
        assert len(messages[0].tool_calls) == 2
        assert messages[1].role == "tool"
        assert messages[2].role == "tool"

    def test_error_observation(self) -> None:
        log = EventLog()
        log.append(ActionEvent(
            tool_name="fail",
            tool_call_id="tc-1",
            arguments={},
            llm_response_id="resp-1",
        ))
        log.append(ObservationEvent(
            tool_call_id="tc-1",
            tool_name="fail",
            error="boom",
            is_error=True,
        ))
        messages = log.to_messages()
        assert messages[1].role == "tool"
        assert messages[1].content == "boom"

    def test_full_conversation_flow(self) -> None:
        """System prompt -> user message -> tool call -> observation -> assistant response."""
        log = EventLog()
        log.append(SystemPromptEvent(content="You are helpful"))
        log.append(MessageEvent(source=EventSource.USER, content="Add 1+2", role="user"))
        log.append(
            ActionEvent(
                tool_name="add",
                tool_call_id="tc-1",
                arguments={
                    "a": 1, "b": 2
                },
                thought="",
                llm_response_id="resp-1",
            ))
        log.append(ObservationEvent(tool_call_id="tc-1", tool_name="add", output="3"))
        log.append(MessageEvent(
            source=EventSource.AGENT,
            content="The answer is 3",
            role="assistant",
        ))

        messages = log.to_messages()
        assert len(messages) == 5
        assert [m.role for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
