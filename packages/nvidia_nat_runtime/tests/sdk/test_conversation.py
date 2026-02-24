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
"""Tests for nat.sdk.conversation — Conversation, ConversationState, ConversationRunner."""

from pathlib import Path
from typing import Any

import pytest

from nat.data_models.skill import Skill
from nat.data_models.workspace import ActionResult
from nat.data_models.workspace import ActionStatus
from nat.sdk.agent.agent import Agent
from nat.sdk.agent.state import AgentStatus
from nat.sdk.conversation.conversation import Conversation
from nat.sdk.conversation.state import ConversationState
from nat.sdk.conversation.state import ConversationStatus
from nat.sdk.conversation.state import UsageStats
from nat.sdk.event.event import ActionEvent
from nat.sdk.event.event import Event
from nat.sdk.event.event import MessageEvent
from nat.sdk.event.event import ObservationEvent
from nat.sdk.event.event import SystemPromptEvent
from nat.sdk.llm.client import LLMClient
from nat.sdk.llm.message import LLMResponse
from nat.sdk.llm.message import Message
from nat.sdk.llm.message import TokenUsage
from nat.sdk.llm.message import ToolCall
from nat.sdk.tool.result import ToolResult
from nat.sdk.tool.tool import Tool
from nat.workspace.types import TypeSchema
from nat.workspace.types import WorkspaceActionSchema
from nat.workspace.types import WorkspaceBase
from nat.workspace.types import WorkspaceSkillSchema

# ---------------------------------------------------------------------------
# UsageStats
# ---------------------------------------------------------------------------


class TestUsageStats:

    def test_defaults(self) -> None:
        s = UsageStats()
        assert s.total_tokens == 0
        assert s.prompt_tokens == 0
        assert s.completion_tokens == 0
        assert s.llm_calls == 0

    def test_record_with_usage(self) -> None:
        s = UsageStats()
        s.record(TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
        assert s.llm_calls == 1
        assert s.total_tokens == 15
        assert s.prompt_tokens == 10

    def test_record_none(self) -> None:
        s = UsageStats()
        s.record(None)
        assert s.llm_calls == 1
        assert s.total_tokens == 0

    def test_accumulate(self) -> None:
        s = UsageStats()
        s.record(TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
        s.record(TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30))
        assert s.llm_calls == 2
        assert s.total_tokens == 45
        assert s.prompt_tokens == 30


# ---------------------------------------------------------------------------
# ConversationState
# ---------------------------------------------------------------------------


class TestConversationState:

    def test_defaults(self) -> None:
        state = ConversationState()
        assert len(state.id) == 36  # UUID
        assert len(state.events) == 0
        assert state.agent_state.status == AgentStatus.IDLE
        assert state.status == ConversationStatus.ACTIVE
        assert state.stats.llm_calls == 0
        assert state.activated_skills == []

    def test_unique_ids(self) -> None:
        s1 = ConversationState()
        s2 = ConversationState()
        assert s1.id != s2.id


# ---------------------------------------------------------------------------
# Fake LLM client for conversation tests
# ---------------------------------------------------------------------------


class FakeLLMClient(LLMClient):
    """LLM client that returns preconfigured responses in order."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.call_log: list[dict[str, Any]] = []

    async def complete(
            self,
            messages: list[Message],
            tools: list[Tool] = (),
            **kwargs: Any,
    ) -> LLMResponse:
        self.call_log.append({
            "messages": list(messages),
            "tools": list(tools),
        })
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


def _text_response(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _tool_call_response(
    tool_name: str,
    arguments: dict[str, Any],
    tool_call_id: str = "tc-1",
) -> LLMResponse:
    return LLMResponse(
        message=Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=tool_call_id, name=tool_name, arguments=arguments)],
        ),
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


# ---------------------------------------------------------------------------
# Mock Workspace implementing WorkspaceBase
# ---------------------------------------------------------------------------


class MockWorkspace(WorkspaceBase):
    """A mock workspace for testing that exposes read_file and write_file actions."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self._files = dict(files or {})

    async def get_actions(self) -> list[WorkspaceActionSchema]:
        return [
            WorkspaceActionSchema(
                name="read_file",
                description="Read a file",
                parameters=[TypeSchema(type="path", description="File path")],
                result=TypeSchema(type="string", description="File content"),
            ),
            WorkspaceActionSchema(
                name="write_file",
                description="Write a file",
                parameters=[
                    TypeSchema(type="path", description="File path"),
                    TypeSchema(type="content", description="Content to write"),
                ],
                result=TypeSchema(type="string", description="Confirmation"),
            ),
        ]

    async def get_skills(self) -> list[WorkspaceSkillSchema]:
        return []

    async def create_skill(self, skill_name: str, skill_description: str) -> ActionResult:
        return ActionResult(status=ActionStatus.SUCCESS, output="created")

    async def execute_action(
        self,
        action_name: str,
        args: dict[str, Any],
        model_metadata: dict[str, Any] | None = None,
    ) -> ActionResult:
        if action_name == "read_file":
            path = args.get("path", "")
            content = self._files.get(path)
            if content is not None:
                return ActionResult(status=ActionStatus.SUCCESS, output=content)
            return ActionResult(status=ActionStatus.FAILURE, error_message=f"File not found: {path}")
        if action_name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            self._files[path] = content
            return ActionResult(status=ActionStatus.SUCCESS, output=f"Wrote {len(content)} bytes")
        return ActionResult(status=ActionStatus.FAILURE, error_message=f"Unknown action: {action_name}")

    async def upload_file(self, file_path: Path, destination_path: Path) -> ActionResult:
        return ActionResult(status=ActionStatus.SUCCESS)

    async def delete_file(self, file_path: Path) -> ActionResult:
        return ActionResult(status=ActionStatus.SUCCESS)

    async def download_file(self, file_path: Path) -> ActionResult:
        return ActionResult(status=ActionStatus.SUCCESS)

    async def upload_directory(self, directory_path: Path, destination_path: Path) -> ActionResult:
        return ActionResult(status=ActionStatus.SUCCESS)

    async def download_directory(self, directory_path: Path, destination_path: Path) -> ActionResult:
        return ActionResult(status=ActionStatus.SUCCESS)

    async def delete_directory(self, directory_path: Path) -> ActionResult:
        return ActionResult(status=ActionStatus.SUCCESS)


# ---------------------------------------------------------------------------
# Conversation — basic send_message
# ---------------------------------------------------------------------------


class TestConversationSendMessage:

    async def test_simple_text_response(self) -> None:
        client = FakeLLMClient([_text_response("Hello!")])
        agent = Agent(system_prompt="You are helpful.", )
        conv = Conversation(agent=agent, client=client)
        response = await conv.send_message("Hi")

        assert response.role == "assistant"
        assert response.content == "Hello!"
        assert conv.state.stats.llm_calls == 1

    async def test_events_recorded(self) -> None:
        client = FakeLLMClient([_text_response("Reply")])
        agent = Agent(system_prompt="Sys")
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("User input")

        events = conv.state.events.events
        types = [type(e).__name__ for e in events]
        assert "SystemPromptEvent" in types
        assert "MessageEvent" in types  # user + agent messages

    async def test_system_prompt_in_events(self) -> None:
        client = FakeLLMClient([_text_response("ok")])
        agent = Agent(system_prompt="Be nice", )
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("test")

        sys_events = [e for e in conv.state.events if isinstance(e, SystemPromptEvent)]
        assert len(sys_events) == 1
        assert "Be nice" in sys_events[0].content

    async def test_event_callback(self) -> None:
        collected: list[Event] = []
        client = FakeLLMClient([_text_response("Hi")])
        agent = Agent()
        conv = Conversation(agent=agent, client=client, on_event=collected.append)
        await conv.send_message("test")

        assert len(collected) > 0
        # Should include at least user message and agent message
        msg_events = [e for e in collected if isinstance(e, MessageEvent)]
        assert len(msg_events) >= 2

    async def test_no_system_prompt_when_empty(self) -> None:
        client = FakeLLMClient([_text_response("ok")])
        agent = Agent()  # no system_prompt
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("test")

        sys_events = [e for e in conv.state.events if isinstance(e, SystemPromptEvent)]
        assert len(sys_events) == 0


# ---------------------------------------------------------------------------
# Conversation — tool calling
# ---------------------------------------------------------------------------


class TestConversationToolCalling:

    async def test_single_tool_call(self) -> None:

        async def _add(a: int, b: int) -> ToolResult:
            return ToolResult(output=a + b)

        add_tool = Tool(
            name="add",
            description="Add numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer"
                    },
                    "b": {
                        "type": "integer"
                    },
                },
            },
            execute=_add,
        )

        client = FakeLLMClient([
            _tool_call_response("add", {
                "a": 2, "b": 3
            }),
            _text_response("The answer is 5"),
        ])
        agent = Agent(tools=[add_tool])
        conv = Conversation(agent=agent, client=client)
        response = await conv.send_message("What is 2+3?")

        assert response.content == "The answer is 5"
        assert conv.state.stats.llm_calls == 2

        # Verify events include action and observation
        action_events = [e for e in conv.state.events if isinstance(e, ActionEvent)]
        obs_events = [e for e in conv.state.events if isinstance(e, ObservationEvent)]
        assert len(action_events) == 1
        assert action_events[0].tool_name == "add"
        assert len(obs_events) == 1
        assert obs_events[0].output == 5

    async def test_tool_error(self) -> None:

        async def _fail() -> ToolResult:
            return ToolResult(error="tool broke")

        fail_tool = Tool(name="fail", description="Fails", execute=_fail)

        client = FakeLLMClient([
            _tool_call_response("fail", {}),
            _text_response("Sorry, the tool failed."),
        ])
        agent = Agent(tools=[fail_tool])
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("Do it")

        obs_events = [e for e in conv.state.events if isinstance(e, ObservationEvent)]
        assert len(obs_events) == 1
        assert obs_events[0].is_error
        assert "tool broke" in obs_events[0].error

    async def test_unknown_tool(self) -> None:
        client = FakeLLMClient([
            _tool_call_response("nonexistent", {}),
            _text_response("I don't have that tool."),
        ])
        agent = Agent()
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("test")

        obs_events = [e for e in conv.state.events if isinstance(e, ObservationEvent)]
        assert len(obs_events) == 1
        assert obs_events[0].is_error
        assert "Unknown tool" in obs_events[0].error

    async def test_tool_exception(self) -> None:

        async def _explode() -> ToolResult:
            raise RuntimeError("kaboom")

        explode_tool = Tool(name="explode", description="Explodes", execute=_explode)

        client = FakeLLMClient([
            _tool_call_response("explode", {}),
            _text_response("Something went wrong."),
        ])
        agent = Agent(tools=[explode_tool])
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("boom")

        obs_events = [e for e in conv.state.events if isinstance(e, ObservationEvent)]
        assert len(obs_events) == 1
        assert obs_events[0].is_error
        assert "kaboom" in obs_events[0].error

    async def test_multiple_tool_calls_in_sequence(self) -> None:

        async def _mul(x: int, y: int) -> ToolResult:
            return ToolResult(output=x * y)

        mul_tool = Tool(
            name="multiply",
            description="Multiply",
            parameters={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer"
                    },
                    "y": {
                        "type": "integer"
                    },
                },
            },
            execute=_mul,
        )

        client = FakeLLMClient([
            _tool_call_response("multiply", {
                "x": 2, "y": 3
            }, "tc-1"),
            _tool_call_response("multiply", {
                "x": 6, "y": 7
            }, "tc-2"),
            _text_response("2*3=6, then 6*7=42"),
        ])
        agent = Agent(tools=[mul_tool])
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("Chain multiply")

        assert conv.state.stats.llm_calls == 3
        action_events = [e for e in conv.state.events if isinstance(e, ActionEvent)]
        assert len(action_events) == 2


# ---------------------------------------------------------------------------
# Conversation — max iterations
# ---------------------------------------------------------------------------


class TestConversationMaxIterations:

    async def test_max_iterations_stop(self) -> None:
        """Agent should stop after max_iterations even if still calling tools."""

        async def _noop() -> ToolResult:
            return ToolResult(output="ok")

        noop = Tool(name="noop", description="Noop", execute=_noop)

        # Return infinite tool calls
        client = FakeLLMClient([_tool_call_response("noop", {})] * 10 + [_text_response("done")])
        agent = Agent(tools=[noop], max_iterations=3)
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("loop forever")

        assert conv.state.agent_state.status == AgentStatus.MAX_ITERATIONS
        assert conv.state.stats.llm_calls <= 3


# ---------------------------------------------------------------------------
# Conversation — workspace integration
# ---------------------------------------------------------------------------


class TestConversationWithWorkspace:
    """Workspace tools are now pre-built and passed via agent.tools.

    The Conversation no longer auto-discovers workspace actions; instead,
    users should use ``Tool.from_function_group_config(WorkspaceActionsConfig())``
    or build tools manually and pass them as ``agent.tools``.
    """

    @staticmethod
    def _make_workspace_tools(ws: MockWorkspace) -> list[Tool]:
        """Build SDK Tools wrapping a MockWorkspace's actions."""

        async def _read_file(path: str = "") -> ToolResult:
            result = await ws.execute_action("read_file", {"path": path})
            return ToolResult.from_action_result(result)

        async def _write_file(path: str = "", content: str = "") -> ToolResult:
            result = await ws.execute_action("write_file", {"path": path, "content": content})
            return ToolResult.from_action_result(result)

        return [
            Tool(name="read_file", description="Read a file", execute=_read_file),
            Tool(name="write_file", description="Write a file", execute=_write_file),
        ]

    async def test_workspace_tools_available(self) -> None:
        client = FakeLLMClient([_text_response("ok")])
        ws = MockWorkspace()
        tools = self._make_workspace_tools(ws)
        agent = Agent(tools=tools)
        conv = Conversation(agent=agent, client=client, workspace=ws)
        await conv.initialize()

        # Workspace tools should be in the registry
        assert "read_file" in conv.tools
        assert "write_file" in conv.tools

    async def test_workspace_tool_execution(self) -> None:
        ws = MockWorkspace(files={"test.txt": "file content"})
        tools = self._make_workspace_tools(ws)

        client = FakeLLMClient([
            _tool_call_response("read_file", {"path": "test.txt"}),
            _text_response("File contains: file content"),
        ])
        agent = Agent(tools=tools)
        conv = Conversation(agent=agent, client=client, workspace=ws)
        await conv.send_message("Read test.txt")

        obs_events = [e for e in conv.state.events if isinstance(e, ObservationEvent)]
        assert len(obs_events) == 1
        assert obs_events[0].output == "file content"


# ---------------------------------------------------------------------------
# Conversation — skills
# ---------------------------------------------------------------------------


class TestConversationWithSkills:

    async def test_skills_in_system_prompt(self) -> None:
        client = FakeLLMClient([_text_response("ok")])
        skill = Skill(name="code-review", description="Review code changes")
        agent = Agent(
            system_prompt="You are helpful.",
            skills=[skill],
        )
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("test")

        sys_events = [e for e in conv.state.events if isinstance(e, SystemPromptEvent)]
        assert len(sys_events) == 1
        assert "<available_skills>" in sys_events[0].content
        assert "code-review" in sys_events[0].content


# ---------------------------------------------------------------------------
# Conversation — context manager
# ---------------------------------------------------------------------------


class TestConversationContextManager:

    async def test_async_context_manager(self) -> None:
        client = FakeLLMClient([_text_response("hi")])
        agent = Agent()
        async with Conversation(agent=agent, client=client) as conv:
            response = await conv.send_message("test")
            assert response.content == "hi"

    async def test_initialize_idempotent(self) -> None:
        client = FakeLLMClient([_text_response("ok")])
        agent = Agent(system_prompt="Sys")
        conv = Conversation(agent=agent, client=client)
        await conv.initialize()
        await conv.initialize()  # second call should be no-op

        sys_events = [e for e in conv.state.events if isinstance(e, SystemPromptEvent)]
        assert len(sys_events) == 1  # only one system prompt


# ---------------------------------------------------------------------------
# Conversation — LLM client receives correct messages
# ---------------------------------------------------------------------------


class TestConversationMessagePassing:

    async def test_messages_passed_to_llm(self) -> None:
        client = FakeLLMClient([_text_response("reply")])
        agent = Agent(system_prompt="System prompt here")
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("User says hello")

        assert len(client.call_log) == 1
        messages = client.call_log[0]["messages"]
        roles = [m.role for m in messages]
        assert "system" in roles
        assert "user" in roles

    async def test_tools_passed_to_llm(self) -> None:
        t = Tool(name="search", description="Search")
        client = FakeLLMClient([_text_response("ok")])
        agent = Agent(tools=[t])
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("test")

        tools = client.call_log[0]["tools"]
        assert len(tools) == 1
        assert tools[0].name == "search"

    async def test_usage_stats_accumulated(self) -> None:
        client = FakeLLMClient([
            _tool_call_response("noop", {}),
            _text_response("done"),
        ])

        async def _noop() -> ToolResult:
            return ToolResult(output="ok")

        noop = Tool(name="noop", description="Noop", execute=_noop)
        agent = Agent(tools=[noop])
        conv = Conversation(agent=agent, client=client)
        await conv.send_message("test")

        assert conv.state.stats.llm_calls == 2
        assert conv.state.stats.total_tokens == 30  # 15 * 2


# ---------------------------------------------------------------------------
# ConversationRunner — new parameters (return_direct, handle_tool_errors,
#                                       max_history, initial_messages)
# ---------------------------------------------------------------------------


class TestConversationRunnerExtensions:
    """Tests for the extended ConversationRunner parameters."""

    async def test_return_direct_stops_on_matching_tool(self) -> None:
        """Runner stops and returns ObservationEvent when a return_direct tool completes."""
        from nat.sdk.conversation.runner import ConversationRunner

        async def _greet(name: str = "world") -> ToolResult:
            return ToolResult(output=f"Hello, {name}!")

        greet = Tool(
            name="greet",
            description="Greet someone",
            parameters={
                "type": "object", "properties": {
                    "name": {
                        "type": "string"
                    }
                }
            },
            execute=_greet,
        )

        client = FakeLLMClient([
            _tool_call_response("greet", {"name": "Alice"}),
        ])
        agent = Agent(tools=[greet], max_iterations=5)
        state = ConversationState()
        state.agent_state.status = AgentStatus.RUNNING

        runner = ConversationRunner(
            agent=agent,
            client=client,
            state=state,
            tools={"greet": greet},
            return_direct_tools={"greet"},
        )
        event = await runner.run_until_done()

        assert isinstance(event, ObservationEvent)
        assert event.output == "Hello, Alice!"
        assert state.agent_state.status == AgentStatus.FINISHED

    async def test_return_direct_does_not_trigger_for_other_tools(self) -> None:
        """Non-return_direct tools proceed normally through the loop."""
        from nat.sdk.conversation.runner import ConversationRunner

        async def _noop() -> ToolResult:
            return ToolResult(output="ok")

        noop = Tool(name="noop", description="Noop", execute=_noop)

        client = FakeLLMClient([
            _tool_call_response("noop", {}),
            _text_response("All done"),
        ])
        agent = Agent(tools=[noop], max_iterations=5)
        state = ConversationState()
        state.agent_state.status = AgentStatus.RUNNING

        runner = ConversationRunner(
            agent=agent,
            client=client,
            state=state,
            tools={"noop": noop},
            return_direct_tools={"some_other_tool"},
        )
        event = await runner.run_until_done()

        assert isinstance(event, MessageEvent)
        assert event.content == "All done"

    async def test_handle_tool_errors_false_raises(self) -> None:
        """When handle_tool_errors=False, tool exceptions propagate."""
        from nat.sdk.conversation.runner import ConversationRunner

        async def _explode() -> ToolResult:
            raise ValueError("kaboom")

        explode = Tool(name="explode", description="Explodes", execute=_explode)

        client = FakeLLMClient([_tool_call_response("explode", {})])
        agent = Agent(tools=[explode], max_iterations=5)
        state = ConversationState()
        state.agent_state.status = AgentStatus.RUNNING

        runner = ConversationRunner(
            agent=agent,
            client=client,
            state=state,
            tools={"explode": explode},
            handle_tool_errors=False,
        )
        with pytest.raises(ValueError, match="kaboom"):
            await runner.run_until_done()

    async def test_handle_tool_errors_true_continues(self) -> None:
        """When handle_tool_errors=True, tool exceptions become error observations."""
        from nat.sdk.conversation.runner import ConversationRunner

        async def _explode() -> ToolResult:
            raise ValueError("kaboom")

        explode = Tool(name="explode", description="Explodes", execute=_explode)

        client = FakeLLMClient([
            _tool_call_response("explode", {}),
            _text_response("I see the error"),
        ])
        agent = Agent(tools=[explode], max_iterations=5)
        state = ConversationState()
        state.agent_state.status = AgentStatus.RUNNING

        runner = ConversationRunner(
            agent=agent,
            client=client,
            state=state,
            tools={"explode": explode},
            handle_tool_errors=True,
        )
        event = await runner.run_until_done()

        assert isinstance(event, MessageEvent)
        assert event.content == "I see the error"

        obs_events = [e for e in state.events if isinstance(e, ObservationEvent)]
        assert len(obs_events) == 1
        assert obs_events[0].is_error
        assert "kaboom" in obs_events[0].error

    async def test_max_history_trims_messages(self) -> None:
        """max_history limits the number of messages sent to the LLM."""
        from nat.sdk.conversation.runner import ConversationRunner

        client = FakeLLMClient([_text_response("ok")])
        agent = Agent(max_iterations=5)
        state = ConversationState()
        state.agent_state.status = AgentStatus.RUNNING

        # Pre-populate with initial messages to exceed the limit
        initial_messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="msg1"),
            Message(role="assistant", content="resp1"),
            Message(role="user", content="msg2"),
            Message(role="assistant", content="resp2"),
            Message(role="user", content="msg3"),
        ]

        runner = ConversationRunner(
            agent=agent,
            client=client,
            state=state,
            tools={},
            max_history=3,
            initial_messages=initial_messages,
        )
        await runner.step()

        # LLM should receive only the last 3 messages
        assert len(client.call_log) == 1
        messages = client.call_log[0]["messages"]
        assert len(messages) == 3
        # last 3 of [sys, msg1, resp1, msg2, resp2, msg3] = [msg2, resp2, msg3]
        assert messages[0].content == "msg2"
        assert messages[1].content == "resp2"
        assert messages[2].content == "msg3"

    async def test_initial_messages_prepended(self) -> None:
        """initial_messages are prepended before event-derived messages."""
        from nat.sdk.conversation.runner import ConversationRunner

        client = FakeLLMClient([_text_response("reply")])
        agent = Agent(max_iterations=5)
        state = ConversationState()
        state.agent_state.status = AgentStatus.RUNNING

        initial = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]

        runner = ConversationRunner(
            agent=agent,
            client=client,
            state=state,
            tools={},
            initial_messages=initial,
        )
        await runner.step()

        messages = client.call_log[0]["messages"]
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "Be helpful"
        assert messages[1].role == "user"
        assert messages[1].content == "Hello"
