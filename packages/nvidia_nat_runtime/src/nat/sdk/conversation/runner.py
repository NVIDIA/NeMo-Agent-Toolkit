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
"""ConversationRunner — the agent loop that drives a conversation forward."""

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

from nat.sdk.agent.agent import Agent
from nat.sdk.agent.state import AgentStatus
from nat.sdk.conversation.state import ConversationState
from nat.sdk.event.event import ActionEvent
from nat.sdk.event.event import ErrorEvent
from nat.sdk.event.event import Event
from nat.sdk.event.event import EventSource
from nat.sdk.event.event import MessageEvent
from nat.sdk.event.event import ObservationEvent
from nat.sdk.llm.client import LLMClient
from nat.sdk.llm.message import LLMResponse
from nat.sdk.llm.message import Message
from nat.sdk.tool.result import ToolResult
from nat.sdk.tool.tool import Tool
from nat.workspace.types import WorkspaceBase

logger = logging.getLogger(__name__)

EventCallback = Callable[["Event"], None]


class ConversationRunner:
    """Drives the agent loop: LLM call -> tool execution -> repeat.

    This class encapsulates the core agentic loop and is used internally
    by :class:`~nat.sdk.conversation.conversation.Conversation`.
    """

    def __init__(
        self,
        agent: Agent,
        client: LLMClient,
        state: ConversationState,
        tools: dict[str, Tool],
        *,
        workspace: WorkspaceBase | None = None,
        on_event: EventCallback | None = None,
        return_direct_tools: set[str] | None = None,
        handle_tool_errors: bool = True,
        max_history: int = 0,
        initial_messages: list[Message] | None = None,
    ) -> None:
        self._agent = agent
        self._client = client
        self._state = state
        self._tools = tools
        self._workspace = workspace
        self._on_event = on_event
        self._return_direct_tools = return_direct_tools or set()
        self._handle_tool_errors = handle_tool_errors
        self._max_history = max_history
        self._initial_messages = initial_messages or []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def step(self) -> Event:
        """Execute a single agent step (one LLM call + tool execution).

        Returns the last event emitted during this step.
        """
        self._state.agent_state.status = AgentStatus.RUNNING

        # Build messages from initial messages + event log
        messages = self._initial_messages + self._state.events.to_messages()

        # Apply max_history trimming
        if self._max_history > 0 and len(messages) > self._max_history:
            messages = messages[-self._max_history:]

        # Call LLM
        tools_list = list(self._tools.values())
        response = await self._client.complete(messages, tools_list)
        self._state.stats.record(response.usage)

        # Process response
        if response.message.tool_calls:
            return await self._handle_tool_calls(response)
        else:
            return self._handle_message(response)

    async def run_until_done(self) -> Event:
        """Run the agent loop until it produces a final message or hits a limit.

        Returns the final event.
        """
        last_event: Event = ErrorEvent(error="No steps executed")

        while self._state.agent_state.status == AgentStatus.RUNNING:
            if self._state.agent_state.iteration >= self._agent.max_iterations:
                self._state.agent_state.status = AgentStatus.MAX_ITERATIONS
                last_event = self._emit(ErrorEvent(error=f"Max iterations ({self._agent.max_iterations}) reached"))
                break

            self._state.agent_state.iteration += 1
            last_event = await self.step()

            # return_direct: check if any recent observation is from a return_direct tool
            if self._return_direct_tools:
                for event in reversed(self._state.events.events):
                    if not isinstance(event, ObservationEvent):
                        break
                    if event.tool_name in self._return_direct_tools:
                        self._state.agent_state.status = AgentStatus.FINISHED
                        return event

            # If the last event is a message from the agent, we're done
            if isinstance(last_event, MessageEvent) and last_event.source == EventSource.AGENT:
                self._state.agent_state.status = AgentStatus.FINISHED
                break

        return last_event

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _handle_tool_calls(self, response: LLMResponse) -> Event:
        """Process tool calls from an LLM response, executing them concurrently."""
        llm_response_id = str(uuid.uuid4())

        # 1. Emit all ActionEvents sequentially (preserves event log order)
        for tc in response.message.tool_calls:
            action_event = ActionEvent(
                tool_name=tc.name,
                tool_call_id=tc.id,
                arguments=tc.arguments,
                thought=response.message.content or None,
                llm_response_id=llm_response_id,
            )
            self._emit(action_event)

        # 2. Execute all tools concurrently
        tasks = [self._execute_tool(tc.name, tc.id, tc.arguments) for tc in response.message.tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Handle any exceptions from gather (shouldn't happen since
        #    _execute_tool already catches exceptions, but be defensive)
        last_event: Event = ErrorEvent(error="No tool calls processed")
        for result in results:
            if isinstance(result, BaseException):
                if not self._handle_tool_errors:
                    raise result
                last_event = self._emit(ErrorEvent(error=str(result)))
            else:
                last_event = result

        return last_event

    async def _execute_tool(self, tool_name: str, tool_call_id: str, arguments: dict[str, Any]) -> ObservationEvent:
        """Execute a single tool and emit the observation."""
        tool = self._tools.get(tool_name)
        if tool is None:
            obs = ObservationEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                error=f"Unknown tool: {tool_name!r}",
                is_error=True,
            )
            return self._emit(obs)

        try:
            result: ToolResult = await tool.invoke(**arguments)
            obs = ObservationEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                output=result.output,
                error=result.error,
                is_error=result.is_error,
            )
        except Exception as exc:
            if not self._handle_tool_errors:
                raise
            obs = ObservationEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                error=str(exc),
                is_error=True,
            )

        return self._emit(obs)

    def _handle_message(self, response: LLMResponse) -> MessageEvent:
        """Handle a text-only LLM response."""
        event = MessageEvent(
            source=EventSource.AGENT,
            content=response.message.content,
            role="assistant",
        )
        return self._emit(event)

    def _emit(self, event: Event) -> Event:
        """Append an event to the log and notify the callback."""
        self._state.events.append(event)
        if self._on_event:
            self._on_event(event)
        return event
