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

import logging
from typing import Any

from nat.data_models.skill import Skill
from nat.sdk.agent.agent import Agent
from nat.sdk.agent.state import AgentStatus
from nat.sdk.conversation.runner import ConversationRunner
from nat.sdk.conversation.runner import EventCallback
from nat.sdk.conversation.state import ConversationState
from nat.sdk.event.event import Event
from nat.sdk.event.event import EventSource
from nat.sdk.event.event import MessageEvent
from nat.sdk.event.event import SystemPromptEvent
from nat.sdk.llm.client import LLMClient
from nat.sdk.llm.message import Message
from nat.sdk.tool.tool import Tool
from nat.workspace.types import WorkspaceBase

logger = logging.getLogger(__name__)


class Conversation:
    """Main user-facing orchestrator that ties an Agent to runtime state.

    Usage::

        agent = Agent(tools=[...], system_prompt="You are...")
        conv = Conversation(agent=agent, client=my_client)
        response = await conv.send_message("Hello!")

    Workspace actions should be added as tools via
    ``Tool.from_function_group_config(WorkspaceActionsConfig())`` rather
    than passing a workspace directly.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        client: LLMClient,
        workspace: WorkspaceBase | None = None,
        state: ConversationState | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self._agent = agent
        self._client = client
        # Explicit workspace overrides agent.workspace
        self._workspace = workspace if workspace is not None else agent.workspace
        self._state = state or ConversationState()
        self._on_event = on_event
        self._initialized = False

        # Index tools by name
        self._tools: dict[str, Tool] = {t.name: t for t in agent.tools}

        # Index skills by name
        self._skills: dict[str, Skill] = {s.name: s for s in agent.skills}

    # -- Properties --

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def tools(self) -> dict[str, Tool]:
        return self._tools

    @property
    def skills(self) -> dict[str, Skill]:
        return self._skills

    # -- Lifecycle --

    async def initialize(self) -> None:
        """Initialize the conversation: discover skills, emit system prompt."""
        if self._initialized:
            return

        # Discover skills from configured directories
        for skill_dir in self._agent.skill_discovery_dirs:
            self._discover_skills(skill_dir)

        # Emit system prompt event
        system_prompt = self._build_system_prompt()
        if system_prompt:
            event = SystemPromptEvent(content=system_prompt)
            self._state.events.append(event)
            if self._on_event:
                self._on_event(event)

        self._initialized = True

    async def close(self) -> None:
        """Clean up resources."""
        pass

    async def __aenter__(self) -> Conversation:
        await self.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # -- Core API --

    async def send_message(self, content: str) -> Message:
        """Send a user message and run the agent until it responds.

        Returns the final assistant message.
        """
        await self.initialize()

        # Append user message
        user_event = MessageEvent(source=EventSource.USER, content=content, role="user")
        self._state.events.append(user_event)
        if self._on_event:
            self._on_event(user_event)

        # Run agent loop
        self._state.agent_state.status = AgentStatus.RUNNING
        runner = self._create_runner()
        last_event = await runner.run_until_done()

        # Extract final message
        if isinstance(last_event, MessageEvent):
            return Message(role="assistant", content=last_event.content)

        return Message(role="assistant", content=str(last_event))

    async def step(self) -> Event:
        """Execute a single agent step (one LLM call + possible tool execution).

        Returns the last event emitted during this step.
        """
        await self.initialize()
        runner = self._create_runner()
        return await runner.step()

    # -- Internal --

    def _create_runner(self) -> ConversationRunner:
        return ConversationRunner(
            agent=self._agent,
            client=self._client,
            state=self._state,
            tools=self._tools,
            workspace=self._workspace,
            on_event=self._on_event,
        )

    def _discover_skills(self, directory: str | Any) -> None:
        """Scan a directory for valid skill folders and register them."""
        from pathlib import Path

        directory = Path(directory)
        if not directory.is_dir():
            return

        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                skill = Skill.load(child)
                self._skills[skill.name] = skill
            except Exception:
                logger.warning("Failed to load skill from %s", child, exc_info=True)

    def _build_system_prompt(self) -> str:
        """Build the full system prompt including skills metadata."""
        parts: list[str] = []

        # Agent system prompt
        rendered = self._agent.render_system_prompt()
        if rendered:
            parts.append(rendered)

        # Skills metadata
        if self._skills:
            activated = set(self._state.activated_skills)
            skill_parts = ["<available_skills>"]
            for skill in self._skills.values():
                include_content = skill.name in activated
                skill_parts.append(skill.to_prompt_xml(include_content=include_content))
            skill_parts.append("</available_skills>")
            parts.append("\n".join(skill_parts))

        return "\n\n".join(parts)
