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
from nat.sdk.conversation.state import ConversationStatus
from nat.sdk.event.event import Event
from nat.sdk.event.event import EventSource
from nat.sdk.event.event import MessageEvent
from nat.sdk.event.event import SystemPromptEvent
from nat.sdk.llm.client import LLMClient
from nat.sdk.llm.message import Message
from nat.sdk.skill.tool import build_read_skill_tool
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

        # Discover skills from configured directories and workspace
        await self._discover_skills()

        # Wire skill-activation tool when skills are available
        if self._skills:
            self._tools["read_skill"] = build_read_skill_tool(
                skills=self._skills,
                state=self._state,
                emit_fn=self._emit_event,
                build_skills_context=self._build_skills_section,
            )

        # Emit system prompt event
        system_prompt = self._build_system_prompt()
        if system_prompt:
            self._emit_event(SystemPromptEvent(content=system_prompt))

        self._initialized = True

    async def close(self) -> None:
        """Mark conversation as completed."""
        self._state.status = ConversationStatus.COMPLETED

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

    # -- Skill mutation API --

    def add_skill(self, skill: Skill) -> None:
        """Add or replace a skill and refresh the conversation context.

        If the conversation has already been initialized, a new
        :class:`~nat.sdk.event.event.SystemPromptEvent` is emitted with the
        updated ``<available_skills>`` section so the LLM sees the new skill
        immediately.  Before initialization the skill is simply registered and
        will appear in the initial system prompt.

        Args:
            skill: The skill to add.  Replaces any existing skill with the
                same name.
        """
        self._skills[skill.name] = skill
        if not self._initialized:
            return
        # Ensure read_skill tool is present (the closure already holds a
        # reference to self._skills, so no rebuild is needed when skills are
        # merely added; but we do need to insert the tool if this is the first
        # skill being added after initialization).
        if "read_skill" not in self._tools:
            self._tools["read_skill"] = build_read_skill_tool(
                skills=self._skills,
                state=self._state,
                emit_fn=self._emit_event,
                build_skills_context=self._build_skills_section,
            )
        self._emit_skills_update()

    def remove_skill(self, skill_name: str) -> None:
        """Remove a skill and refresh the conversation context.

        If the conversation has already been initialized, a new
        :class:`~nat.sdk.event.event.SystemPromptEvent` is emitted reflecting
        the reduced skill list.  If no skills remain, the ``read_skill`` tool
        is removed from the active tool set.

        Args:
            skill_name: Name of the skill to remove.  No-op if not found.
        """
        self._skills.pop(skill_name, None)
        if not self._initialized:
            return
        self._emit_skills_update()
        if not self._skills:
            self._tools.pop("read_skill", None)

    def update_skill(self, skill: Skill) -> None:
        """Replace an existing skill's definition and refresh the conversation context.

        Functionally equivalent to :meth:`add_skill` but signals intent that
        the skill already exists.  If the skill is currently activated its new
        body will appear in the emitted ``<available_skills>`` section
        immediately.

        Args:
            skill: Updated skill.  Must have the same ``name`` as the skill
                being replaced.
        """
        self._skills[skill.name] = skill
        if not self._initialized:
            return
        self._emit_skills_update()

    # -- Internal --

    def _emit_event(self, event: Event) -> None:
        """Append an event to the state log and notify the callback."""
        self._state.events.append(event)
        if self._on_event:
            self._on_event(event)

    def _emit_skills_update(self) -> None:
        """Emit a SystemPromptEvent containing the current skills section.

        Called after any skill mutation (add/remove/update/activate) so the
        LLM's context always reflects the live state of ``self._skills``.
        Does nothing when there are no skills to describe.
        """
        section = self._build_skills_section()
        if section:
            self._emit_event(SystemPromptEvent(content=section))

    def _create_runner(self) -> ConversationRunner:
        return ConversationRunner(
            agent=self._agent,
            client=self._client,
            state=self._state,
            tools=self._effective_tools(),
            workspace=self._workspace,
            on_event=self._on_event,
        )

    def _effective_tools(self) -> dict[str, Tool]:
        """Return the tool set, filtered by active skills' allowed_tools.

        If no skills are activated, or no activated skill carries an
        ``allowed_tools`` restriction, the full tool dict is returned.
        Otherwise only tools listed by at least one active skill are kept,
        plus ``"read_skill"`` which is always available when skills are present.
        """
        if not self._skills:
            return self._tools
        activated = set(self._state.activated_skills)
        if not activated:
            return self._tools
        # Union of all active skills' allowed_tools; empty list means no restriction
        restricted: set[str] = set()
        has_restriction = False
        for name in activated:
            skill = self._skills.get(name)
            if skill and skill.allowed_tools:
                restricted.update(skill.allowed_tools)
                has_restriction = True
        if not has_restriction:
            return self._tools
        # Always keep read_skill available so the LLM can activate more skills
        restricted.add("read_skill")
        return {k: v for k, v in self._tools.items() if k in restricted}

    async def _discover_skills(self) -> None:
        """Discover skills from configured directories and the workspace.

        Filesystem discovery runs first so local skills take precedence over
        workspace-provided stubs.
        """
        for skill_dir in self._agent.skill_discovery_dirs:
            self._scan_skill_directory(skill_dir)

        if self._workspace is not None:
            try:
                summaries = await self._workspace.get_skills()
                for summary in summaries:
                    if summary.name not in self._skills:
                        self._skills[summary.name] = Skill(name=summary.name, description=summary.description)
            except Exception:
                logger.warning("Failed to retrieve skills from workspace", exc_info=True)

    def _scan_skill_directory(self, directory: str | Any) -> None:
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

    def _build_skills_section(self) -> str:
        """Build the ``<available_skills>`` XML section for the current skill state.

        Includes full body content (``<content>`` tag) for every skill that
        appears in ``state.activated_skills``.

        Returns:
            The ``<available_skills>`` XML string, or ``""`` if no skills exist.
        """
        if not self._skills:
            return ""
        activated = set(self._state.activated_skills)
        parts = ["<available_skills>"]
        for skill in self._skills.values():
            parts.append(skill.to_prompt_xml(include_content=skill.name in activated))
        parts.append("</available_skills>")
        return "\n".join(parts)

    def _build_system_prompt(self) -> str:
        """Build the full system prompt including skills metadata."""
        parts: list[str] = []

        # Agent system prompt
        rendered = self._agent.render_system_prompt()
        if rendered:
            parts.append(rendered)

        # Skills metadata
        skills_section = self._build_skills_section()
        if skills_section:
            parts.append(skills_section)

        return "\n\n".join(parts)
