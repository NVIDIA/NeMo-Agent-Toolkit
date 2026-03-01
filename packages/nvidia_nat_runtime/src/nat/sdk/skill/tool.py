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
"""Skill activation tool for the NAT SDK conversation runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from nat.sdk.event.event import SystemPromptEvent
from nat.sdk.tool.tool import Tool

if TYPE_CHECKING:
    from nat.data_models.skill import Skill
    from nat.sdk.conversation.state import ConversationState
    from nat.sdk.event.event import Event


def build_read_skill_tool(
    skills: dict[str, Skill],
    state: ConversationState,
    emit_fn: Callable[[Event], None],
    build_skills_context: Callable[[], str],
) -> Tool:
    """Build a tool that activates a named skill for the current conversation.

    When invoked by the LLM, this tool:

    1. Looks up the skill by name.
    2. Marks it as activated in ``state.activated_skills`` (deduplicating).
    3. Calls ``build_skills_context()`` to get the current ``<available_skills>``
       section — which now includes the activated skill's full body — and emits
       it as a new :class:`~nat.sdk.event.event.SystemPromptEvent` so the LLM
       sees the updated context in its message history.
    4. Returns a short confirmation string.

    Using ``build_skills_context`` rather than ``skill.to_full_prompt()`` ensures
    that the emitted event always reflects the *complete, current* skills state
    (all skills, with content for every activated one) rather than a bare body
    snippet.

    Args:
        skills: Mapping of skill name → Skill, populated during conversation
            initialisation (filesystem + workspace discovery).  Mutations to
            this dict (add/remove/update) are visible to the tool immediately.
        state: Mutable conversation state.  ``activated_skills`` is updated
            in-place.
        emit_fn: Callable that appends an event to the state log and fires the
            on_event callback (mirrors ``Conversation._emit_event``).
        build_skills_context: Zero-argument callable that returns the current
            ``<available_skills>`` XML section as a string, incorporating all
            currently activated skills.  Supplied by
            ``Conversation._build_skills_section``.

    Returns:
        A :class:`~nat.sdk.tool.tool.Tool` named ``"read_skill"``.
    """

    async def _execute(skill_name: str) -> str:
        skill = skills.get(skill_name)
        if skill is None:
            available = ", ".join(sorted(skills.keys()))
            return (f"Error: skill {skill_name!r} not found. "
                    f"Available skills: {available or 'none'}.")

        if skill.name not in state.activated_skills:
            state.activated_skills.append(skill.name)

        context = build_skills_context()
        if context:
            emit_fn(SystemPromptEvent(content=context))

        return f"Skill {skill.name!r} activated."

    return Tool(
        name="read_skill",
        description=("Activate a skill by name to load its full instructions into the "
                     "conversation. Use this when you need detailed guidance from a skill."),
        parameters={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to activate.",
                },
            },
            "required": ["skill_name"],
        },
        execute=_execute,
    )
