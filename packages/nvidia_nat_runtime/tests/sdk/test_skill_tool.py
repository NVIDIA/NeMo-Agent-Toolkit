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
"""Unit tests for nat.sdk.skill.tool.build_read_skill_tool."""

from nat.data_models.skill import Skill
from nat.sdk.conversation.state import ConversationState
from nat.sdk.event.event import Event
from nat.sdk.event.event import SystemPromptEvent
from nat.sdk.skill.tool import build_read_skill_tool
from nat.sdk.tool.tool import Tool

# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _build_skills_section(skills: dict[str, Skill], state: ConversationState) -> str:
    """Inline replica of Conversation._build_skills_section for isolated tests."""
    if not skills:
        return ""
    activated = set(state.activated_skills)
    parts = ["<available_skills>"]
    for skill in skills.values():
        parts.append(skill.to_prompt_xml(include_content=skill.name in activated))
    parts.append("</available_skills>")
    return "\n".join(parts)


def _make_tool(
    skills: dict[str, Skill],
    state: ConversationState | None = None,
) -> tuple[Tool, ConversationState, list[Event]]:
    """Wire up a read_skill tool with event capture and an inline skills-context builder."""
    state = state or ConversationState()
    emitted: list[Event] = []

    tool = build_read_skill_tool(
        skills=skills,
        state=state,
        emit_fn=emitted.append,
        build_skills_context=lambda: _build_skills_section(skills, state),
    )
    return tool, state, emitted


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildReadSkillTool:

    def test_tool_name(self) -> None:
        tool, _, _ = _make_tool({})
        assert tool.name == "read_skill"

    def test_tool_has_description(self) -> None:
        tool, _, _ = _make_tool({})
        assert tool.description

    def test_tool_has_skill_name_parameter(self) -> None:
        tool, _, _ = _make_tool({})
        props = tool.parameters.get("properties", {})
        assert "skill_name" in props
        assert tool.parameters.get("required") == ["skill_name"]

    async def test_unknown_skill_returns_error_string(self) -> None:
        tool, state, emitted = _make_tool({})
        result = await tool(skill_name="no-such-skill")

        assert result.error is None  # tool itself didn't raise
        assert "not found" in result.output
        assert state.activated_skills == []
        assert emitted == []

    async def test_known_skill_appends_to_activated_skills(self) -> None:
        skill = Skill(name="my-skill", description="Does stuff")
        tool, state, _ = _make_tool({"my-skill": skill})

        await tool(skill_name="my-skill")

        assert "my-skill" in state.activated_skills

    async def test_known_skill_emits_system_prompt_event(self) -> None:
        skill = Skill(name="my-skill", description="Does stuff", content="# My Skill\nDo X.")
        tool, _, emitted = _make_tool({"my-skill": skill})

        await tool(skill_name="my-skill")

        assert len(emitted) == 1
        assert isinstance(emitted[0], SystemPromptEvent)

    async def test_emitted_content_is_available_skills_section(self) -> None:
        """The emitted SystemPromptEvent contains the full <available_skills> section."""
        skill = Skill(name="my-skill", description="Does stuff", content="Full body here.")
        tool, _, emitted = _make_tool({"my-skill": skill})

        await tool(skill_name="my-skill")

        content = emitted[0].content
        assert "<available_skills>" in content
        assert "my-skill" in content
        # Body appears inside <content> tag because the skill is now activated
        assert "Full body here." in content

    async def test_emitted_content_includes_all_skills(self) -> None:
        """The updated section lists every skill, not just the activated one."""
        skill_a = Skill(name="alpha", description="Alpha skill")
        skill_b = Skill(name="beta", description="Beta skill", content="Beta body.")
        skills = {"alpha": skill_a, "beta": skill_b}
        tool, _, emitted = _make_tool(skills)

        await tool(skill_name="beta")

        content = emitted[0].content
        assert "alpha" in content
        assert "beta" in content
        # Only the activated skill has body content
        assert "Beta body." in content

    async def test_activation_is_idempotent(self) -> None:
        """Calling the tool twice for the same skill does not duplicate activated_skills."""
        skill = Skill(name="my-skill", description="Does stuff")
        tool, state, emitted = _make_tool({"my-skill": skill})

        await tool(skill_name="my-skill")
        await tool(skill_name="my-skill")

        assert state.activated_skills.count("my-skill") == 1
        # A fresh SystemPromptEvent is still emitted on each call
        assert len(emitted) == 2

    async def test_returns_confirmation_string(self) -> None:
        skill = Skill(name="my-skill", description="Does stuff")
        tool, _, _ = _make_tool({"my-skill": skill})

        result = await tool(skill_name="my-skill")

        assert result.error is None
        assert "my-skill" in result.output
        assert "activated" in result.output

    async def test_error_string_lists_available_skills(self) -> None:
        skill_a = Skill(name="skill-a", description="Skill A")
        skill_b = Skill(name="skill-b", description="Skill B")
        tool, _, _ = _make_tool({"skill-a": skill_a, "skill-b": skill_b})

        result = await tool(skill_name="missing")

        assert "skill-a" in result.output
        assert "skill-b" in result.output

    async def test_multiple_skills_each_activatable(self) -> None:
        skills = {
            "alpha": Skill(name="alpha", description="Alpha skill", content="Alpha body."),
            "beta": Skill(name="beta", description="Beta skill", content="Beta body."),
        }
        tool, state, emitted = _make_tool(skills)

        await tool(skill_name="alpha")
        await tool(skill_name="beta")

        assert "alpha" in state.activated_skills
        assert "beta" in state.activated_skills
        assert len(emitted) == 2
        # Second emission should have both skills' bodies
        second_content = emitted[1].content
        assert "Alpha body." in second_content
        assert "Beta body." in second_content

    async def test_no_event_emitted_when_skills_section_is_empty(self) -> None:
        """When build_skills_context returns '', no event is emitted."""
        skill = Skill(name="my-skill", description="Does stuff")
        state = ConversationState()
        emitted: list[Event] = []

        # Provide a context builder that always returns empty
        tool = build_read_skill_tool(
            skills={"my-skill": skill},
            state=state,
            emit_fn=emitted.append,
            build_skills_context=lambda: "",
        )

        await tool(skill_name="my-skill")

        assert emitted == []
        assert "my-skill" in state.activated_skills
