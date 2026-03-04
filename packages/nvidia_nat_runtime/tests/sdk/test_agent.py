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
"""Tests for nat.sdk.agent — Agent and AgentState."""

import pytest
from pydantic import ValidationError

from nat.data_models.skill import Skill
from nat.sdk.agent.agent import Agent
from nat.sdk.agent.state import AgentState
from nat.sdk.agent.state import AgentStatus
from nat.sdk.tool.tool import Tool

# ---------------------------------------------------------------------------
# AgentStatus
# ---------------------------------------------------------------------------


class TestAgentStatus:

    def test_values(self) -> None:
        assert AgentStatus.IDLE == "idle"
        assert AgentStatus.RUNNING == "running"
        assert AgentStatus.FINISHED == "finished"
        assert AgentStatus.ERROR == "error"
        assert AgentStatus.MAX_ITERATIONS == "max_iterations"


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


class TestAgentState:

    def test_defaults(self) -> None:
        state = AgentState()
        assert state.iteration == 0
        assert state.status == AgentStatus.IDLE
        assert state.metadata == {}

    def test_mutable(self) -> None:
        state = AgentState()
        state.iteration = 5
        state.status = AgentStatus.RUNNING
        state.metadata["key"] = "value"
        assert state.iteration == 5
        assert state.status == AgentStatus.RUNNING
        assert state.metadata["key"] == "value"

    def test_reset(self) -> None:
        state = AgentState(
            iteration=10,
            status=AgentStatus.FINISHED,
            metadata={"x": 1},
        )
        state.reset()
        assert state.iteration == 0
        assert state.status == AgentStatus.IDLE
        assert state.metadata == {}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TestAgent:

    def test_minimal(self) -> None:
        agent = Agent()
        assert agent.tools == []
        assert agent.skills == []
        assert agent.system_prompt == ""
        assert agent.system_prompt_kwargs == {}
        assert agent.max_iterations == 50

    def test_with_tools(self) -> None:
        t = Tool(name="search", description="Search")
        agent = Agent(tools=[t])
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "search"

    def test_with_skills(self) -> None:
        s = Skill(name="code-review", description="Review code")
        agent = Agent(skills=[s])
        assert len(agent.skills) == 1
        assert agent.skills[0].name == "code-review"

    def test_frozen(self) -> None:
        agent = Agent()
        with pytest.raises(ValidationError):
            agent.max_iterations = 100  # type: ignore[misc]

    def test_render_system_prompt_plain(self) -> None:
        agent = Agent(system_prompt="You are a helpful assistant.", )
        assert agent.render_system_prompt() == "You are a helpful assistant."

    def test_render_system_prompt_with_kwargs(self) -> None:
        agent = Agent(
            system_prompt="Hello {name}, you are a {role}.",
            system_prompt_kwargs={
                "name": "Agent", "role": "helper"
            },
        )
        result = agent.render_system_prompt()
        assert result == "Hello Agent, you are a helper."

    def test_render_system_prompt_with_extra_kwargs(self) -> None:
        agent = Agent(
            system_prompt="Hi {name}!",
            system_prompt_kwargs={"name": "Default"},
        )
        result = agent.render_system_prompt(name="Override")
        assert result == "Hi Override!"

    def test_render_system_prompt_empty(self) -> None:
        agent = Agent()
        assert agent.render_system_prompt() == ""

    def test_render_system_prompt_missing_key_returns_raw(self) -> None:
        """If a template key is missing, return the raw template."""
        agent = Agent(system_prompt="Hello {unknown_key}!", )
        result = agent.render_system_prompt()
        assert result == "Hello {unknown_key}!"
