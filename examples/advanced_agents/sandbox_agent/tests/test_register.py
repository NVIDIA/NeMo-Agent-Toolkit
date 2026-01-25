# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for workflow registration configuration."""

import pytest

from nat_sandbox_agent.register import AgentState
from nat_sandbox_agent.register import SandboxAgentWorkflowConfig
from nat_sandbox_agent.sandbox.factory import DaytonaSandboxConfig
from nat_sandbox_agent.sandbox.factory import DockerSandboxConfig
from nat_sandbox_agent.sandbox.factory import SandboxType


class TestDockerSandboxConfig:
    """Tests for DockerSandboxConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = DockerSandboxConfig()

        assert config.type == "docker"
        assert config.image == "python:3.12-slim"
        assert config.memory_limit == "512m"
        assert config.cpu_limit == 1.0
        assert config.network_enabled is True
        assert config.work_dir == "/workspace"
        assert config.auto_remove is False

    def test_custom_values(self):
        """Test Docker-specific configuration."""
        config = DockerSandboxConfig(
            image="nat-sandbox:latest",
            memory_limit="4g",
            cpu_limit=4.0,
            network_enabled=False,
        )

        assert config.type == "docker"
        assert config.image == "nat-sandbox:latest"
        assert config.memory_limit == "4g"
        assert config.cpu_limit == 4.0
        assert config.network_enabled is False

    def test_volumes_config(self):
        """Test volume mounting configuration."""
        config = DockerSandboxConfig(volumes={"/host/path": "/container/path"}, )

        assert config.volumes == {"/host/path": "/container/path"}

    def test_environment_config(self):
        """Test environment variables configuration."""
        config = DockerSandboxConfig(
            environment={"MY_VAR": "value"},
            pass_env_vars=["TAVILY_API_KEY"],
        )

        assert config.environment == {"MY_VAR": "value"}
        assert config.pass_env_vars == ["TAVILY_API_KEY"]


class TestDaytonaSandboxConfig:
    """Tests for DaytonaSandboxConfig class."""

    def test_api_key_required(self):
        """Test that api_key is required."""
        with pytest.raises(Exception):  # Pydantic validation error
            DaytonaSandboxConfig()

    def test_default_values(self):
        """Test default values with required api_key."""
        config = DaytonaSandboxConfig(api_key="test-key")

        assert config.type == "daytona"
        assert config.api_key == "test-key"
        assert config.server_url == "https://api.daytona.io"
        assert config.target == "us"
        assert config.image == "daytonaio/workspace:latest"
        assert config.cpu == 2
        assert config.memory == 4
        assert config.disk == 10
        assert config.auto_stop_interval == 30

    def test_custom_values(self):
        """Test Daytona-specific configuration."""
        config = DaytonaSandboxConfig(
            api_key="test-api-key",
            server_url="https://custom.daytona.io",
            target="eu",
            cpu=4,
            memory=8,
        )

        assert config.type == "daytona"
        assert config.api_key == "test-api-key"
        assert config.server_url == "https://custom.daytona.io"
        assert config.target == "eu"
        assert config.cpu == 4
        assert config.memory == 8


class TestSandboxType:
    """Tests for SandboxType enum."""

    def test_enum_values(self):
        """Test SandboxType enum values."""
        assert SandboxType.DOCKER == "docker"
        assert SandboxType.DAYTONA == "daytona"


class TestSandboxAgentWorkflowConfig:
    """Tests for SandboxAgentWorkflowConfig class."""

    def test_llm_name_required(self):
        """Test that llm_name is required."""
        with pytest.raises(Exception):  # Pydantic validation error
            SandboxAgentWorkflowConfig()

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SandboxAgentWorkflowConfig(llm_name="test_llm")

        assert config.llm_name == "test_llm"
        assert config.max_iterations == 20
        assert config.max_observation_tokens == 10000
        assert config.enabled_tools is None
        assert config.system_prompt is None
        assert config.additional_instructions is None

    def test_default_sandbox_config(self):
        """Test that default sandbox config is properly set."""
        config = SandboxAgentWorkflowConfig(llm_name="test_llm")

        assert config.sandbox_config["type"] == "docker"
        assert config.sandbox_config["image"] == "python:3.12-slim"
        assert config.sandbox_config["memory_limit"] == "1g"
        assert config.sandbox_config["cpu_limit"] == 2.0
        assert config.sandbox_config["network_enabled"] is True

    def test_custom_sandbox_config(self):
        """Test custom sandbox configuration."""
        config = SandboxAgentWorkflowConfig(
            llm_name="test_llm",
            sandbox_config={
                "type": "docker",
                "image": "custom:latest",
                "memory_limit": "2g",
            },
        )

        assert config.sandbox_config["image"] == "custom:latest"
        assert config.sandbox_config["memory_limit"] == "2g"

    def test_enabled_tools_filtering(self):
        """Test that enabled_tools can be specified."""
        config = SandboxAgentWorkflowConfig(
            llm_name="test_llm",
            enabled_tools=["shell", "python", "web_search"],
        )

        assert config.enabled_tools == ["shell", "python", "web_search"]

    def test_custom_prompts(self):
        """Test custom system prompt and instructions."""
        config = SandboxAgentWorkflowConfig(
            llm_name="test_llm",
            system_prompt="Custom system prompt",
            additional_instructions="Extra instructions",
        )

        assert config.system_prompt == "Custom system prompt"
        assert config.additional_instructions == "Extra instructions"

    def test_max_iterations_range(self):
        """Test various max_iterations values."""
        # Should accept positive integers
        config = SandboxAgentWorkflowConfig(llm_name="test_llm", max_iterations=50)
        assert config.max_iterations == 50

        config = SandboxAgentWorkflowConfig(llm_name="test_llm", max_iterations=1)
        assert config.max_iterations == 1


class TestAgentState:
    """Tests for AgentState TypedDict."""

    def test_agent_state_structure(self):
        """Test that AgentState has expected keys."""
        state: AgentState = {
            "messages": [],
            "iteration_count": 0,
            "sandbox_id": "test-id",
        }

        assert "messages" in state
        assert "iteration_count" in state
        assert "sandbox_id" in state

    def test_agent_state_messages(self):
        """Test that messages can hold message objects."""
        from langchain_core.messages import HumanMessage

        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "iteration_count": 0,
            "sandbox_id": "test-id",
        }

        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "Hello"

    def test_agent_state_iteration_tracking(self):
        """Test iteration count tracking."""
        state: AgentState = {
            "messages": [],
            "iteration_count": 5,
            "sandbox_id": "test-id",
        }

        assert state["iteration_count"] == 5
