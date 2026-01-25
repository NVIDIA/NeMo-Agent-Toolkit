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
"""Tests for sandbox implementations."""

import pytest

from nat_sandbox_agent.sandbox.base import CommandResult
from nat_sandbox_agent.sandbox.factory import DaytonaSandboxConfig
from nat_sandbox_agent.sandbox.factory import DockerSandboxConfig
from nat_sandbox_agent.sandbox.factory import SandboxType
from nat_sandbox_agent.sandbox.factory import create_sandbox
from nat_sandbox_agent.sandbox.factory import create_sandbox_from_dict


class TestCommandResult:
    """Tests for CommandResult dataclass."""

    def test_success_property_true(self):
        """Test success property when exit code is 0."""
        result = CommandResult(exit_code=0, stdout="output", stderr="")
        assert result.success is True

    def test_success_property_false(self):
        """Test success property when exit code is non-zero."""
        result = CommandResult(exit_code=1, stdout="", stderr="error")
        assert result.success is False

    def test_to_dict(self):
        """Test to_dict method."""
        result = CommandResult(exit_code=0, stdout="output", stderr="error")
        d = result.to_dict()

        assert d["exit_code"] == 0
        assert d["stdout"] == "output"
        assert d["stderr"] == "error"
        assert d["success"] is True


class TestDockerSandboxConfig:
    """Tests for DockerSandboxConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DockerSandboxConfig()

        assert config.type == "docker"
        assert config.image == "python:3.12-slim"
        assert config.memory_limit == "512m"
        assert config.cpu_limit == 1.0
        assert config.network_enabled is True
        assert config.work_dir == "/workspace"
        assert config.auto_remove is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DockerSandboxConfig(
            image="custom:latest",
            memory_limit="2g",
            cpu_limit=4.0,
            network_enabled=False,
        )

        assert config.image == "custom:latest"
        assert config.memory_limit == "2g"
        assert config.cpu_limit == 4.0
        assert config.network_enabled is False


class TestDaytonaSandboxConfig:
    """Tests for DaytonaSandboxConfig."""

    def test_required_api_key(self):
        """Test that api_key is required."""
        with pytest.raises(ValueError):
            DaytonaSandboxConfig()

    def test_with_api_key(self):
        """Test configuration with api_key."""
        config = DaytonaSandboxConfig(api_key="test-key")

        assert config.type == "daytona"
        assert config.api_key == "test-key"
        assert config.server_url == "https://api.daytona.io"
        assert config.target == "us"
        assert config.cpu == 2
        assert config.memory == 4
        assert config.disk == 10


class TestSandboxFactory:
    """Tests for sandbox factory functions."""

    def test_create_docker_sandbox(self, docker_sandbox_config):
        """Test creating a Docker sandbox."""
        config = DockerSandboxConfig(**docker_sandbox_config)
        sandbox = create_sandbox(config)

        assert sandbox is not None
        # Check type without importing DockerSandbox to avoid circular imports
        assert sandbox.__class__.__name__ == "DockerSandbox"

    def test_create_sandbox_from_dict_docker(self, docker_sandbox_config):
        """Test creating a Docker sandbox from dict."""
        sandbox = create_sandbox_from_dict(docker_sandbox_config)

        assert sandbox is not None
        assert sandbox.__class__.__name__ == "DockerSandbox"

    def test_create_sandbox_from_dict_default(self):
        """Test creating sandbox with minimal config defaults to Docker."""
        sandbox = create_sandbox_from_dict({})

        assert sandbox is not None
        assert sandbox.__class__.__name__ == "DockerSandbox"

    def test_create_sandbox_invalid_type(self):
        """Test error handling for invalid sandbox type."""
        with pytest.raises(ValueError, match="Unknown sandbox type"):
            create_sandbox_from_dict({"type": "invalid"})


class TestSandboxType:
    """Tests for SandboxType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert SandboxType.DOCKER == "docker"
        assert SandboxType.DAYTONA == "daytona"
