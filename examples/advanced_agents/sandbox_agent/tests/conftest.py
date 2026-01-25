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
"""Pytest configuration and fixtures for sandbox agent tests."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from nat_sandbox_agent.sandbox.base import BaseSandbox
from nat_sandbox_agent.sandbox.base import CommandResult


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Create a mock sandbox instance for testing.

    Mocks the 5 core methods of BaseSandbox:
    - start, cleanup (lifecycle)
    - run_command (execution)
    - read_file, write_file (file I/O)
    """
    sandbox = MagicMock(spec=BaseSandbox)

    # Mock command execution
    sandbox.run_command = AsyncMock(return_value=CommandResult(exit_code=0, stdout="output", stderr=""))

    # Mock file operations
    sandbox.read_file = AsyncMock(return_value="file content")
    sandbox.write_file = AsyncMock(return_value=None)

    # Mock lifecycle
    sandbox.start = AsyncMock(return_value=None)
    sandbox.cleanup = AsyncMock(return_value=None)

    return sandbox


@pytest.fixture
def docker_sandbox_config() -> dict:
    """Configuration for Docker sandbox testing."""
    return {
        "type": "docker",
        "image": "python:3.12-slim",
        "memory_limit": "512m",
        "cpu_limit": 1.0,
        "network_enabled": True,
        "work_dir": "/workspace",
    }


@pytest.fixture
def daytona_sandbox_config() -> dict:
    """Configuration for Daytona sandbox testing."""
    return {
        "type": "daytona",
        "api_key": "test-api-key",
        "server_url": "https://api.daytona.io",
        "target": "us",
        "image": "daytonaio/workspace:latest",
        "cpu": 2,
        "memory": 4,
        "disk": 10,
    }
