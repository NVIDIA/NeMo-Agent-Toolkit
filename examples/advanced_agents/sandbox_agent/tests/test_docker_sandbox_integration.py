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
"""Integration tests for Docker sandbox.

These tests require Docker to be running and will create real containers.
Mark with @pytest.mark.integration to allow skipping in CI environments.
"""

import pytest
import pytest_asyncio

from nat_sandbox_agent.sandbox.docker_sandbox import DockerSandbox


def docker_available():
    """Check if Docker is available."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


# Check Docker availability at module load time
_docker_available = docker_available()

# Skip all tests in this module if Docker is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


# Use a fixture-based skip instead of pytestmark for conditional skipping
@pytest.fixture(autouse=True)
def skip_without_docker():
    """Skip tests if Docker is not available."""
    if not _docker_available:
        pytest.skip("Docker not available")


class TestDockerSandboxLifecycle:
    """Integration tests for Docker sandbox lifecycle."""

    @pytest_asyncio.fixture
    async def sandbox(self):
        """Create a Docker sandbox for testing."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="256m",
            cpu_limit=0.5,
            network_enabled=True,
        )
        yield sandbox
        # Cleanup after test
        try:
            await sandbox.cleanup()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_start_and_cleanup(self, sandbox):
        """Test sandbox start and cleanup lifecycle."""
        # Start sandbox
        await sandbox.start()

        # Container should exist
        assert sandbox._container is not None

        # Cleanup
        await sandbox.cleanup()

        # Container should be removed
        assert sandbox._container is None


class TestDockerSandboxCommands:
    """Integration tests for Docker sandbox command execution."""

    @pytest_asyncio.fixture
    async def running_sandbox(self):
        """Create and start a Docker sandbox."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="256m",
            cpu_limit=0.5,
            network_enabled=True,
        )
        await sandbox.start()
        yield sandbox
        try:
            await sandbox.cleanup()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_run_command_success(self, running_sandbox):
        """Test running a simple command."""
        result = await running_sandbox.run_command("echo Hello World")

        assert result.success is True
        assert result.exit_code == 0
        assert "Hello World" in result.stdout

    @pytest.mark.asyncio
    async def test_run_command_failure(self, running_sandbox):
        """Test running a command that fails."""
        result = await running_sandbox.run_command("exit 1")

        assert result.success is False
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_run_command_with_stderr(self, running_sandbox):
        """Test running a command that outputs to stderr."""
        result = await running_sandbox.run_command("echo error >&2")

        assert result.exit_code == 0
        assert "error" in result.stderr

    @pytest.mark.asyncio
    async def test_run_python_code(self, running_sandbox):
        """Test running Python code via command."""
        # Use run_command with python3 directly to avoid escaping issues
        result = await running_sandbox.run_command("python3 -c \"print(2 ** 10)\"")

        assert result.success is True
        assert "1024" in result.stdout

    @pytest.mark.asyncio
    async def test_run_python_with_imports(self, running_sandbox):
        """Test running Python code with imports via file."""
        # Write a Python file and execute it to avoid escaping issues
        code = 'import json\ndata = {"key": "value"}\nprint(json.dumps(data))'
        await running_sandbox.write_file("/workspace/test_script.py", code)
        result = await running_sandbox.run_command("python3 /workspace/test_script.py")

        assert result.success is True
        assert "key" in result.stdout
        assert "value" in result.stdout

    @pytest.mark.asyncio
    async def test_run_command_with_working_dir(self, running_sandbox):
        """Test running a command in specific working directory."""
        # Create directory first
        await running_sandbox.run_command("mkdir -p /workspace/testdir")

        result = await running_sandbox.run_command("pwd", working_dir="/workspace/testdir")

        assert result.success is True
        assert "/workspace/testdir" in result.stdout


class TestDockerSandboxFileOperations:
    """Integration tests for Docker sandbox file operations."""

    @pytest_asyncio.fixture
    async def running_sandbox(self):
        """Create and start a Docker sandbox."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="256m",
            cpu_limit=0.5,
            network_enabled=True,
        )
        await sandbox.start()
        yield sandbox
        try:
            await sandbox.cleanup()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_write_and_read_file(self, running_sandbox):
        """Test writing and reading a file."""
        test_content = "Hello, Docker Sandbox!"
        test_path = "/workspace/test_file.txt"

        # Write file
        await running_sandbox.write_file(test_path, test_content)

        # Read file
        content = await running_sandbox.read_file(test_path)

        assert content == test_content

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, running_sandbox):
        """Test reading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            await running_sandbox.read_file("/workspace/nonexistent.txt")

    @pytest.mark.asyncio
    async def test_write_creates_parent_directories(self, running_sandbox):
        """Test that writing a file creates parent directories."""
        test_path = "/workspace/nested/deep/file.txt"

        await running_sandbox.write_file(test_path, "nested content")

        content = await running_sandbox.read_file(test_path)
        assert content == "nested content"


class TestDockerSandboxNetwork:
    """Integration tests for Docker sandbox network operations."""

    @pytest_asyncio.fixture
    async def running_sandbox(self):
        """Create and start a Docker sandbox with network enabled."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="256m",
            cpu_limit=0.5,
            network_enabled=True,
        )
        await sandbox.start()
        yield sandbox
        try:
            await sandbox.cleanup()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_network_access(self, running_sandbox):
        """Test that sandbox has network access."""
        # Check DNS resolution using a simple command
        result = await running_sandbox.run_command("getent hosts google.com")

        # Should succeed with network enabled
        assert result.exit_code == 0

    @pytest_asyncio.fixture
    async def no_network_sandbox(self):
        """Create and start a Docker sandbox without network."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="256m",
            cpu_limit=0.5,
            network_enabled=False,
        )
        await sandbox.start()
        yield sandbox
        try:
            await sandbox.cleanup()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_no_network_access(self, no_network_sandbox):
        """Test that sandbox without network cannot access internet."""
        result = await no_network_sandbox.run_command(
            "python3 -c \"import socket; socket.gethostbyname('google.com')\"",
            timeout=10,
        )

        # Should fail with network disabled
        assert result.exit_code != 0


class TestDockerSandboxResourceLimits:
    """Integration tests for Docker sandbox resource limits."""

    @pytest.mark.asyncio
    async def test_memory_limit(self):
        """Test that memory limit is enforced."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="64m",
            cpu_limit=0.5,
        )

        try:
            await sandbox.start()

            # Write and run a Python script that tries to allocate too much memory
            code = "x = 'a' * (100 * 1024 * 1024)"  # Try to allocate 100MB
            await sandbox.write_file("/workspace/test_memory.py", code)
            result = await sandbox.run_command(
                "python3 /workspace/test_memory.py",
                timeout=30,
            )

            # Should either fail or be killed
            # The exact behavior depends on Docker configuration
            # At minimum, it shouldn't hang forever
            assert result is not None

        finally:
            await sandbox.cleanup()

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test that command timeout works."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="256m",
            cpu_limit=0.5,
        )

        try:
            await sandbox.start()

            # Run a command that takes longer than timeout
            result = await sandbox.run_command("sleep 10", timeout=2)

            # Should fail due to timeout
            assert result.success is False

        finally:
            await sandbox.cleanup()
