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

"""Base sandbox interface - Abstract base class for sandbox implementations."""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from types import TracebackType

# ============ Workspace Constants ============
# These constants define the standard workspace directory structure.
# All sandbox implementations should use these paths.

WORKSPACE_ROOT = "/workspace"
WORKSPACE_INPUT = f"{WORKSPACE_ROOT}/input"
WORKSPACE_OUTPUT = f"{WORKSPACE_ROOT}/output"
WORKSPACE_TEMP = f"{WORKSPACE_ROOT}/temp"
WORKSPACE_DOWNLOADS = f"{WORKSPACE_ROOT}/downloads"

# Command to initialize workspace directories
WORKSPACE_INIT_COMMAND = (
    f"mkdir -p {WORKSPACE_INPUT} {WORKSPACE_OUTPUT} {WORKSPACE_TEMP} {WORKSPACE_DOWNLOADS}"
)

# Default script path for Python execution
DEFAULT_SCRIPT_PATH = f"{WORKSPACE_TEMP}/_script.py"


@dataclass
class CommandResult:
    """Result of a command execution in the sandbox."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Check if the command executed successfully."""
        return self.exit_code == 0

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "success": self.success,
        }


class BaseSandbox(ABC):
    """Abstract base class for sandbox implementations.

    This class defines the minimal interface that all sandbox implementations
    (Docker, Daytona, etc.) must follow. It provides 5 core methods:
    - Lifecycle: start, cleanup
    - Execution: run_command
    - File I/O: read_file, write_file
    """

    # ============ Lifecycle Management ============

    @abstractmethod
    async def start(self) -> None:
        """Start the sandbox environment."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Destroy the sandbox completely (cannot be recovered)."""
        pass

    # ============ Command Execution ============

    @abstractmethod
    async def run_command(
        self,
        command: str,
        working_dir: str = "/workspace",
        timeout: float = 120,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Execute a shell command in the sandbox.

        Args:
            command: The shell command to execute.
            working_dir: Working directory for the command.
            timeout: Maximum execution time in seconds.
            env: Additional environment variables.

        Returns:
            CommandResult: The result of the command execution.
        """
        pass

    # ============ File Operations ============

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read file content from the sandbox.

        Args:
            path: Path to the file in the sandbox.

        Returns:
            str: File content as string.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        pass

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the sandbox.

        Args:
            path: Path to the file in the sandbox.
            content: Content to write.
        """
        pass

    # ============ Context Manager Support ============

    async def __aenter__(self) -> "BaseSandbox":
        """Async context manager entry - starts the sandbox."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit - cleans up the sandbox."""
        await self.cleanup()
