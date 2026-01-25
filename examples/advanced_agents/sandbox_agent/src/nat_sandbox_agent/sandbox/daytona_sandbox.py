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

"""Daytona cloud-based sandbox implementation."""

import asyncio
import logging

from nat_sandbox_agent.sandbox.base import WORKSPACE_INIT_COMMAND
from nat_sandbox_agent.sandbox.base import BaseSandbox
from nat_sandbox_agent.sandbox.base import CommandResult

logger = logging.getLogger(__name__)


class DaytonaSandbox(BaseSandbox):
    """Daytona cloud-based sandbox implementation.

    This sandbox uses the Daytona cloud service to provide isolated
    execution environments.

    Note: Requires daytona-sdk package to be installed.
    """

    DEFAULT_IMAGE = "daytonaio/workspace:latest"

    def __init__(
        self,
        api_key: str,
        server_url: str = "https://api.daytona.io",
        target: str = "us",
        image: str = DEFAULT_IMAGE,
        cpu: int = 2,
        memory: int = 4,  # GB
        disk: int = 10,  # GB
        auto_stop_interval: int = 30,  # minutes
    ):
        """Initialize Daytona sandbox configuration.

        Args:
            api_key: Daytona API key for authentication.
            server_url: Daytona API server URL.
            target: Target region (e.g., "us", "eu").
            image: Docker image to use for the workspace.
            cpu: Number of CPU cores.
            memory: Memory in GB.
            disk: Disk space in GB.
            auto_stop_interval: Auto-stop interval in minutes (0 to disable).
        """
        self._api_key = api_key
        self._server_url = server_url
        self._target = target
        self._image = image
        self._cpu = cpu
        self._memory = memory
        self._disk = disk
        self._auto_stop_interval = auto_stop_interval

        self._client = None
        self._sandbox = None

    def _get_client(self):
        """Get or create Daytona client."""
        if self._client is None:
            try:
                from daytona_sdk import Daytona
                from daytona_sdk import DaytonaConfig

                config = DaytonaConfig(
                    api_key=self._api_key,
                    server_url=self._server_url,
                    target=self._target,
                )
                self._client = Daytona(config)
            except ImportError:
                raise ImportError(
                    "daytona-sdk package is required for DaytonaSandbox. "
                    "Install it with: pip install daytona-sdk"
                )
        return self._client

    async def start(self) -> None:
        """Start the Daytona sandbox."""
        logger.info("Starting Daytona sandbox")

        try:
            client = self._get_client()

            # Create sandbox configuration
            from daytona_sdk import CreateSandboxParams

            params = CreateSandboxParams(
                image=self._image,
                cpu=self._cpu,
                memory=self._memory,
                disk=self._disk,
                auto_stop_interval=self._auto_stop_interval,
            )

            # Create and start the sandbox
            self._sandbox = client.create(params)

            # Initialize workspace directories
            await self.run_command(WORKSPACE_INIT_COMMAND)

            logger.info(f"Daytona sandbox started: {self._sandbox.id}")

        except Exception as e:
            logger.error(f"Failed to start Daytona sandbox: {e}")
            raise

    async def cleanup(self) -> None:
        """Delete the Daytona sandbox."""
        if self._sandbox:
            logger.info(f"Cleaning up Daytona sandbox: {self._sandbox.id}")
            try:
                self._sandbox.delete()
                self._sandbox = None
            except Exception as e:
                logger.error(f"Failed to cleanup Daytona sandbox: {e}")
                raise
            finally:
                self._client = None

    async def run_command(
        self,
        command: str,
        working_dir: str = "/workspace",
        timeout: float = 120,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Execute a shell command in the Daytona sandbox."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")

        logger.debug(f"Executing command: {command[:100]}...")

        try:
            # Wrap synchronous SDK call with async timeout to prevent blocking
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._sandbox.process.exec(
                        command,
                        cwd=working_dir,
                        env=env,
                        timeout=int(timeout),
                    ),
                ),
                timeout=timeout,
            )

            return CommandResult(
                exit_code=result.exit_code,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )

        except TimeoutError:
            logger.warning(f"Command timed out after {timeout}s: {command[:50]}...")
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
            )
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
            )

    async def read_file(self, path: str) -> str:
        """Read file content from the Daytona sandbox."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")

        try:
            content = self._sandbox.fs.read_file(path)
            return content
        except Exception as e:
            if "not found" in str(e).lower():
                raise FileNotFoundError(f"File not found: {path}")
            logger.error(f"Failed to read file {path}: {e}")
            raise

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the Daytona sandbox."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")

        try:
            # Ensure parent directory exists
            dir_path = "/".join(path.split("/")[:-1])
            if dir_path:
                await self.run_command(f"mkdir -p {dir_path}")

            self._sandbox.fs.write_file(path, content)
        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            raise
