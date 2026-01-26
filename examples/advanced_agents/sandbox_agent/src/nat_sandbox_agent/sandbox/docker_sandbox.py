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

"""Docker-based sandbox implementation."""

import asyncio
import io
import logging
import shlex
import tarfile
import uuid

from docker.errors import ContainerError
from docker.errors import ImageNotFound
from docker.errors import NotFound

import docker
from nat_sandbox_agent.sandbox.base import WORKSPACE_INIT_COMMAND
from nat_sandbox_agent.sandbox.base import WORKSPACE_ROOT
from nat_sandbox_agent.sandbox.base import BaseSandbox
from nat_sandbox_agent.sandbox.base import CommandResult

logger = logging.getLogger(__name__)


class DockerSandbox(BaseSandbox):
    """Docker-based sandbox implementation.

    This sandbox uses Docker containers to provide isolated execution
    environments. Each sandbox instance corresponds to a single container
    with its own filesystem, network, and process space.
    """

    DEFAULT_IMAGE = "python:3.12-slim"
    DEFAULT_WORK_DIR = WORKSPACE_ROOT

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
        network_enabled: bool = True,
        work_dir: str = DEFAULT_WORK_DIR,
        container_name: str | None = None,
        auto_remove: bool = False,
        environment: dict[str, str] | None = None,
        volumes: dict[str, dict[str, str]] | None = None,
    ):
        """Initialize Docker sandbox configuration.

        Args:
            image: Docker image to use for the container.
            memory_limit: Memory limit (e.g., "512m", "1g").
            cpu_limit: CPU limit (number of CPUs).
            network_enabled: Whether to enable network access.
            work_dir: Working directory inside the container.
            container_name: Optional container name (auto-generated if None).
            auto_remove: Whether to automatically remove container on stop.
            environment: Environment variables to pass to the container.
            volumes: Volume mounts dict {host_path: {"bind": container_path, "mode": "rw"}}.
        """
        self._image = image
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit
        self._network_enabled = network_enabled
        self._work_dir = work_dir
        self._container_name = container_name or f"sandbox_{uuid.uuid4().hex[:8]}"
        self._auto_remove = auto_remove
        self._environment = environment or {}
        self._volumes = volumes or {}

        self._client: docker.DockerClient | None = None
        self._container = None

    async def start(self) -> None:
        """Start the Docker container sandbox."""
        logger.info(f"Starting Docker sandbox: {self._container_name}")

        try:
            self._client = docker.from_env()

            # Ensure image is available
            try:
                self._client.images.get(self._image)
            except ImageNotFound:
                logger.info(f"Pulling image: {self._image}")
                await asyncio.get_running_loop().run_in_executor(
                    None, self._client.images.pull, self._image
                )

            # Container configuration
            container_config = {
                "image": self._image,
                "name": self._container_name,
                "detach": True,
                "tty": True,
                "stdin_open": True,
                "working_dir": self._work_dir,
                "mem_limit": self._memory_limit,
                "nano_cpus": int(self._cpu_limit * 1e9),
                "network_mode": "bridge" if self._network_enabled else "none",
                "command": "/bin/bash",
                "auto_remove": self._auto_remove,
                "environment": self._environment,
            }

            # Add volume mounts if specified
            if self._volumes:
                container_config["volumes"] = self._volumes
                logger.info(f"Mounting volumes: {list(self._volumes.keys())}")

            # Create and start container
            self._container = await asyncio.get_running_loop().run_in_executor(
                None, lambda: self._client.containers.create(**container_config)
            )

            await asyncio.get_running_loop().run_in_executor(
                None, self._container.start
            )

            # Initialize workspace directories
            await self.run_command(WORKSPACE_INIT_COMMAND)

            logger.info(f"Docker sandbox started: {self._container_name}")

        except Exception as e:
            logger.error(f"Failed to start Docker sandbox: {e}")
            # Close client to prevent resource leak
            if self._client:
                self._client.close()
                self._client = None
            raise

    async def cleanup(self) -> None:
        """Remove the Docker container completely."""
        if self._container:
            logger.info(f"Cleaning up Docker sandbox: {self._container_name}")
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self._container.remove(force=True)
                )
                self._container = None
            except NotFound:
                # Container already removed
                self._container = None
            except Exception as e:
                logger.error(f"Failed to cleanup Docker sandbox: {e}")
                raise
            finally:
                if self._client:
                    self._client.close()
                    self._client = None

    async def run_command(
        self,
        command: str,
        working_dir: str = "/workspace",
        timeout: float = 120,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Execute a shell command in the container."""
        if not self._container:
            raise RuntimeError("Sandbox not started")

        logger.debug(f"Executing command: {command[:100]}...")

        # Use Linux timeout command to ensure container-side process termination.
        # Without this, asyncio.wait_for() only cancels the Python await,
        # but the process inside the container continues running as orphan.
        timeout_int = int(timeout)
        wrapped_command = f"timeout {timeout_int} /bin/bash -c {shlex.quote(command)}"

        try:
            exec_result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._container.exec_run(
                        cmd=wrapped_command,
                        workdir=working_dir,
                        environment=env,
                        demux=True,
                    ),
                ),
                timeout=timeout + 5,  # Give container timeout a chance to fire first
            )

            exit_code = exec_result.exit_code
            stdout_bytes, stderr_bytes = exec_result.output

            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            # Linux timeout command returns 124 when the command times out
            if exit_code == 124:
                logger.warning(f"Command timed out after {timeout_int}s: {command[:50]}...")
                return CommandResult(
                    exit_code=-1,
                    stdout=stdout,
                    stderr=f"Command timed out after {timeout_int} seconds\n{stderr}".strip(),
                )

            return CommandResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
            )

        except TimeoutError:
            logger.warning(f"Command timed out after {timeout}s: {command[:50]}...")
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
            )
        except ContainerError as e:
            logger.exception("Container error")
            return CommandResult(
                exit_code=e.exit_status,
                stdout="",
                stderr=str(e),
            )
        except Exception as e:
            logger.exception("Command execution failed")
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
            )

    async def read_file(self, path: str) -> str:
        """Read file content from the container."""
        if not self._container:
            raise RuntimeError("Sandbox not started")

        def _extract_from_archive():
            """Synchronous helper to fetch and extract file from container."""
            bits, _ = self._container.get_archive(path)

            # Extract file content from tar archive
            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)

            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                member = tar.getmembers()[0]
                file_obj = tar.extractfile(member)
                if file_obj:
                    return file_obj.read().decode("utf-8", errors="replace")
                raise FileNotFoundError(f"File not found: {path}")

        try:
            # Wrap all blocking I/O in run_in_executor
            return await asyncio.get_running_loop().run_in_executor(
                None, _extract_from_archive
            )

        except NotFound:
            raise FileNotFoundError(f"File not found: {path}") from None
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the container."""
        if not self._container:
            raise RuntimeError("Sandbox not started")

        try:
            # Create tar archive with the file
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                data = content.encode("utf-8")
                tarinfo = tarfile.TarInfo(name=path.split("/")[-1])
                tarinfo.size = len(data)
                tar.addfile(tarinfo, io.BytesIO(data))

            tar_stream.seek(0)

            # Get directory path
            dir_path = "/".join(path.split("/")[:-1]) or "/"

            # Ensure directory exists (shell-escape to prevent injection)
            await self.run_command(f"mkdir -p {shlex.quote(dir_path)}")

            # Upload the tar archive
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._container.put_archive(dir_path, tar_stream.getvalue()),
            )

        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            raise
