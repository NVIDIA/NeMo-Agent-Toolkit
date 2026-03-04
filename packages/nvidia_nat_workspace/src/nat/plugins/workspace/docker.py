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
"""Docker-backed workspace implementation."""

import asyncio
import socket
import threading
import typing
from collections.abc import AsyncIterator
from dataclasses import dataclass

from pydantic import Field

from nat.cli.register_workflow import add_registered_workspace
from nat.data_models.workspace import WorkspaceBaseConfig
from nat.plugins.workspace.client import ApiWorkspace
from nat.plugins.workspace.client import ApiWorkspaceManagerBase
from nat.plugins.workspace.client import ApiWorkspaceSessionEntry
from nat.plugins.workspace.client import WorkspaceApiClient
from nat.plugins.workspace.server import _wait_for_server
from nat.plugins.workspace.types import ImagePullPolicy
from nat.plugins.workspace.types import Mount

if typing.TYPE_CHECKING:
    from docker.client import DockerClient


class DockerWorkspaceConfig(WorkspaceBaseConfig, name="docker"):
    """Configuration for Docker workspaces."""

    mounts: list[Mount] = Field(
        default_factory=list,
        description="List of (host_path, container_path, access_mode) tuples.",
    )
    image: str = Field(description="Docker image to use for the workspace.")
    image_pull_policy: ImagePullPolicy = Field(description="Policy for pulling the Docker image.")
    container_port: int = Field(default=8000, description="Container port for the workspace API endpoint.")
    http_timeout_seconds: float = Field(default=30.0, description="HTTP timeout for workspace API requests.")
    environment: dict[str, str] = Field(default_factory=dict, description="Environment variables for the container.")
    command: str | list[str] | None = Field(
        default=None,
        description="Command to start the workspace API server inside the container.",
    )


class DockerWorkspace(ApiWorkspace):
    """Docker workspace."""

    def __init__(self,
                 config: DockerWorkspaceConfig,
                 api_client: WorkspaceApiClient,
                 container_id: str | None = None,
                 session_id: str | None = None):
        super().__init__(config=config, api_client=api_client, session_id=session_id)
        self.config = config
        self.container_id = container_id


@dataclass
class _DockerBackend:
    container_id: str | None


class DockerWorkspaceManager(ApiWorkspaceManagerBase[DockerWorkspaceConfig, DockerWorkspace, _DockerBackend]):
    _sessions: typing.ClassVar[dict[str, ApiWorkspaceSessionEntry[_DockerBackend]]] = {}
    _session_lock: typing.ClassVar[threading.Lock] = threading.Lock()

    def _build_workspace(self, entry: ApiWorkspaceSessionEntry[_DockerBackend]) -> DockerWorkspace:
        container_id = entry.backend.container_id if entry.backend else None
        return DockerWorkspace(
            config=self.config,
            api_client=entry.api_client,
            container_id=container_id,
            session_id=entry.session_id,
        )

    async def _start_backend(self, session_id: str) -> tuple[str, _DockerBackend]:
        container_id, base_url = await asyncio.to_thread(self._start_container, session_id)
        await asyncio.to_thread(_wait_for_server, "127.0.0.1", int(base_url.rsplit(":", 1)[1]), 10.0)
        return base_url, _DockerBackend(container_id=container_id)

    async def _shutdown_backend(self, backend: _DockerBackend | None) -> None:
        if backend is None or backend.container_id is None:
            return
        await asyncio.to_thread(self._stop_container, backend.container_id)

    @staticmethod
    def _import_docker() -> typing.Any:
        try:
            import docker
        except ImportError as exc:
            raise ImportError(
                "docker is required for Docker workspaces. Install with " + \
                    "`uv pip install 'nvidia-nat-workspace[docker]'`."
            ) from exc
        return docker

    def _volume_bindings(self) -> dict[str, dict[str, str]]:
        volumes: dict[str, dict[str, str]] = {}
        for mount in self.config.mounts:
            mode = "ro" if mount.access_mode == "r" else mount.access_mode
            volumes[str(mount.host_path)] = {"bind": str(mount.container_path), "mode": mode}
        return volumes

    def _ensure_image(self, docker_module, docker_client: "DockerClient") -> None:

        policy = self.config.image_pull_policy
        image = self.config.image

        if policy == "Always":
            docker_client.images.pull(image)
            return

        try:
            docker_client.images.get(image)
        except docker_module.errors.ImageNotFound as exc:
            if policy == "Never":
                raise RuntimeError(f"Docker image {image} not found.") from exc
            docker_client.images.pull(image)

    def _allocate_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def _start_container(self, session_id: str) -> tuple[str, str]:
        docker_module: typing.Any = self._import_docker()
        docker_client = docker_module.from_env()
        self._ensure_image(docker_module, docker_client)

        host_port = self._allocate_port()
        environment = dict(self.config.environment)
        environment.setdefault("NAT_WORKSPACE_ROOT", str(self.config.working_directory))
        command = self.config.command
        if command is None:
            command = [
                "python",
                "-m",
                "nat.plugins.workspace.server",
                "--host",
                "0.0.0.0",
                "--port",
                str(self.config.container_port),
            ]

        container = docker_client.containers.run(
            self.config.image,
            detach=True,
            ports={f"{self.config.container_port}/tcp": host_port},
            volumes=self._volume_bindings(),
            environment=environment or None,
            command=command,
            labels={"nat_session_id": session_id},
        )
        docker_client.close()
        base_url = f"http://127.0.0.1:{host_port}"
        return container.id, base_url

    def _stop_container(self, container_id: str) -> None:
        docker_module: typing.Any = self._import_docker()
        docker_client = docker_module.from_env()
        try:
            container = docker_client.containers.get(container_id)
        except docker_module.errors.NotFound:
            docker_client.close()
            return

        try:
            container.stop()
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass
            docker_client.close()


@add_registered_workspace(DockerWorkspaceConfig)
async def docker_workspace(config: DockerWorkspaceConfig) -> AsyncIterator[DockerWorkspaceManager]:
    yield DockerWorkspaceManager(config)
