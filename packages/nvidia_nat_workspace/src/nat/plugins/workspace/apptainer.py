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
"""Apptainer-backed workspace implementation."""

import asyncio
import hashlib
import os
import socket
import subprocess
import threading
import typing
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

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
    from cloudmesh.apptainer.apptainer import Apptainer


class ApptainerWorkspaceConfig(WorkspaceBaseConfig, name="apptainer"):
    """Configuration for Apptainer workspaces."""

    mounts: list[Mount] = Field(
        default_factory=list,
        description="List of (host_path, container_path, access_mode) tuples.",
    )
    image: str = Field(description="Apptainer image to use for the workspace.")
    image_pull_policy: ImagePullPolicy = Field(description="Policy for pulling the Apptainer image.")
    http_timeout_seconds: float = Field(default=30.0, description="HTTP timeout for workspace API requests.")
    apptainer_options: list[str] = Field(
        default_factory=list,
        description="Extra Apptainer options for instance startup.",
    )


class ApptainerWorkspace(ApiWorkspace):
    """Apptainer workspace."""

    def __init__(self,
                 config: ApptainerWorkspaceConfig,
                 api_client: WorkspaceApiClient,
                 instance_name: str | None = None,
                 session_id: str | None = None):
        super().__init__(config=config, api_client=api_client, session_id=session_id)
        self.config = config
        self.instance_name = instance_name


@dataclass
class _ApptainerBackend:
    instance_name: str | None
    server_process: subprocess.Popen[str] | None


class ApptainerWorkspaceManager(ApiWorkspaceManagerBase[ApptainerWorkspaceConfig, ApptainerWorkspace,
                                                        _ApptainerBackend]):
    _sessions: typing.ClassVar[dict[str, ApiWorkspaceSessionEntry[_ApptainerBackend]]] = {}
    _session_lock: typing.ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config: ApptainerWorkspaceConfig):
        super().__init__(config=config)
        self._apptainer: Apptainer | None = None

    def _build_workspace(self, entry: ApiWorkspaceSessionEntry[_ApptainerBackend]) -> ApptainerWorkspace:
        instance_name = entry.backend.instance_name if entry.backend else None
        return ApptainerWorkspace(
            config=self.config,
            api_client=entry.api_client,
            instance_name=instance_name,
            session_id=entry.session_id,
        )

    async def _start_backend(self, session_id: str) -> tuple[str, _ApptainerBackend]:
        instance_name, base_url, server_process = await asyncio.to_thread(self._start_instance, session_id)
        return base_url, _ApptainerBackend(instance_name=instance_name, server_process=server_process)

    async def _shutdown_backend(self, backend: _ApptainerBackend | None) -> None:
        if backend is None or backend.instance_name is None:
            return
        await asyncio.to_thread(self._stop_instance, backend.instance_name, backend.server_process)

    def _import_apptainer(self) -> "Apptainer":
        try:
            from cloudmesh.apptainer.apptainer import Apptainer
        except ImportError as exc:
            raise ImportError("cloudmesh-apptainer is required for Apptainer workspaces. "
                              "Install with `uv pip install 'nvidia-nat-workspace[apptainer]'`.") from exc
        return Apptainer()

    def _build_bind_options(self) -> list[str]:
        options: list[str] = []
        for mount in self.config.mounts:
            mode = "ro" if mount.access_mode == "r" else mount.access_mode
            options.append(f"--bind {mount.host_path}:{mount.container_path}:{mode}")
        options.extend(self.config.apptainer_options)
        return options

    def _resolve_image_name(self) -> str:
        assert self._apptainer is not None
        image_ref = self.config.image
        image_path = Path(image_ref)
        if image_path.exists():
            self._apptainer.add_location(str(image_path))
            return image_path.name

        if "://" in image_ref:
            if self.config.image_pull_policy == "Never":
                raise RuntimeError(f"Apptainer image {image_ref} is not available locally.")
            images_dir = self.config.working_directory / ".apptainer"
            images_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha1(image_ref.encode("utf-8")).hexdigest()[:10]
            target_path = images_dir / f"{digest}.sif"
            if self.config.image_pull_policy == "Always" or not target_path.exists():
                self._apptainer.download(name=str(target_path), url=image_ref)
            self._apptainer.add_location(str(target_path))
            return target_path.name

        try:
            self._apptainer.find_image(image_ref)
            return image_ref
        except Exception as exc:
            raise RuntimeError(f"Apptainer image {image_ref} not found.") from exc

    def _allocate_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def _start_instance(self, session_id: str) -> tuple[str, str, subprocess.Popen[str]]:
        self._apptainer = self._import_apptainer()
        instance_name = f"nat-workspace-{uuid.uuid4().hex[:10]}"
        image_name = self._resolve_image_name()
        self._apptainer.start(
            name=instance_name,
            image=image_name,
            options=self._build_bind_options(),
            clean=False,
        )

        host_port = self._allocate_port()
        env = os.environ.copy()
        env["APPTAINERENV_NAT_WORKSPACE_ROOT"] = str(self.config.working_directory)
        command = [
            "apptainer",
            "exec",
            f"instance://{instance_name}",
            "python",
            "-m",
            "nat.plugins.workspace.server",
            "--host",
            "0.0.0.0",
            "--port",
            str(host_port),
        ]
        server_process = subprocess.Popen(command,
                                          env=env,
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL,
                                          text=True)
        _wait_for_server("127.0.0.1", host_port, 10.0)
        base_url = f"http://127.0.0.1:{host_port}"
        return instance_name, base_url, server_process

    def _stop_instance(self, instance_name: str, server_process: subprocess.Popen[str] | None) -> None:
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

        apptainer = self._import_apptainer()
        apptainer.stop(name=instance_name)


@add_registered_workspace(ApptainerWorkspaceConfig)
async def apptainer_workspace(config: ApptainerWorkspaceConfig) -> AsyncIterator[ApptainerWorkspaceManager]:
    yield ApptainerWorkspaceManager(config)
