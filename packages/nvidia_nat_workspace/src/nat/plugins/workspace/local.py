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
"""Local workspace implementation backed by a workspace API."""

from __future__ import annotations

import threading
import typing
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
from nat.plugins.workspace.server import WorkspaceApiServerHandle
from nat.plugins.workspace.server import start_workspace_api_server
from nat.plugins.workspace.server import stop_workspace_api_server


class LocalWorkspaceConfig(WorkspaceBaseConfig, name="local"):
    """Configuration for local workspaces."""
    http_timeout_seconds: float = Field(default=30.0, description="HTTP timeout for workspace API requests.")


class LocalWorkspace(ApiWorkspace):
    """Local workspace."""

    def __init__(self, config: LocalWorkspaceConfig, api_client: WorkspaceApiClient, session_id: str | None = None):
        super().__init__(config=config, api_client=api_client, session_id=session_id)
        self.config = config


@dataclass
class _LocalBackend:
    server: WorkspaceApiServerHandle | None


class LocalWorkspaceManager(ApiWorkspaceManagerBase[LocalWorkspaceConfig, LocalWorkspace, _LocalBackend]):
    _sessions: typing.ClassVar[dict[str, ApiWorkspaceSessionEntry[_LocalBackend]]] = {}
    _session_lock: typing.ClassVar[threading.Lock] = threading.Lock()

    def _build_workspace(self, entry: ApiWorkspaceSessionEntry[_LocalBackend]) -> LocalWorkspace:
        session_root = self._session_working_directory(entry.session_id)
        session_config = self.config.model_copy(update={"working_directory": session_root})
        return LocalWorkspace(config=session_config, api_client=entry.api_client, session_id=entry.session_id)

    def _session_working_directory(self, session_id: str) -> Path:
        session_root = self.config.working_directory.resolve() / session_id
        session_root.mkdir(parents=True, exist_ok=True)
        return session_root

    async def _start_backend(self, session_id: str) -> tuple[str, _LocalBackend]:
        server = start_workspace_api_server(self.config.working_directory)
        return server.base_url, _LocalBackend(server=server)

    async def _shutdown_backend(self, backend: _LocalBackend | None) -> None:
        if backend is None or backend.server is None:
            return
        stop_workspace_api_server(backend.server)


@add_registered_workspace(LocalWorkspaceConfig)
async def local_workspace(config: LocalWorkspaceConfig) -> AsyncIterator[LocalWorkspaceManager]:
    yield LocalWorkspaceManager(config)
