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
"""Remote workspace implementation backed by a runtime API."""

import threading
import typing
from collections.abc import AsyncIterator

from pydantic import Field

from nat.cli.register_workflow import add_registered_workspace
from nat.data_models.workspace import WorkspaceBaseConfig
from nat.plugins.workspace.client import ApiWorkspace
from nat.plugins.workspace.client import ApiWorkspaceManagerBase
from nat.plugins.workspace.client import ApiWorkspaceSessionEntry
from nat.plugins.workspace.client import WorkspaceApiClient
from nat.plugins.workspace.types import ImagePullPolicy


class RemoteWorkspaceConfig(WorkspaceBaseConfig, name="remote"):
    """Configuration for remote workspaces."""

    runtime_api_url: str = Field(description="Runtime API endpoint for workspace execution.")
    runtime_api_key: str = Field(description="Runtime API key for workspace execution.")
    image: str = Field(description="Runtime image to use for the workspace.")
    image_pull_policy: ImagePullPolicy = Field(description="Policy for pulling the runtime image.", )
    http_timeout_seconds: float = Field(default=30.0, description="HTTP timeout for workspace API requests.")


class RemoteWorkspace(ApiWorkspace):
    """Remote workspace."""

    def __init__(self, config: RemoteWorkspaceConfig, api_client: WorkspaceApiClient, session_id: str | None = None):
        super().__init__(config=config, api_client=api_client, session_id=session_id)
        self.config = config


class RemoteWorkspaceManager(ApiWorkspaceManagerBase[RemoteWorkspaceConfig, RemoteWorkspace, None]):
    _sessions: typing.ClassVar[dict[str, ApiWorkspaceSessionEntry[None]]] = {}
    _session_lock: typing.ClassVar[threading.Lock] = threading.Lock()

    def _build_workspace(self, entry: ApiWorkspaceSessionEntry[None]) -> RemoteWorkspace:
        return RemoteWorkspace(config=self.config, api_client=entry.api_client, session_id=entry.session_id)

    def _endpoint_override(self) -> str | None:
        return self.config.endpoint or self.config.runtime_api_url

    def _session_headers(self, session_id: str) -> dict[str, str]:
        headers = super()._session_headers(session_id)
        if self.config.runtime_api_key:
            headers["Authorization"] = f"Bearer {self.config.runtime_api_key}"
        return headers

    async def _start_backend(self, session_id: str) -> tuple[str, None]:
        raise RuntimeError("Remote workspaces do not start a local backend.")


@add_registered_workspace(RemoteWorkspaceConfig)
async def remote_workspace(config: RemoteWorkspaceConfig) -> AsyncIterator["RemoteWorkspaceManager"]:
    yield RemoteWorkspaceManager(config)
