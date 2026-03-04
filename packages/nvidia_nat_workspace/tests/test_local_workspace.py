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
from __future__ import annotations

import typing
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import nat.plugins.workspace.client as api_client_module
from nat.plugins.workspace import local as local_module
from nat.plugins.workspace.local import LocalWorkspaceConfig
from nat.plugins.workspace.local import LocalWorkspaceManager
from nat.plugins.workspace.server import WorkspaceApiServerHandle


@pytest.fixture(name="fixture_httpx_clients")
def fixture_httpx_clients(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Patch httpx.AsyncClient to capture construction parameters."""
    clients: list[Any] = []

    class FakeAsyncClient:

        def __init__(self, base_url: str, headers: dict[str, str] | None, timeout: httpx.Timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout
            self.post_calls: list[str] = []
            self.closed = False
            clients.append(self)

        async def post(self, path: str) -> None:
            self.post_calls.append(path)

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(api_client_module.httpx, "AsyncClient", FakeAsyncClient)
    return clients


@pytest.fixture(name="fixture_reset_local_sessions")
def fixture_reset_local_sessions() -> None:
    """Reset session cache for local workspace tests."""
    LocalWorkspaceManager._sessions = {}


@pytest.fixture(name="fixture_server_handle")
def fixture_server_handle(tmp_path: Path) -> WorkspaceApiServerHandle:
    """Return a fake workspace server handle."""
    return WorkspaceApiServerHandle(
        process=typing.cast(BaseProcess, SimpleNamespace()),
        host="127.0.0.1",
        port=8123,
        root_path=tmp_path / "workspace-root",
    )


@pytest.fixture(name="fixture_server_controls")
def fixture_server_controls(
    monkeypatch: pytest.MonkeyPatch,
    fixture_server_handle: WorkspaceApiServerHandle,
) -> dict[str, Any]:
    """Patch workspace server start/stop helpers."""
    calls: dict[str, Any] = {"start": [], "stop": []}

    def fake_start(root_path: Path, host: str = "127.0.0.1", timeout_seconds: float = 10.0) -> WorkspaceApiServerHandle:
        calls["start"].append((root_path, host, timeout_seconds))
        return fixture_server_handle

    def fake_stop(handle: WorkspaceApiServerHandle, timeout_seconds: float = 5.0) -> None:
        calls["stop"].append((handle, timeout_seconds))

    monkeypatch.setattr(local_module, "start_workspace_api_server", fake_start)
    monkeypatch.setattr(local_module, "stop_workspace_api_server", fake_stop)
    return calls


async def test_local_workspace_uses_session_directory(
    tmp_path: Path,
    fixture_httpx_clients: list[Any],
    fixture_server_controls: dict[str, Any],
    fixture_server_handle: WorkspaceApiServerHandle,
    fixture_reset_local_sessions: None,
) -> None:
    base_dir = tmp_path / "workspace-root"
    config = LocalWorkspaceConfig(
        working_directory=base_dir,
        initial_commands=[],
        session_id="local-session",
    )
    manager = LocalWorkspaceManager(config)

    workspace = await manager.__aenter__()

    assert fixture_server_controls["start"] == [(base_dir, "127.0.0.1", 10.0)]
    assert workspace.config.working_directory == base_dir / "local-session"
    assert workspace.config.working_directory.exists()
    assert config.working_directory == base_dir

    client = fixture_httpx_clients[0]
    assert client.base_url == "http://127.0.0.1:8123"

    await manager.__aexit__(None, None, None)
    assert fixture_server_controls["stop"] == [(fixture_server_handle, 5.0)]


async def test_local_workspace_endpoint_override_skips_server(
    tmp_path: Path,
    fixture_httpx_clients: list[Any],
    fixture_server_controls: dict[str, Any],
    fixture_reset_local_sessions: None,
) -> None:
    base_dir = tmp_path / "workspace-root"
    config = LocalWorkspaceConfig(
        working_directory=base_dir,
        initial_commands=[],
        session_id="override-session",
        endpoint="localhost:9000",
    )
    manager = LocalWorkspaceManager(config)

    workspace = await manager.__aenter__()

    assert fixture_server_controls["start"] == []
    assert workspace.config.working_directory == base_dir / "override-session"
    assert workspace.config.working_directory.exists()

    client = fixture_httpx_clients[0]
    assert client.base_url == "http://localhost:9000"

    await manager.__aexit__(None, None, None)
    assert fixture_server_controls["stop"] == []


async def test_local_workspace_reuses_session_and_cleans_up(
    tmp_path: Path,
    fixture_httpx_clients: list[Any],
    fixture_server_controls: dict[str, Any],
    fixture_reset_local_sessions: None,
) -> None:
    base_dir = tmp_path / "workspace-root"
    config = LocalWorkspaceConfig(
        working_directory=base_dir,
        initial_commands=[],
        session_id="shared-session",
    )
    manager_one = LocalWorkspaceManager(config)
    manager_two = LocalWorkspaceManager(config)

    await manager_one.__aenter__()
    await manager_two.__aenter__()

    assert len(fixture_server_controls["start"]) == 1

    await manager_one.__aexit__(None, None, None)
    assert fixture_server_controls["stop"] == []

    await manager_two.__aexit__(None, None, None)
    assert fixture_server_controls["stop"] != []
    assert fixture_httpx_clients[0].post_calls == ["/api/session/close"]
    assert fixture_httpx_clients[0].closed is True
