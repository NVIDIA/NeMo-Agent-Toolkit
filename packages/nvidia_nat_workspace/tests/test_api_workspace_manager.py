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

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import Field

import nat.plugins.workspace.client as api_client_module
from nat.data_models.skill import Skill
from nat.data_models.workspace import ActionRequest
from nat.data_models.workspace import ActionResult
from nat.data_models.workspace import ActionStatus
from nat.data_models.workspace import WorkspaceBaseConfig
from nat.guardrails.workspace import WorkspaceGuardrail
from nat.guardrails.workspace import WorkspaceGuardrailViolation
from nat.plugins.workspace.client import ApiWorkspace
from nat.plugins.workspace.client import ApiWorkspaceManagerBase
from nat.plugins.workspace.client import ApiWorkspaceSessionEntry
from nat.plugins.workspace.client import WorkspaceApiClient
from nat.plugins.workspace.server import WorkspaceApiServerHandle
from nat.plugins.workspace.server import start_workspace_api_server
from nat.plugins.workspace.server import stop_workspace_api_server
from nat.workspace.types import SkillSummary


class TestWorkspaceConfig(WorkspaceBaseConfig, name="test"):
    """Configuration for API workspace manager tests."""

    __test__ = False
    http_timeout_seconds: float = Field(default=12.0, description="HTTP timeout for workspace API requests.")


@dataclass
class TestBackend:
    """Backend metadata for API workspace manager tests."""

    __test__ = False
    session_id: str


class TestWorkspace(ApiWorkspace):
    """Workspace implementation for API workspace manager tests."""

    __test__ = False

    def __init__(self,
                 config: TestWorkspaceConfig,
                 api_client: WorkspaceApiClient,
                 session_id: str | None = None,
                 backend: object | None = None):
        super().__init__(config=config, api_client=api_client, session_id=session_id)
        self.backend = backend
        self.config = config


class TestWorkspaceManager(ApiWorkspaceManagerBase[TestWorkspaceConfig, TestWorkspace, TestBackend]):
    """Concrete API workspace manager implementation for tests."""

    __test__ = False
    _sessions: dict[str,
                    ApiWorkspaceSessionEntry[TestBackend]] = {}  # pyright: ignore[reportIncompatibleVariableOverride]
    _session_lock: threading.Lock = threading.Lock()  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self,
                 config: TestWorkspaceConfig,
                 *,
                 base_url: str = "http://backend",
                 extra_header: str | None = None):
        super().__init__(config=config)
        self.base_url = base_url
        self.extra_header = extra_header
        self.start_calls: list[str] = []
        self.shutdown_calls: list[TestBackend | None] = []

    def _build_workspace(self, entry: ApiWorkspaceSessionEntry[TestBackend]) -> TestWorkspace:
        return TestWorkspace(
            config=self.config,
            api_client=entry.api_client,
            session_id=entry.session_id,
            backend=entry.backend,
        )

    def _session_headers(self, session_id: str) -> dict[str, str]:
        headers = super()._session_headers(session_id)
        if self.extra_header is not None:
            headers["X-Test-Header"] = self.extra_header
        return headers

    async def _start_backend(self, session_id: str) -> tuple[str, TestBackend]:
        self.start_calls.append(session_id)
        return self.base_url, TestBackend(session_id=session_id)

    async def _shutdown_backend(self, backend: TestBackend | None) -> None:
        self.shutdown_calls.append(backend)


@dataclass
class E2EBackend:
    """Backend metadata for end-to-end API workspace tests."""

    handle: WorkspaceApiServerHandle


class E2EWorkspaceManager(ApiWorkspaceManagerBase[TestWorkspaceConfig, TestWorkspace, E2EBackend]):
    """API workspace manager that starts a real workspace server."""

    _sessions: dict[str,
                    ApiWorkspaceSessionEntry[E2EBackend]] = {}  # pyright: ignore[reportIncompatibleVariableOverride]
    _session_lock: threading.Lock = threading.Lock()  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, config: TestWorkspaceConfig, *, base_root: Path | None = None):
        super().__init__(config=config)
        self.base_root = base_root or config.working_directory

    def _build_workspace(self, entry: ApiWorkspaceSessionEntry[E2EBackend]) -> TestWorkspace:
        return TestWorkspace(
            config=self.config,
            api_client=entry.api_client,
            session_id=entry.session_id,
            backend=entry.backend,
        )

    async def _start_backend(self, session_id: str) -> tuple[str, E2EBackend]:
        handle = start_workspace_api_server(self.base_root)
        return handle.base_url, E2EBackend(handle=handle)

    async def _shutdown_backend(self, backend: E2EBackend | None) -> None:
        if backend is None:
            return
        stop_workspace_api_server(backend.handle)


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
            self.raise_on_post: Exception | None = None
            clients.append(self)

        async def post(self, path: str) -> None:
            if self.raise_on_post is not None:
                raise self.raise_on_post
            self.post_calls.append(path)

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(api_client_module.httpx, "AsyncClient", FakeAsyncClient)
    return clients


@pytest.fixture(name="fixture_reset_manager_sessions")
def fixture_reset_manager_sessions() -> None:
    """Reset session cache for manager tests."""
    TestWorkspaceManager._sessions = {}


@pytest.fixture(name="fixture_reset_e2e_sessions")
def fixture_reset_e2e_sessions() -> None:
    """Reset session cache for end-to-end manager tests."""
    E2EWorkspaceManager._sessions = {}


@pytest.fixture(name="fixture_test_config")
def fixture_test_config(tmp_path: Path) -> TestWorkspaceConfig:
    """Provide a base workspace config for manager tests."""
    return TestWorkspaceConfig(
        working_directory=tmp_path / "workspace-root",
        initial_commands=[],
        session_id="session-123",
    )


async def test_manager_reuses_session_and_cleans_up(
    fixture_test_config: TestWorkspaceConfig,
    fixture_httpx_clients: list[Any],
    fixture_reset_manager_sessions: None,
) -> None:
    manager_one = TestWorkspaceManager(fixture_test_config)
    manager_two = TestWorkspaceManager(fixture_test_config)

    workspace_one = await manager_one.__aenter__()
    workspace_two = await manager_two.__aenter__()

    assert manager_one.start_calls == ["session-123"]
    assert manager_two.start_calls == []
    assert workspace_one.backend is workspace_two.backend
    assert workspace_one.session_id == "session-123"
    assert workspace_one.backend is not None

    assert len(fixture_httpx_clients) == 1
    client = fixture_httpx_clients[0]
    assert client.base_url == "http://backend"
    assert client.headers == {"X-Workspace-Session": "session-123"}
    assert isinstance(client.timeout, httpx.Timeout)
    assert client.timeout.connect == pytest.approx(12.0)
    assert client.timeout.read == pytest.approx(12.0)
    assert client.timeout.write == pytest.approx(12.0)
    assert client.timeout.pool == pytest.approx(12.0)

    await manager_one.__aexit__(None, None, None)
    assert client.post_calls == []
    assert manager_one.shutdown_calls == []
    assert manager_two.shutdown_calls == []

    await manager_two.__aexit__(None, None, None)
    assert client.post_calls == ["/api/session/close"]
    assert client.closed is True
    assert manager_two.shutdown_calls == [workspace_one.backend]


async def test_manager_uses_endpoint_override_and_custom_headers(
    fixture_test_config: TestWorkspaceConfig,
    fixture_httpx_clients: list[Any],
    fixture_reset_manager_sessions: None,
) -> None:
    config = fixture_test_config.model_copy(update={"endpoint": "localhost:4321"})
    manager = TestWorkspaceManager(config, extra_header="value")

    workspace = await manager.__aenter__()

    assert manager.start_calls == []
    assert workspace.backend is None
    assert len(fixture_httpx_clients) == 1
    client = fixture_httpx_clients[0]
    assert client.base_url == "http://localhost:4321"
    assert client.headers == {"X-Workspace-Session": "session-123", "X-Test-Header": "value"}

    await manager.__aexit__(None, None, None)
    assert manager.shutdown_calls == [None]


async def test_manager_ignores_close_errors(
    fixture_test_config: TestWorkspaceConfig,
    fixture_httpx_clients: list[Any],
    fixture_reset_manager_sessions: None,
) -> None:
    manager = TestWorkspaceManager(fixture_test_config)
    await manager.__aenter__()

    client = fixture_httpx_clients[0]
    client.raise_on_post = httpx.RequestError("boom", request=None)

    await manager.__aexit__(None, None, None)
    assert client.closed is True
    assert manager.shutdown_calls == [TestBackend(session_id="session-123")]


async def test_api_workspace_manager_e2e_single_session(
    tmp_path: Path,
    fixture_reset_e2e_sessions: None,
) -> None:
    base_root = tmp_path / "workspace-root"
    config = TestWorkspaceConfig(
        working_directory=base_root,
        initial_commands=[],
        session_id="e2e-single",
    )
    manager = E2EWorkspaceManager(config)

    workspace = await manager.__aenter__()
    backend = workspace.backend
    assert isinstance(backend, E2EBackend)
    assert backend.handle.process.is_alive()

    actions = await workspace.get_actions()
    assert any(action.name == "bash" for action in actions)

    action_result = await workspace.execute_action("bash", {"command": "echo hello"})
    assert isinstance(action_result, ActionResult)
    assert action_result.status is ActionStatus.SUCCESS
    assert action_result.error_message is None
    assert action_result.execution_time is not None
    assert isinstance(action_result.output, dict)
    assert action_result.output["stdout"].strip() == "hello"

    create_result = await workspace.create_skill(Skill(name="alpha", description="Skill for API workspace tests."))
    assert isinstance(create_result, ActionResult)
    assert create_result.status is ActionStatus.SUCCESS
    skills = await workspace.get_skills()
    assert all(isinstance(s, SkillSummary) for s in skills)
    assert any(skill.name == "alpha" for skill in skills)
    full_skill = await workspace.read_skill("alpha")
    assert full_skill is not None
    assert full_skill.description == "Skill for API workspace tests."

    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    (upload_dir / "note.txt").write_text("hello", encoding="utf-8")
    upload_result = await workspace.upload_directory(upload_dir, Path("project"))
    assert isinstance(upload_result, ActionResult)
    assert upload_result.status is ActionStatus.SUCCESS

    download_dir = tmp_path / "downloaded"
    download_result = await workspace.download_directory(Path("project"), download_dir)
    assert isinstance(download_result, ActionResult)
    assert download_result.status is ActionStatus.SUCCESS
    assert (download_dir / "note.txt").read_text(encoding="utf-8") == "hello"

    local_file = tmp_path / "local.txt"
    local_file.write_text("payload", encoding="utf-8")
    upload_file_result = await workspace.upload_file(local_file, Path("file.txt"))
    assert isinstance(upload_file_result, ActionResult)
    assert upload_file_result.status is ActionStatus.SUCCESS

    list_response = await workspace._api_client._http_client.get(
        "/api/file/list",
        params={
            "type": "folder", "path": "."
        },
    )
    list_response.raise_for_status()
    list_payload = list_response.json()
    entries = {entry["name"]: entry["type"] for entry in list_payload["entries"]}
    assert entries["project"] == "folder"
    assert entries["file.txt"] == "file"

    delete_dir_result = await workspace.delete_directory(Path("project"))
    assert isinstance(delete_dir_result, ActionResult)
    assert delete_dir_result.status is ActionStatus.SUCCESS
    delete_file_result = await workspace.delete_file(Path("file.txt"))
    assert isinstance(delete_file_result, ActionResult)
    assert delete_file_result.status is ActionStatus.SUCCESS

    await manager.__aexit__(None, None, None)
    assert backend.handle.process.is_alive() is False


async def test_api_workspace_manager_e2e_multiple_sessions(
    tmp_path: Path,
    fixture_reset_e2e_sessions: None,
) -> None:
    base_root = tmp_path / "workspace-root"
    server_handle = start_workspace_api_server(base_root)
    try:
        config_one = TestWorkspaceConfig(
            working_directory=base_root,
            initial_commands=[],
            session_id="session-a",
            endpoint=server_handle.base_url,
        )
        config_two = TestWorkspaceConfig(
            working_directory=base_root,
            initial_commands=[],
            session_id="session-b",
            endpoint=server_handle.base_url,
        )
        manager_one = E2EWorkspaceManager(config_one)
        manager_two = E2EWorkspaceManager(config_two)

        workspace_one = await manager_one.__aenter__()
        workspace_two = await manager_two.__aenter__()

        local_a = tmp_path / "a.txt"
        local_a.write_text("alpha", encoding="utf-8")
        upload_a = await workspace_one.upload_file(local_a, Path("shared.txt"))
        assert isinstance(upload_a, ActionResult)
        assert upload_a.status is ActionStatus.SUCCESS

        not_found = await workspace_two.delete_file(Path("shared.txt"))
        assert isinstance(not_found, ActionResult)
        assert not_found.status is ActionStatus.FAILURE
        assert not_found.error_message is not None

        local_b = tmp_path / "b.txt"
        local_b.write_text("beta", encoding="utf-8")
        upload_b = await workspace_two.upload_file(local_b, Path("shared.txt"))
        assert isinstance(upload_b, ActionResult)
        assert upload_b.status is ActionStatus.SUCCESS

        delete_a = await workspace_one.delete_file(Path("shared.txt"))
        assert isinstance(delete_a, ActionResult)
        assert delete_a.status is ActionStatus.SUCCESS
        delete_b = await workspace_two.delete_file(Path("shared.txt"))
        assert isinstance(delete_b, ActionResult)
        assert delete_b.status is ActionStatus.SUCCESS

        await manager_one.__aexit__(None, None, None)
        await manager_two.__aexit__(None, None, None)
    finally:
        stop_workspace_api_server(server_handle)
        assert server_handle.process.is_alive() is False


class _BlockActionGuardrail(WorkspaceGuardrail):
    name = "block-test"

    async def validate_action(self, action: ActionRequest) -> WorkspaceGuardrailViolation | None:
        return WorkspaceGuardrailViolation(guardrail_name=self.name, message=f"Blocked {action.action_name}")


async def test_api_workspace_guardrails_block_and_remove(
    tmp_path: Path,
    fixture_reset_e2e_sessions: None,
) -> None:
    base_root = tmp_path / "workspace-root"
    config = TestWorkspaceConfig(
        working_directory=base_root,
        initial_commands=[],
        session_id="guardrail-session",
    )
    manager = E2EWorkspaceManager(config)

    workspace = await manager.__aenter__()
    workspace.add_workspace_guardrail(_BlockActionGuardrail())

    blocked = await workspace.execute_action("bash", {"command": "echo hello"})
    assert isinstance(blocked, ActionResult)
    assert blocked.status is ActionStatus.BLOCKED_BY_GUARDRAIL
    assert blocked.error_message is not None

    removed = workspace.remove_workspace_guardrail("block-test")
    assert removed is True
    allowed = await workspace.execute_action("bash", {"command": "echo hello"})
    assert isinstance(allowed, ActionResult)
    assert allowed.status is ActionStatus.SUCCESS
    assert isinstance(allowed.output, dict)
    assert allowed.output["stdout"].strip() == "hello"

    await manager.__aexit__(None, None, None)
