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
"""HTTP utilities for workspace API integrations."""

import io
import shutil
import sys
import tarfile
import threading
import typing
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path

import httpx
from pydantic import ValidationError

from nat.data_models.skill import Skill
from nat.data_models.workspace import ActionRequest
from nat.data_models.workspace import ActionResult
from nat.data_models.workspace import ActionStatus
from nat.data_models.workspace import WorkspaceBaseConfig
from nat.guardrails.workspace import WorkspaceGuardrail
from nat.guardrails.workspace import WorkspaceGuardrailViolation
from nat.workspace.types import SkillSummary
from nat.workspace.types import TypeSchema
from nat.workspace.types import WorkspaceActionSchema
from nat.workspace.types import WorkspaceBase
from nat.workspace.types import WorkspaceManagerBase


def normalize_base_url(url: str) -> str:
    """Normalize a base URL string for HTTP clients."""
    trimmed = url.rstrip("/")
    if "://" not in trimmed:
        return f"http://{trimmed}"
    return trimmed


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Safely extract a tar archive to the destination directory."""
    root = destination.resolve()

    if sys.version_info >= (3, 12):
        archive.extractall(path=destination, filter="data")
    else:
        for member in archive.getmembers():
            member_path = (root / member.name).resolve()
            if root != member_path and root not in member_path.parents:
                raise ValueError(f"Archive member path escapes destination: {member.name}")
        archive.extractall(path=destination)


def _guardrail_name(guardrail: WorkspaceGuardrail) -> str:
    return getattr(guardrail, "name", guardrail.__class__.__name__)


def _format_guardrail_message(violation: WorkspaceGuardrailViolation) -> str:
    if violation.message:
        return f"{violation.guardrail_name}: {violation.message}"
    return violation.guardrail_name


def _new_action_result(
    *,
    action_id: uuid.UUID | None = None,
    status: ActionStatus,
    output: typing.Any | None,
    error_message: str | None,
    execution_time: float | None = None,
) -> ActionResult:
    return ActionResult(
        action_id=action_id or uuid.uuid4(),
        status=status,
        output=output,
        error_message=error_message,
        execution_time=execution_time,
    )


async def _run_workspace_guardrails(
    action: ActionRequest,
    workspace_guardrails: list[WorkspaceGuardrail],
) -> WorkspaceGuardrailViolation | None:
    for workspace_guardrail in workspace_guardrails:
        guardrail_name = _guardrail_name(workspace_guardrail)
        try:
            violation = await workspace_guardrail.validate_action(action)
        except Exception as exc:  # noqa: BLE001
            return WorkspaceGuardrailViolation(guardrail_name=guardrail_name, message=f"Guardrail error: {exc}")
        if violation is not None:
            return violation
    return None


async def _apply_workspace_guardrails(
    action: ActionRequest,
    result: ActionResult,
    workspace_guardrails: list[WorkspaceGuardrail],
) -> ActionResult:
    updated = result
    for workspace_guardrail in workspace_guardrails:
        guardrail_name = _guardrail_name(workspace_guardrail)
        try:
            updated = await workspace_guardrail.sanitize_result(action, updated)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Guardrail {guardrail_name} failed: {exc}",
                execution_time=updated.execution_time,
            )
    return updated


class WorkspaceApiClient:
    """HTTP client for workspace API endpoints."""

    def __init__(self, http_client: httpx.AsyncClient):
        self._http_client = http_client

    @staticmethod
    def _url(path: str) -> str:
        return f"/api/{path.lstrip('/')}"

    @staticmethod
    def _extract_list(payload: typing.Any, key: str) -> list[typing.Any]:
        if isinstance(payload, dict):
            items = payload.get(key)
        else:
            items = payload
        if not isinstance(items, list):
            raise ValueError(f"Expected list payload for '{key}'")
        return items

    @staticmethod
    def _parse_type_schema(raw: dict[str, typing.Any]) -> TypeSchema:
        return TypeSchema(type=str(raw.get("type", "")), description=str(raw.get("description", "")))

    def _parse_action_schema(self, raw: dict[str, typing.Any]) -> WorkspaceActionSchema:
        parameters = [self._parse_type_schema(param) for param in raw.get("parameters", [])]
        result = self._parse_type_schema(raw.get("result", {}))
        return WorkspaceActionSchema(
            name=str(raw.get("name", "")),
            description=str(raw.get("description", "")),
            parameters=parameters,
            result=result,
        )

    @staticmethod
    def _parse_skill_summary(raw: dict[str, typing.Any]) -> SkillSummary:
        return SkillSummary(name=str(raw.get("name", "")), description=str(raw.get("description", "")))

    async def get_actions(self) -> list[WorkspaceActionSchema]:
        """Return action schemas from the workspace API."""
        try:
            response = await self._http_client.get(self._url("actions"))
            response.raise_for_status()
            payload = response.json()
            items = self._extract_list(payload, "actions")
            return [self._parse_action_schema(item) for item in items]
        except httpx.RequestError as exc:
            raise RuntimeError("Workspace action discovery request failed.") from exc
        except (httpx.HTTPStatusError, ValueError) as exc:
            raise RuntimeError("Workspace action discovery returned invalid data.") from exc

    async def get_skills(self) -> list[SkillSummary]:
        """Return skill summaries from the workspace API."""
        try:
            response = await self._http_client.get(self._url("skills"))
            response.raise_for_status()
            payload = response.json()
            items = self._extract_list(payload, "skills")
            return [self._parse_skill_summary(item) for item in items]
        except httpx.RequestError as exc:
            raise RuntimeError("Workspace skill discovery request failed.") from exc
        except (httpx.HTTPStatusError, ValueError) as exc:
            raise RuntimeError("Workspace skill discovery returned invalid data.") from exc

    async def read_skill(self, skill_name: str) -> Skill | None:
        """Return a single skill by name from the workspace API."""
        try:
            response = await self._http_client.get(self._url(f"skills/{skill_name}"))
        except httpx.RequestError as exc:
            raise RuntimeError(f"Workspace read_skill request failed for {skill_name!r}.") from exc
        if response.status_code == 404:
            return None
        try:
            response.raise_for_status()
            return Skill.from_portable_dict(response.json())
        except (httpx.HTTPStatusError, ValueError) as exc:
            raise RuntimeError(f"Workspace read_skill returned invalid data for {skill_name!r}.") from exc

    async def create_skill(self, skill: Skill) -> ActionResult:
        """Create a skill via the workspace API."""
        action_id = uuid.uuid4()
        payload = skill.to_portable_dict()
        try:
            response = await self._http_client.post(self._url("skills/new"), json=payload)
        except httpx.RequestError as exc:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to create skill {skill.name}: {exc}",
            )

        if response.status_code in {200, 201}:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.SUCCESS,
                output={
                    "status": "created", "name": skill.name
                },
                error_message=None,
            )
        if response.status_code == 409:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Skill {skill.name} already exists",
            )
        return _new_action_result(
            action_id=action_id,
            status=ActionStatus.FAILURE,
            output=None,
            error_message=f"Failed to create skill {skill.name}: {response.status_code} {response.text}",
        )

    async def execute_action(self, action_request: ActionRequest) -> ActionResult:
        """Execute an action request via the workspace API."""
        payload = action_request.model_dump(mode="json", exclude_none=True)
        try:
            response = await self._http_client.post(
                self._url(f"actions/{action_request.action_name}/execute"),
                json=payload,
            )
        except httpx.RequestError as exc:
            return _new_action_result(
                action_id=action_request.action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Action {action_request.action_name} failed to execute: {exc}",
            )

        if response.status_code == 404:
            return _new_action_result(
                action_id=action_request.action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Action {action_request.action_name} not found",
            )
        if not response.is_success:
            return _new_action_result(
                action_id=action_request.action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=(f"Action {action_request.action_name} failed to execute: "
                               f"{response.status_code} {response.text}"),
            )
        try:
            payload = response.json()
        except ValueError:
            payload = response.text

        if isinstance(payload, dict):
            try:
                return ActionResult.model_validate(payload)
            except ValidationError:
                output = payload.get("result", payload)
        else:
            output = payload

        return _new_action_result(
            action_id=action_request.action_id,
            status=ActionStatus.SUCCESS,
            output=output,
            error_message=None,
        )

    async def execute_action_from_args(
        self,
        action_name: str,
        args: dict[str, typing.Any],
        model_metadata: dict[str, typing.Any] | None = None,
    ) -> ActionResult:
        """Execute an action via the workspace API."""
        if args is None:
            args = {}
        try:
            action_request = ActionRequest(
                action_id=uuid.uuid4(),
                action_name=action_name,
                arguments=args,
                timestamp=datetime.now(tz=UTC),
                model_metadata=model_metadata,
            )
        except ValidationError as exc:
            return _new_action_result(
                action_id=uuid.uuid4(),
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Action {action_name} failed to serialize request: {exc}",
            )
        return await self.execute_action(action_request)

    async def upload_file(self, file_path: Path, destination_path: Path) -> ActionResult:
        """Upload a file to the workspace API."""
        action_id = uuid.uuid4()
        try:
            content = file_path.read_bytes()
        except FileNotFoundError:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Local file {file_path} not found",
            )
        except Exception as exc:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to read local file {file_path}: {exc}",
            )

        try:
            response = await self._http_client.post(
                self._url("file/upload"),
                params={
                    "type": "file", "destination": str(destination_path)
                },
                content=content,
            )
        except httpx.RequestError as exc:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to upload file {file_path}: {exc}",
            )

        if response.status_code in {200, 201}:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.SUCCESS,
                output={
                    "status": "uploaded", "path": str(destination_path)
                },
                error_message=None,
            )
        if response.status_code == 409:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Workspace file {destination_path} already exists",
            )
        if response.status_code == 404:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Workspace path {destination_path} not found",
            )
        return _new_action_result(
            action_id=action_id,
            status=ActionStatus.FAILURE,
            output=None,
            error_message=(f"Failed to upload file {file_path} to {destination_path}: "
                           f"{response.status_code} {response.text}"),
        )

    async def delete_file(self, file_path: Path) -> ActionResult:
        """Delete a file via the workspace API."""
        action_id = uuid.uuid4()
        try:
            response = await self._http_client.post(
                self._url("file/delete"),
                params={
                    "type": "file", "path": str(file_path)
                },
            )
        except httpx.RequestError as exc:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to delete file {file_path}: {exc}",
            )

        if response.status_code in {200, 204}:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.SUCCESS,
                output={
                    "status": "deleted", "path": str(file_path)
                },
                error_message=None,
            )
        if response.status_code == 404:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Workspace file {file_path} not found",
            )
        return _new_action_result(
            action_id=action_id,
            status=ActionStatus.FAILURE,
            output=None,
            error_message=f"Failed to delete file {file_path}: {response.status_code} {response.text}",
        )

    async def download_file(self, file_path: Path) -> ActionResult:
        """Download a file from the workspace API."""
        action_id = uuid.uuid4()
        try:
            response = await self._http_client.get(
                self._url("file/download"),
                params={
                    "type": "file", "path": str(file_path)
                },
            )
        except httpx.RequestError as exc:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to download file {file_path}: {exc}",
            )

        if response.status_code == 404:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Workspace file {file_path} not found",
            )
        if not response.is_success:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to download file {file_path}: {response.status_code} {response.text}",
            )

        destination_path = Path(".") / file_path.name
        try:
            destination_path.write_bytes(response.content)
        except Exception as exc:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to write file {destination_path}: {exc}",
            )
        return _new_action_result(
            action_id=action_id,
            status=ActionStatus.SUCCESS,
            output={
                "status": "downloaded", "path": str(destination_path)
            },
            error_message=None,
        )

    async def upload_directory(self, directory_path: Path, destination_path: Path) -> ActionResult:
        """Upload a directory to the workspace API."""
        action_id = uuid.uuid4()
        if not directory_path.exists():
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Local directory {directory_path} not found",
            )
        if not directory_path.is_dir():
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Local path {directory_path} is not a directory",
            )

        try:
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w:gz") as tar:
                tar.add(directory_path, arcname=".")
            archive.seek(0)
        except Exception as exc:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to archive directory {directory_path}: {exc}",
            )

        try:
            response = await self._http_client.post(
                self._url("file/upload"),
                params={
                    "type": "folder", "destination": str(destination_path)
                },
                content=archive.getvalue(),
                headers={"Content-Type": "application/gzip"},
            )
        except httpx.RequestError as exc:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to upload directory {directory_path}: {exc}",
            )

        if response.status_code in {200, 201}:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.SUCCESS,
                output={
                    "status": "uploaded", "path": str(destination_path)
                },
                error_message=None,
            )
        if response.status_code == 409:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Workspace directory {destination_path} already exists",
            )
        if response.status_code == 404:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Workspace path {destination_path} not found",
            )
        return _new_action_result(
            action_id=action_id,
            status=ActionStatus.FAILURE,
            output=None,
            error_message=(f"Failed to upload directory {directory_path} to {destination_path}: "
                           f"{response.status_code} {response.text}"),
        )

    async def download_directory(self, directory_path: Path, destination_path: Path) -> ActionResult:
        """Download a directory from the workspace API."""
        action_id = uuid.uuid4()
        if destination_path.exists():
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Local directory {destination_path} already exists",
            )

        try:
            response = await self._http_client.get(
                self._url("file/download"),
                params={
                    "type": "folder", "path": str(directory_path)
                },
            )
        except httpx.RequestError as exc:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to download directory {directory_path}: {exc}",
            )

        if response.status_code == 404:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Workspace directory {directory_path} not found",
            )
        if not response.is_success:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=(f"Failed to download directory {directory_path}: "
                               f"{response.status_code} {response.text}"),
            )

        try:
            destination_path.mkdir(parents=True, exist_ok=False)
            with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
                _safe_extract(tar, destination_path)
        except Exception as exc:
            shutil.rmtree(destination_path, ignore_errors=True)
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to extract directory {directory_path}: {exc}",
            )

        return _new_action_result(
            action_id=action_id,
            status=ActionStatus.SUCCESS,
            output={
                "status": "downloaded", "path": str(destination_path)
            },
            error_message=None,
        )

    async def delete_directory(self, directory_path: Path) -> ActionResult:
        """Delete a directory via the workspace API."""
        action_id = uuid.uuid4()
        try:
            response = await self._http_client.post(
                self._url("file/delete"),
                params={
                    "type": "folder", "path": str(directory_path)
                },
            )
        except httpx.RequestError as exc:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Failed to delete directory {directory_path}: {exc}",
            )

        if response.status_code in {200, 204}:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.SUCCESS,
                output={
                    "status": "deleted", "path": str(directory_path)
                },
                error_message=None,
            )
        if response.status_code == 404:
            return _new_action_result(
                action_id=action_id,
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Workspace directory {directory_path} not found",
            )
        return _new_action_result(
            action_id=action_id,
            status=ActionStatus.FAILURE,
            output=None,
            error_message=(f"Failed to delete directory {directory_path}: "
                           f"{response.status_code} {response.text}"),
        )


class ApiWorkspace(WorkspaceBase):
    """Workspace implementation backed by a workspace API client."""

    def __init__(self, config: WorkspaceBaseConfig, api_client: WorkspaceApiClient, session_id: str | None = None):
        self.config = config
        self._api_client = api_client
        self.session_id = session_id
        self._workspace_guardrails: list[WorkspaceGuardrail] = []

    def add_workspace_guardrail(self, workspace_guardrail: WorkspaceGuardrail) -> None:
        """Add a workspace guardrail that runs before action execution."""
        guardrail_name = _guardrail_name(workspace_guardrail)
        if any(_guardrail_name(existing) == guardrail_name for existing in self._workspace_guardrails):
            raise ValueError(f"Workspace guardrail '{guardrail_name}' already exists.")
        self._workspace_guardrails.append(workspace_guardrail)

    def remove_workspace_guardrail(self, guardrail_name: str) -> bool:
        """Remove a workspace guardrail by name."""
        for index, workspace_guardrail in enumerate(self._workspace_guardrails):
            if _guardrail_name(workspace_guardrail) == guardrail_name:
                del self._workspace_guardrails[index]
                return True
        return False

    def clear_workspace_guardrails(self) -> None:
        """Remove all workspace guardrails from the workspace."""
        self._workspace_guardrails.clear()

    def list_workspace_guardrails(self) -> tuple[WorkspaceGuardrail, ...]:
        """Return the workspace guardrails currently configured on this workspace."""
        return tuple(self._workspace_guardrails)

    async def get_actions(self) -> list[WorkspaceActionSchema]:
        """Return action schemas from the workspace API."""
        return await self._api_client.get_actions()

    async def get_skills(self) -> list[SkillSummary]:
        """Return skill summaries from the workspace API."""
        return await self._api_client.get_skills()

    async def read_skill(self, skill_name: str) -> Skill | None:
        """Return a single skill by name from the workspace API."""
        return await self._api_client.read_skill(skill_name)

    async def create_skill(self, skill: Skill) -> ActionResult:
        """Create a skill using the workspace API."""
        return await self._api_client.create_skill(skill=skill)

    async def upload_file(self, file_path: Path, destination_path: Path) -> ActionResult:
        """Upload a file using the workspace API."""
        return await self._api_client.upload_file(file_path=file_path, destination_path=destination_path)

    async def delete_file(self, file_path: Path) -> ActionResult:
        """Delete a file using the workspace API."""
        return await self._api_client.delete_file(file_path=file_path)

    async def download_file(self, file_path: Path) -> ActionResult:
        """Download a file using the workspace API."""
        return await self._api_client.download_file(file_path=file_path)

    async def upload_directory(self, directory_path: Path, destination_path: Path) -> ActionResult:
        """Upload a directory using the workspace API."""
        return await self._api_client.upload_directory(directory_path=directory_path, destination_path=destination_path)

    async def download_directory(self, directory_path: Path, destination_path: Path) -> ActionResult:
        """Download a directory using the workspace API."""
        return await self._api_client.download_directory(directory_path=directory_path,
                                                         destination_path=destination_path)

    async def delete_directory(self, directory_path: Path) -> ActionResult:
        """Delete a directory using the workspace API."""
        return await self._api_client.delete_directory(directory_path=directory_path)

    async def execute_action(
        self,
        action_name: str,
        args: dict[str, typing.Any],
        model_metadata: dict[str, typing.Any] | None = None,
    ) -> ActionResult:
        """Execute an action using the workspace API."""
        if args is None:
            args = {}
        try:
            action_request = ActionRequest(
                action_id=uuid.uuid4(),
                action_name=action_name,
                arguments=args,
                timestamp=datetime.now(tz=UTC),
                model_metadata=model_metadata,
            )
        except ValidationError as exc:
            return _new_action_result(
                action_id=uuid.uuid4(),
                status=ActionStatus.FAILURE,
                output=None,
                error_message=f"Action {action_name} failed to serialize request: {exc}",
            )

        violation = await _run_workspace_guardrails(action_request, self._workspace_guardrails)
        if violation is not None:
            return ActionResult(
                action_id=action_request.action_id,
                status=ActionStatus.BLOCKED_BY_GUARDRAIL,
                output=None,
                error_message=_format_guardrail_message(violation),
                execution_time=None,
            )

        result = await self._api_client.execute_action(action_request)
        if isinstance(result, ActionResult):
            return await _apply_workspace_guardrails(action_request, result, self._workspace_guardrails)
        return result


ApiConfigT = typing.TypeVar("ApiConfigT", bound=WorkspaceBaseConfig)
ApiWorkspaceT = typing.TypeVar("ApiWorkspaceT", bound=ApiWorkspace)
BackendT = typing.TypeVar("BackendT")


@dataclass
class ApiWorkspaceSessionEntry(typing.Generic[BackendT]):
    """Session entry for API-backed workspace managers.

    Attributes:
        session_id: Unique identifier for the workspace session.
        base_url: Base URL used for the workspace API.
        http_client: HTTP client bound to the workspace API.
        api_client: Workspace API client wrapper.
        ref_count: Reference count for the shared session.
        backend: Backend-specific metadata for cleanup and workspace construction.
    """

    session_id: str
    base_url: str
    http_client: httpx.AsyncClient
    api_client: WorkspaceApiClient
    ref_count: int
    backend: BackendT | None


class ApiWorkspaceManagerBase(
        WorkspaceManagerBase[ApiConfigT, ApiWorkspaceT],
        typing.Generic[ApiConfigT, ApiWorkspaceT, BackendT],
):
    """Base manager for API-backed workspace sessions."""

    _sessions: typing.ClassVar[dict[str, ApiWorkspaceSessionEntry[typing.Any]]]
    _session_lock: typing.ClassVar[threading.Lock]

    def __init__(self, config: ApiConfigT):
        super().__init__(config=config)
        self._session_id: str | None = None

    async def __aenter__(self) -> ApiWorkspaceT:
        """Open or reuse a workspace session."""
        self._session_id = self.config.session_id or uuid.uuid4().hex
        entry = await self._get_or_create_session(self._session_id)
        return self._build_workspace(entry)

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """Release a workspace session."""
        if self._session_id is None:
            return
        await self._release_session(self._session_id)
        self._session_id = None

    @abstractmethod
    def _build_workspace(self, entry: ApiWorkspaceSessionEntry[BackendT]) -> ApiWorkspaceT:
        """Construct a workspace instance from a session entry."""
        raise NotImplementedError

    def _endpoint_override(self) -> str | None:
        return self.config.endpoint

    def _session_headers(self, session_id: str) -> dict[str, str]:
        return {"X-Workspace-Session": session_id}

    def _http_timeout_seconds(self) -> float:
        return typing.cast(float, getattr(self.config, "http_timeout_seconds", 30.0))

    @abstractmethod
    async def _start_backend(self, session_id: str) -> tuple[str, BackendT | None]:
        """Start the workspace backend and return its base URL and metadata."""
        raise NotImplementedError

    async def _shutdown_backend(self, backend: BackendT | None) -> None:
        """Stop any backend resources associated with a session."""
        return

    async def _resolve_base_url(self, session_id: str) -> tuple[str, BackendT | None]:
        endpoint = self._endpoint_override()
        if endpoint:
            return normalize_base_url(endpoint), None
        return await self._start_backend(session_id)

    async def _get_or_create_session(self, session_id: str) -> ApiWorkspaceSessionEntry[BackendT]:
        with self._session_lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.ref_count += 1
                return entry

        base_url, backend = await self._resolve_base_url(session_id)
        headers = self._session_headers(session_id)
        http_client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers or None,
            timeout=httpx.Timeout(self._http_timeout_seconds()),
        )
        api_client = WorkspaceApiClient(http_client)
        entry = ApiWorkspaceSessionEntry(
            session_id=session_id,
            base_url=base_url,
            http_client=http_client,
            api_client=api_client,
            ref_count=1,
            backend=backend,
        )

        with self._session_lock:
            self._sessions[session_id] = entry
        return entry

    async def _release_session(self, session_id: str) -> None:
        entry: ApiWorkspaceSessionEntry[BackendT] | None = None
        with self._session_lock:
            current = self._sessions.get(session_id)
            if current is None:
                return
            current.ref_count -= 1
            if current.ref_count <= 0:
                entry = self._sessions.pop(session_id)

        if entry is None:
            return

        try:
            await entry.http_client.post("/api/session/close")
        except httpx.RequestError:
            pass
        await entry.http_client.aclose()
        await self._shutdown_backend(entry.backend)
