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
"""Workspace API server for session-scoped operations."""

import argparse
import asyncio
import inspect
import io
import multiprocessing
import multiprocessing.process
import os
import socket
import tarfile
import threading
import time
import typing
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from nat.cli.type_registry import GlobalTypeRegistry
from nat.data_models.skill import Skill
from nat.data_models.workspace import ActionRequest
from nat.data_models.workspace import ActionResult
from nat.data_models.workspace import ActionStatus
from nat.workspace.types import ActionContext
from nat.workspace.types import WorkspaceAction
from nat.workspace.types import WorkspaceActionSchema

SESSION_HEADER = "X-Workspace-Session"


@dataclass
class WorkspaceApiServerHandle:
    """Handle for a running workspace API server process."""

    process: multiprocessing.process.BaseProcess
    host: str
    port: int
    root_path: Path

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class SkillCreateRequest(BaseModel):
    """Request payload for creating a skill (portable dict format)."""

    name: str = Field(description="Skill name.")
    description: str = Field(description="Skill description.")
    license: str | None = Field(default=None, description="License identifier.")
    compatibility: str | None = Field(default=None, description="Compatibility info.")
    allowed_tools: list[str] = Field(default_factory=list, description="Allowed tools.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Metadata key-value pairs.")
    content: str = Field(default="", description="Skill body / instructions.")
    resources: dict[str, str] = Field(default_factory=dict, description="Base64-encoded resource files.")


@dataclass
class SessionState:
    """In-memory state for a workspace session."""

    root: Path
    skills: dict[str, Skill] = field(default_factory=dict)


class SessionStore:
    """Thread-safe store of session state and filesystem roots."""

    def __init__(self, base_root: Path):
        self._base_root = base_root
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session_root = (self._base_root / session_id).resolve()
                session_root.mkdir(parents=True, exist_ok=True)
                session = SessionState(root=session_root)
                self._sessions[session_id] = session
            return session


class ActionInstanceStore:
    """Manages per-session action instances."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._instances: dict[str, dict[str, WorkspaceAction]] = {}

    async def get_action(self, session_id: str, action_cls: type[WorkspaceAction],
                         context: ActionContext) -> WorkspaceAction:
        async with self._lock:
            session_actions = self._instances.setdefault(session_id, {})
            action = session_actions.get(action_cls.name)
            if action is None:
                action = action_cls()
                action.set_context(context)
                session_actions[action_cls.name] = action
                created = True
            else:
                action.set_context(context)
                created = False

        if created:
            await action.__aenter__()
        return action

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            actions = self._instances.pop(session_id, {})
        for action in actions.values():
            await action.__aexit__(None, None, None)


def _require_session_id(request: Request) -> str:
    session_id = request.headers.get(SESSION_HEADER)
    if not session_id:
        raise HTTPException(status_code=400, detail=f"Missing {SESSION_HEADER} header.")
    return session_id


def _resolve_path(root: Path, relative_path: str | None) -> Path:
    rel = Path(relative_path or ".")
    if rel.is_absolute():
        rel = Path(str(rel).lstrip("/"))
    resolved = (root / rel).resolve()
    if root != resolved and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="Invalid path outside workspace root.")
    return resolved


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        member_path = (root / member.name).resolve()
        if root != member_path and root not in member_path.parents:
            raise HTTPException(status_code=400, detail=f"Archive member escapes destination: {member.name}")
    archive.extractall(path=destination)


def _action_schema_payload(schema: WorkspaceActionSchema) -> dict[str, object]:
    return {
        "name": schema.name,
        "description": schema.description,
        "parameters": [{
            "type": param.type, "description": param.description
        } for param in schema.parameters],
        "result": {
            "type": schema.result.type,
            "description": schema.result.description,
        },
    }


def _action_request_from_payload(action_name: str, payload: object) -> ActionRequest:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Action payload must be a JSON object.")
    payload = typing.cast(dict[str, typing.Any], payload)

    payload_keys = {"action_name", "action_id", "arguments", "timestamp", "model_metadata"}
    if payload_keys.intersection(payload.keys()):
        try:
            action_request = ActionRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail="Invalid action request payload.") from exc
        if action_request.action_name != action_name:
            raise HTTPException(status_code=400, detail="Action name does not match the route.")
        return action_request

    args = payload.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise HTTPException(status_code=400, detail="Action args must be an object.")

    return ActionRequest(
        action_id=uuid.uuid4(),
        action_name=action_name,
        arguments=args,
        timestamp=datetime.now(tz=UTC),
    )


def create_workspace_app(root_path: Path) -> FastAPI:
    """Create the workspace API app rooted at the provided path."""
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins

    discover_and_register_plugins(PluginTypes.WORKSPACE_ACTION)

    app = FastAPI()
    root_path = root_path.resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    store = SessionStore(root_path)
    action_store = ActionInstanceStore()

    @app.get("/api/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/actions")
    async def list_actions(request: Request) -> JSONResponse:
        payload = [
            _action_schema_payload(action.action_schema)
            for action in GlobalTypeRegistry.get().get_registered_workspace_actions()
        ]
        return JSONResponse({"actions": payload})

    @app.get("/api/skills")
    async def list_skills(request: Request) -> JSONResponse:
        session = store.get_session(_require_session_id(request))
        payload = [{"name": s.name, "description": s.description} for s in session.skills.values()]
        return JSONResponse({"skills": payload})

    @app.get("/api/skills/{skill_name}")
    async def get_skill(skill_name: str, request: Request) -> JSONResponse:
        session = store.get_session(_require_session_id(request))
        skill = session.skills.get(skill_name)
        if skill is None:
            raise HTTPException(status_code=404, detail="Skill not found.")
        return JSONResponse(skill.to_portable_dict())

    @app.post("/api/skills/new")
    async def create_skill(request: Request, payload: SkillCreateRequest) -> JSONResponse:
        session = store.get_session(_require_session_id(request))
        if payload.name in session.skills:
            raise HTTPException(status_code=409, detail="Skill already exists.")
        try:
            skill = Skill.from_portable_dict(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid skill data: {exc}") from exc
        session.skills[skill.name] = skill
        return JSONResponse({"status": "created", "name": skill.name})

    @app.post("/api/actions/{action_name}/execute")
    async def execute_action(action_name: str, request: Request) -> JSONResponse:
        session_id = _require_session_id(request)
        session = store.get_session(session_id)
        payload = await request.json()
        action_request = _action_request_from_payload(action_name, payload)

        try:
            action_info = GlobalTypeRegistry.get().get_workspace_action(action_request.action_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Action not found.") from exc
        action_cls = typing.cast(type[WorkspaceAction], action_info.action_cls)

        context = ActionContext(session_id=session_id, root_path=session.root)
        action = await action_store.get_action(session_id, action_cls, context)
        start_time = time.monotonic()
        try:
            output = action.execute(context=context, args=action_request.arguments)
            if inspect.isawaitable(output):
                output = await output
            status = ActionStatus.SUCCESS
            error_message = None
        except Exception as exc:  # noqa: BLE001
            status = ActionStatus.FAILURE
            error_message = f"{type(exc).__name__}: {exc}"
            output = None

        result = ActionResult(
            action_id=action_request.action_id,
            status=status,
            output=output,
            error_message=error_message,
            execution_time=time.monotonic() - start_time,
        )
        return JSONResponse(result.model_dump(mode="json", exclude_none=True))

    @app.post("/api/session/close")
    async def close_session(request: Request) -> JSONResponse:
        session_id = _require_session_id(request)
        await action_store.close_session(session_id)
        return JSONResponse({"status": "closed"})

    @app.post("/api/file/upload")
    async def upload_file(
            request: Request,
            type: str,  # noqa: A002
            destination: str | None = None) -> JSONResponse:
        session = store.get_session(_require_session_id(request))
        session_root = session.root
        if destination is None or destination == "":
            if type == "folder":
                destination = "."
            else:
                destination = f"upload-{uuid.uuid4().hex}"
        dest = _resolve_path(session_root, destination)
        content = await request.body()

        if type == "file":
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            return JSONResponse({"status": "uploaded", "path": str(dest.relative_to(session_root))})
        if type == "folder":
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
                _safe_extract(archive, dest)
            return JSONResponse({"status": "uploaded", "path": str(dest.relative_to(session_root))})

        raise HTTPException(status_code=400, detail="Invalid upload type.")

    @app.get("/api/file/download", response_model=None)
    async def download_file(request: Request, type: str, path: str) -> Response:  # noqa: A002
        session = store.get_session(_require_session_id(request))
        session_root = session.root
        target = _resolve_path(session_root, path)
        if not target.exists():
            raise HTTPException(status_code=404, detail="Path not found.")

        if type == "file":
            return FileResponse(target)
        if type == "folder":
            archive_buffer = io.BytesIO()
            with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
                archive.add(target, arcname=".")
            archive_buffer.seek(0)
            return StreamingResponse(archive_buffer, media_type="application/gzip")

        raise HTTPException(status_code=400, detail="Invalid download type.")

    @app.get("/api/file/list")
    async def list_file(request: Request, type: str, path: str | None = None) -> JSONResponse:  # noqa: A002
        session = store.get_session(_require_session_id(request))
        session_root = session.root
        target = _resolve_path(session_root, path)
        if not target.exists():
            raise HTTPException(status_code=404, detail="Path not found.")

        if type == "file":
            content = target.read_text(encoding="utf-8", errors="replace")
            return JSONResponse({"type": "file", "content": content})
        if type == "folder":
            entries = [{
                "name": entry.name, "type": "folder" if entry.is_dir() else "file"
            } for entry in sorted(target.iterdir(), key=lambda item: item.name)]
            return JSONResponse({"type": "folder", "entries": entries})

        raise HTTPException(status_code=400, detail="Invalid list type.")

    @app.post("/api/file/delete")
    async def delete_file(request: Request, type: str, path: str) -> JSONResponse:  # noqa: A002
        session = store.get_session(_require_session_id(request))
        session_root = session.root
        target = _resolve_path(session_root, path)
        if not target.exists():
            raise HTTPException(status_code=404, detail="Path not found.")

        if type == "file":
            target.unlink()
            return JSONResponse({"status": "deleted"})
        if type == "folder":
            for child in sorted(target.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            target.rmdir()
            return JSONResponse({"status": "deleted"})

        raise HTTPException(status_code=400, detail="Invalid delete type.")

    return app


def _resolve_root_path(root_path: str | None) -> Path:
    if root_path:
        return Path(root_path)
    env_root = os.environ.get("NAT_WORKSPACE_ROOT")
    if env_root:
        return Path(env_root)
    return Path.cwd()


def run_workspace_api_server(root_path: str | None, host: str, port: int) -> None:
    """Run the workspace API server with the given root."""
    import uvicorn

    app = create_workspace_app(_resolve_root_path(root_path))
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _wait_for_server(host: str, port: int, timeout_seconds: float) -> None:
    url = f"http://{host}:{port}/api/health"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return
        except httpx.RequestError:
            time.sleep(0.1)
    raise RuntimeError("Workspace API server failed to start.")


def _allocate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_workspace_api_server(root_path: Path,
                               host: str = "127.0.0.1",
                               timeout_seconds: float = 10.0) -> WorkspaceApiServerHandle:
    """Start the workspace API server in a background process."""
    port = _allocate_port()
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=run_workspace_api_server, args=(str(root_path), host, port), daemon=True)
    process.start()
    _wait_for_server(host=host, port=port, timeout_seconds=timeout_seconds)
    return WorkspaceApiServerHandle(process=process, host=host, port=port, root_path=root_path)


def stop_workspace_api_server(handle: WorkspaceApiServerHandle, timeout_seconds: float = 5.0) -> None:
    """Stop a running workspace API server process."""
    if handle.process.is_alive():
        handle.process.terminate()
        handle.process.join(timeout=timeout_seconds)
        if handle.process.is_alive():
            handle.process.kill()


def main() -> None:
    """Run the workspace API server from the command line."""
    parser = argparse.ArgumentParser(description="Workspace API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--root", default=None, help="Workspace root path.")
    args = parser.parse_args()
    run_workspace_api_server(root_path=args.root, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
