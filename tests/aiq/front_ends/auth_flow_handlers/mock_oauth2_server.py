# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import base64
import hashlib
import secrets
import string
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import uvicorn
from fastapi import FastAPI
from fastapi import Form
from fastapi import Header
from fastapi import HTTPException
from fastapi import Query
from fastapi import status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from pydantic import Field


# =============================================================================
# Models
# =============================================================================
@dataclass
class _Client:
    client_id: str
    client_secret: str | None
    redirect_uri: str  # e.g. http://localhost:9000/auth/redirect


@dataclass
class _AuthCode:
    code: str
    client_id: str
    redirect_uri: str
    scope: str
    expires_at: float
    state: str | None = None
    used: bool = False
    # PKCE
    code_challenge: str | None = None
    code_challenge_method: str | None = None


@dataclass
class _DeviceCodeEntry:
    device_code: str
    user_code: str
    client_id: str
    scope: str
    expires_at: float
    interval: int
    authorized: bool = False


class _Token(BaseModel):
    access_token: str = Field(..., alias="access_token")
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: str | None = None
    scope: str = "read"


# =============================================================================
# Helper functions
# =============================================================================
def _pkce_verify(code_verifier: str, code_challenge: str, method: str) -> bool:
    if method == "plain":
        return secrets.compare_digest(code_verifier, code_challenge)
    if method == "S256":
        digest = hashlib.sha256(code_verifier.encode()).digest()
        derived = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        return secrets.compare_digest(derived, code_challenge)
    return False


def _parse_basic_auth(auth_header: str | None) -> tuple[str, str] | None:
    if not auth_header or not auth_header.startswith("Basic "):
        return None
    try:
        decoded = base64.b64decode(auth_header.split(None, 1)[1]).decode()
        cid, secret = decoded.split(":", 1)
    except Exception:
        return None
    return cid, secret


# =============================================================================
# Server
# =============================================================================
class MockOAuth2Server:

    def __init__(self, host: str = "localhost", port: int = 0) -> None:
        self._app = FastAPI(title="Mock OAuth 2 Server")
        self._host, self._port_cfg = host, port
        self._uvicorn: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

        self._clients: dict[str, _Client] = {}
        self._codes: dict[str, _AuthCode] = {}
        self._device_codes: dict[str, _DeviceCodeEntry] = {}
        self.tokens: dict[str, _Token] = {}

        self._mount_routes()

    # -------------------- public helpers ---------------------------------
    def register_client(self, *, client_id: str, client_secret: str | None, redirect_base: str) -> _Client:
        client = _Client(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=f"{redirect_base.rstrip('/')}/auth/redirect",
        )
        self._clients[client_id] = client
        return client

    def base_url(self) -> str:
        if not self._uvicorn:
            raise RuntimeError("Server not started")
        return f"http://{self._host}:{self._uvicorn.config.port}"

    def authorization_url(self) -> str:
        return f"{self.base_url()}/oauth/authorize"

    def token_url(self) -> str:
        return f"{self.base_url()}/oauth/token"

    def device_code_url(self) -> str:
        return f"{self.base_url()}/oauth/device/code"

    # -------------------- lifecycle --------------------------------------
    def start_server(self, *, threaded: bool = True, log_level: str = "error") -> None:
        cfg = uvicorn.Config(self._app, host=self._host, port=self._port_cfg, log_level=log_level)
        self._uvicorn = uvicorn.Server(cfg)

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._uvicorn.serve())

        if threaded:
            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()
            while not self._uvicorn.started:
                time.sleep(0.02)
        else:
            _run()

    def stop_server(self):
        if self._uvicorn and self._uvicorn.started:
            self._uvicorn.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)

    def __enter__(self):
        self.start_server()
        return self

    def __exit__(self, *exc):
        self.stop_server()

    # -------------------- routes -----------------------------------------
    def _mount_routes(self):
        app = self._app

        # ---- Authorization endpoint ---------------------------------
        @app.get("/oauth/authorize")
        async def authorize(
                response_type: str = Query(...),
                client_id: str = Query(...),
                redirect_uri: str = Query(...),
                scope: str = Query("read"),
                state: str | None = Query(None),
                code_challenge: str | None = Query(None),
                code_challenge_method: str | None = Query("S256"),
        ):
            if response_type != "code":
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "unsupported_response_type")

            client = self._clients.get(client_id)
            if not client or client.redirect_uri != redirect_uri:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid_client")

            code = secrets.token_urlsafe(16)
            self._codes[code] = _AuthCode(
                code=code,
                client_id=client_id,
                redirect_uri=redirect_uri,
                scope=scope,
                state=state,
                expires_at=(datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp(),
                code_challenge=code_challenge,
                code_challenge_method=code_challenge_method,
            )

            params = {"code": code}
            if state:
                params["state"] = state
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            return RedirectResponse(f"{redirect_uri}?{qs}", status_code=302)

        # ---- Device‑Code issuance -----------------------------------
        @app.post("/oauth/device/code")
        async def device_code(
                client_id: str = Form(...),
                scope: str = Form("read"),
                interval: int = Form(5),
        ):
            if client_id not in self._clients:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid_client")

            dc = secrets.token_urlsafe(24)
            user_code = "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            self._device_codes[dc] = _DeviceCodeEntry(
                device_code=dc,
                user_code=user_code,
                client_id=client_id,
                scope=scope,
                interval=interval,
                expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp(),
            )
            return {
                "device_code": dc,
                "user_code": user_code,
                "verification_uri": f"{self.base_url()}/device",
                "interval": interval,
                "expires_in": 300,
            }

        # ---- Token endpoint -----------------------------------------
        @app.post("/oauth/token")
        async def token(
                grant_type: str = Form(...),
                code: str | None = Form(None),
                redirect_uri: str | None = Form(None),
                code_verifier: str | None = Form(None),
                device_code: str | None = Form(None),
                authorization: str | None = Header(None),
                client_id_form: str | None = Form(None, alias="client_id"),
                client_secret_form: str | None = Form(None, alias="client_secret"),
        ):
            # ---- Authorization‑Code grant ---------------------------
            if grant_type == "authorization_code":
                return self._handle_auth_code_grant(
                    code,
                    redirect_uri,
                    code_verifier,
                    authorization,
                    client_id_form,
                    client_secret_form,
                )
            # ---- Device‑Code grant ----------------------------------
            if grant_type == "urn:ietf:params:oauth:grant-type:device_code":
                return self._handle_device_code_grant(client_id_form, device_code)

            raise HTTPException(status.HTTP_400_BAD_REQUEST, "unsupported_grant_type")

    # ------------------- grant handlers ----------------------------------
    def _handle_auth_code_grant(
        self,
        code: str | None,
        redirect_uri: str | None,
        code_verifier: str | None,
        auth_header: str | None,
        client_id_form: str | None,
        client_secret_form: str | None,
    ):
        # 1) locate & validate auth‑code
        if not code or code not in self._codes:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid_grant")

        auth_code = self._codes[code]
        if auth_code.used or auth_code.expires_at < time.time():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid_grant")

        if redirect_uri != auth_code.redirect_uri:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid_grant")

        # 2) determine client creds (Basic header > form > stored client)
        client_id = client_secret = None
        if creds := _parse_basic_auth(auth_header):
            client_id, client_secret = creds
        elif client_id_form:
            client_id, client_secret = client_id_form, client_secret_form
        else:
            client_id = auth_code.client_id  # public client

        client = self._clients.get(client_id or "")
        if not client:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid_client")
        if client.client_secret and client.client_secret != client_secret:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid_client")

        # 3) mark code as used and issue token
        auth_code.used = True
        return self._generate_token(scope=auth_code.scope).model_dump()

    def _handle_device_code_grant(self, client_id: str | None, device_code: str | None):
        entry = self._device_codes.get(device_code or "")
        if not entry or entry.client_id != client_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid_request")
        if entry.expires_at < time.time():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "expired_token")
        if not entry.authorized:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "authorization_pending")

        del self._device_codes[device_code]  # one‑time
        return self._generate_token(scope=entry.scope).model_dump()

    # ------------------- token factory -----------------------------------
    def _generate_token(self, *, scope: str) -> _Token:
        at = secrets.token_urlsafe(24)
        token = _Token(
            access_token=at,
            refresh_token=secrets.token_urlsafe(24),
            scope=scope,
        )
        self.tokens[at] = token
        return token
