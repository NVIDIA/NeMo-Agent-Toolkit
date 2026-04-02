# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""WebSocket route registration."""

import logging
import re
from typing import Any

from fastapi import FastAPI
from starlette.websockets import WebSocket

from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler
from nat.front_ends.fastapi.message_handler import WebSocketMessageHandler
from nat.runtime.session import SESSION_COOKIE_NAME
from nat.runtime.session import SessionManager
from nat.runtime.user_manager import UserManager

logger = logging.getLogger(__name__)

# Only allow URL-safe characters in session IDs (alphanumeric, hyphen, underscore, period, tilde).
_SAFE_SESSION_ID_RE = re.compile(r'^[A-Za-z0-9\-_.~]+$')


def _is_origin_allowed(origin: str | None, allowed_origins: list[str], allow_origin_regex: str | None) -> bool:
    """Return True if *origin* should be treated as an allowed CORS origin.

    Mirrors the three-tier check used by Starlette's CORSMiddleware:
    1. Wildcard ``"*"`` in *allowed_origins* accepts any non-empty origin.
    2. Exact membership in *allowed_origins*.
    3. Full-string match against *allow_origin_regex* (when set).
    """
    if not origin:
        return False
    if "*" in allowed_origins:
        return True
    if origin in allowed_origins:
        return True
    if allow_origin_regex and re.fullmatch(allow_origin_regex, origin):
        return True
    return False


def websocket_endpoint(*, worker: Any, session_manager: SessionManager):
    """Build websocket endpoint handler with auth-flow integration."""

    async def _websocket_endpoint(websocket: WebSocket):
        session_id = websocket.query_params.get("session")
        if session_id and not _SAFE_SESSION_ID_RE.match(session_id):
            logger.warning("WebSocket: Rejected session ID with unsafe characters")
            await websocket.close(code=1008, reason="Invalid session ID")
            return

        if session_id:
            headers = list(websocket.scope.get("headers", []))
            cookie_header = f"{SESSION_COOKIE_NAME}={session_id}"

            cookie_exists = False
            existing_session_cookie = False

            for i, (name, value) in enumerate(headers):
                if name != b"cookie":
                    continue

                cookie_exists = True
                cookie_str = value.decode()

                if f"{SESSION_COOKIE_NAME}=" in cookie_str:
                    existing_session_cookie = True
                    logger.info("WebSocket: Session cookie already present in headers (same-origin)")
                else:
                    headers[i] = (name, f"{cookie_str}; {cookie_header}".encode())
                    logger.info("WebSocket: Added session cookie to existing cookie header: %s",
                                session_id[:10] + "...")
                break

            if not cookie_exists and not existing_session_cookie:
                headers.append((b"cookie", cookie_header.encode()))
                logger.info("WebSocket: Added new session cookie header: %s", session_id[:10] + "...")

            websocket.scope["headers"] = headers

        async with WebSocketMessageHandler(websocket, session_manager, worker.get_step_adaptor(), worker) as handler:
            origin = websocket.headers.get("origin")
            allowed_origins = worker.front_end_config.cors.allow_origins or []
            allow_origin_regex = worker.front_end_config.cors.allow_origin_regex
            return_url = origin if _is_origin_allowed(origin, allowed_origins, allow_origin_regex) else None
            nat_session_id = UserManager._get_session_cookie(websocket)
            flow_handler = WebSocketAuthenticationFlowHandler(worker._add_flow,
                                                              worker._remove_flow,
                                                              handler,
                                                              return_url=return_url,
                                                              token_store=worker._oauth_token_store,
                                                              session_id=nat_session_id)
            handler.set_flow_handler(flow_handler)
            skip_eager_auth = websocket.query_params.get("skip_eager_auth") == "true"
            if not skip_eager_auth:
                try:
                    await flow_handler.run_eager_auth(worker._config.authentication)
                except Exception as e:
                    logger.info("Pre-authentication did not complete: %s", e)
            await handler.run()

    return _websocket_endpoint


async def add_websocket_routes(
    worker: Any,
    app: FastAPI,
    endpoint: Any,
    session_manager: SessionManager,
):
    """Add websocket route for an endpoint."""
    if endpoint.websocket_path:
        app.add_websocket_route(endpoint.websocket_path,
                                websocket_endpoint(
                                    worker=worker,
                                    session_manager=session_manager,
                                ))
