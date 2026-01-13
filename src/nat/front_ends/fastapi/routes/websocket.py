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
#
# WebSocket endpoint wiring.
from __future__ import annotations

import logging
import typing
import uuid
from typing import TYPE_CHECKING

from fastapi import FastAPI
from starlette.websockets import WebSocket

from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler
from nat.front_ends.fastapi.message_handler import WebSocketMessageHandler
from nat.runtime.session import SessionManager

if TYPE_CHECKING:
    from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

logger = logging.getLogger(__name__)


async def register_websocket_route(worker: FastApiFrontEndPluginWorker,
                                   app: FastAPI,
                                   endpoint: FastApiFrontEndConfig.EndpointBase,
                                   session_manager: SessionManager) -> None:
    """Register websocket endpoint if configured on the endpoint."""
    if not getattr(endpoint, "websocket_path", None):
        return
    path = typing.cast(str, endpoint.websocket_path)

    async def websocket_endpoint(websocket: WebSocket):
        # Universal cookie handling: works for both cross-origin and same-origin connections
        session_id = websocket.query_params.get("session") or str(uuid.uuid4())
        headers = list(websocket.scope.get("headers", []))
        cookie_header = f"nat-session={session_id}"

        # Check if the session cookie already exists to avoid duplicates
        cookie_exists = False
        existing_session_cookie = False

        for i, (name, value) in enumerate(headers):
            if name == b"cookie":
                cookie_exists = True
                cookie_str = value.decode()

                # Check if nat-session already exists in cookies
                if "nat-session=" in cookie_str:
                    existing_session_cookie = True
                    logger.info("WebSocket: Session cookie already present in headers (same-origin)")
                else:
                    # Append to existing cookie header (cross-origin case)
                    headers[i] = (name, f"{cookie_str}; {cookie_header}".encode())
                    logger.info("WebSocket: Added session cookie to existing cookie header: %s",
                                session_id[:10] + "...")
                break

        # Add new cookie header only if no cookies exist and no session cookie found
        if not cookie_exists and not existing_session_cookie:
            headers.append((b"cookie", cookie_header.encode()))
            logger.info("WebSocket: Added new session cookie header: %s", session_id[:10] + "...")

        # Update the websocket scope with the modified headers
        websocket.scope["headers"] = headers

        async with WebSocketMessageHandler(websocket, session_manager, worker.get_step_adaptor()) as handler:
            flow_handler = WebSocketAuthenticationFlowHandler(worker._add_flow, worker._remove_flow, handler)
            handler.set_flow_handler(flow_handler)
            await handler.run()

    worker._register_websocket_route(app, path=path, endpoint=websocket_endpoint)


__all__ = ["register_websocket_route"]
