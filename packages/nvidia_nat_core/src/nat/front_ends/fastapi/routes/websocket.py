# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WebSocket route registration."""

import logging
from typing import Any

from fastapi import FastAPI
from starlette.websockets import WebSocket

from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler
from nat.front_ends.fastapi.message_handler import WebSocketMessageHandler
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


def websocket_endpoint(*, worker: Any, session_manager: SessionManager):
    """Build websocket endpoint handler with auth-flow integration."""

    async def _websocket_endpoint(websocket: WebSocket):
        session_id = websocket.query_params.get("session")
        if session_id:
            headers = list(websocket.scope.get("headers", []))
            cookie_header = f"nat-session={session_id}"

            cookie_exists = False
            existing_session_cookie = False

            for i, (name, value) in enumerate(headers):
                if name != b"cookie":
                    continue

                cookie_exists = True
                cookie_str = value.decode()

                if "nat-session=" in cookie_str:
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
            flow_handler = WebSocketAuthenticationFlowHandler(worker._add_flow, worker._remove_flow, handler)
            handler.set_flow_handler(flow_handler)
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
