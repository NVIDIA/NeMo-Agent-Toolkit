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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from nat.data_models.api_server import OAuthMode
from nat.data_models.api_server import OAuthModePreferencePayload
from nat.data_models.api_server import WebSocketAuthMessage
from nat.data_models.api_server import WebSocketMessageType
from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler
from nat.front_ends.fastapi.message_handler import WebSocketMessageHandler


def _make_message_handler() -> tuple[WebSocketMessageHandler, AsyncMock, WebSocketAuthenticationFlowHandler]:
    """Build a WebSocketMessageHandler with a mockable socket and a real flow handler."""
    socket = AsyncMock()
    session_manager = MagicMock()
    session_manager.get_workflow_single_output_schema.return_value = None
    session_manager.get_workflow_streaming_output_schema.return_value = None
    handler = WebSocketMessageHandler(
        socket=socket,
        session_manager=session_manager,
        step_adaptor=MagicMock(),
        worker=MagicMock(),
    )
    flow_handler = WebSocketAuthenticationFlowHandler(
        add_flow_cb=AsyncMock(),
        remove_flow_cb=AsyncMock(),
        web_socket_message_handler=handler,
    )
    handler.set_flow_handler(flow_handler)
    return handler, socket, flow_handler


async def test_process_auth_message_sets_oauth_mode_and_sends_no_response():
    """An oauth_mode_preference payload updates the flow handler's mode and emits no auth response."""
    handler, socket, flow_handler = _make_message_handler()
    msg = WebSocketAuthMessage(
        type=WebSocketMessageType.AUTH_MESSAGE,
        payload=OAuthModePreferencePayload(method="oauth_mode_preference", mode="popup"),
    )

    await handler._process_auth_message(msg)

    assert flow_handler._oauth_mode is OAuthMode.POPUP
    socket.send_json.assert_not_called()


async def test_process_auth_message_oauth_mode_preference_no_flow_handler():
    """An oauth_mode_preference payload is a no-op (no response) when no flow handler is set."""
    handler, socket, _ = _make_message_handler()
    handler.set_flow_handler(None)  # type: ignore[arg-type]
    msg = WebSocketAuthMessage(
        type=WebSocketMessageType.AUTH_MESSAGE,
        payload=OAuthModePreferencePayload(method="oauth_mode_preference", mode="redirect"),
    )

    await handler._process_auth_message(msg)

    socket.send_json.assert_not_called()
