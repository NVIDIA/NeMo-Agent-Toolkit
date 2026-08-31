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

import pytest
from starlette.websockets import WebSocket

from nat.front_ends.fastapi.routes.websocket import _is_origin_allowed
from nat.front_ends.fastapi.routes.websocket import websocket_endpoint
from nat.runtime.session import SESSION_COOKIE_NAME


@pytest.mark.parametrize(
    "origin,allowed_origins,allow_origin_regex,expected",
    [
        # Exact match
        ("http://localhost:3000", ["http://localhost:3000"], None, True),
        # Not in list, no regex
        ("http://evil.com", ["http://localhost:3000"], None, False),
        # Wildcard accepts any non-empty origin
        ("http://anything.example.com", ["*"], None, True),
        # Wildcard with multiple entries
        ("http://foo.com", ["http://bar.com", "*"], None, True),
        # Regex match
        ("http://app.example.com", [], r"https?://[a-z]+\.example\.com", True),
        # Regex no match
        ("http://evil.com", [], r"https?://[a-z]+\.example\.com", False),
        # Regex with partial string does not match (fullmatch)
        ("http://app.example.com/extra", [], r"https?://[a-z]+\.example\.com", False),
        # None origin is always rejected
        (None, ["*"], None, False),
        (None, ["http://localhost:3000"], r".*", False),
        # Empty-string origin is always rejected (falsy-origin guard)
        ("", ["*"], None, False),
        ("", ["http://localhost:3000"], r".*", False),
        # Empty allowed list, no regex
        ("http://localhost:3000", [], None, False),
        # Regex takes precedence when list is empty
        ("http://localhost:3000", [], r"http://localhost:\d+", True),
        # Both list and regex configured; list matches first
        ("http://localhost:3000", ["http://localhost:3000"], r"http://other\.com", True),
    ],
)
def test_is_origin_allowed(origin, allowed_origins, allow_origin_regex, expected):
    assert _is_origin_allowed(origin, allowed_origins, allow_origin_regex) is expected


async def test_legacy_session_query_uses_configured_identity_policy():
    websocket = MagicMock(spec=WebSocket)
    websocket.query_params = {"session": "legacy-session"}
    websocket.scope = {"headers": []}
    websocket.headers.get.return_value = None
    websocket.accept = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.close = AsyncMock()

    worker = MagicMock()
    worker.front_end_config.accepted_identity_credentials = []
    worker.front_end_config.cors.allow_origins = []
    worker.front_end_config.cors.allow_origin_regex = None
    session_manager = MagicMock()

    endpoint = websocket_endpoint(worker=worker, session_manager=session_manager, jwt_validators={})
    await endpoint(websocket)

    assert (b"cookie", f"{SESSION_COOKIE_NAME}=legacy-session".encode()) in websocket.scope["headers"]
    websocket.send_json.assert_awaited_once()
    websocket.close.assert_awaited_once_with(code=1008, reason="Identity credential was rejected")
    worker.get_conversation_handler.assert_not_called()
