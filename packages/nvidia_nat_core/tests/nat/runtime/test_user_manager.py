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
"""Tests for UserManager — stateless credential resolver."""

import base64
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr
from starlette.requests import Request
from starlette.websockets import WebSocket

from nat.data_models.api_server import ApiKeyAuthPayload
from nat.data_models.api_server import BasicAuthPayload
from nat.data_models.api_server import JwtAuthPayload
from nat.data_models.user_info import BasicUserInfo
from nat.data_models.user_info import JwtUserInfo
from nat.data_models.user_info import UserInfo
from nat.runtime.user_manager import SESSION_COOKIE_NAME
from nat.runtime.user_manager import UserManager


def _make_jwt(claims: dict) -> str:
    """Build a minimal unsigned JWT (header.payload.signature) for testing."""
    header: str = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    payload: str = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}."


def _mock_request(cookies: dict[str, str] | None = None, headers: dict[str, str] | None = None) -> MagicMock:
    """Create a MagicMock that passes ``isinstance(obj, Request)``."""
    mock = MagicMock(spec=Request)
    mock.cookies = cookies or {}
    mock.headers = MagicMock()
    mock.headers.get = (headers or {}).get
    return mock


def _mock_websocket(cookie_header: str | None = None, auth_header: str | None = None) -> MagicMock:
    """Create a MagicMock that passes ``isinstance(obj, WebSocket)``."""
    raw_headers: list[tuple[bytes, bytes]] = []
    if cookie_header:
        raw_headers.append((b"cookie", cookie_header.encode()))
    if auth_header:
        raw_headers.append((b"authorization", auth_header.encode()))

    mock = MagicMock(spec=WebSocket)
    mock.scope = {"headers": raw_headers}
    return mock


# ---------------------------------------------------------------------------
# from_connection — session cookie via Request
# ---------------------------------------------------------------------------
class TestFromConnectionRequestCookie:

    def test_session_cookie_returns_user_info(self):
        req = _mock_request(cookies={SESSION_COOKIE_NAME: "abc123"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        assert info.get_user_id()
        assert info.get_user_details() == "abc123"

    def test_deterministic_uuid_from_cookie(self):
        req1 = _mock_request(cookies={SESSION_COOKIE_NAME: "same-cookie"})
        req2 = _mock_request(cookies={SESSION_COOKIE_NAME: "same-cookie"})

        assert UserManager.extract_user_from_connection(req1).get_user_id() == \
               UserManager.extract_user_from_connection(req2).get_user_id()

    def test_different_cookies_different_uuids(self):
        req1 = _mock_request(cookies={SESSION_COOKIE_NAME: "cookie-a"})
        req2 = _mock_request(cookies={SESSION_COOKIE_NAME: "cookie-b"})

        assert UserManager.extract_user_from_connection(req1).get_user_id() != \
               UserManager.extract_user_from_connection(req2).get_user_id()


# ---------------------------------------------------------------------------
# from_connection — JWT via Request
# ---------------------------------------------------------------------------
class TestFromConnectionRequestJwt:

    def test_jwt_returns_user_info(self):
        token: str = _make_jwt({"sub": "user-123", "email": "test@example.com"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        assert info.get_user_id()
        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.email == "test@example.com"
        assert details.subject == "user-123"

    def test_jwt_identity_claim_email_preferred(self):
        token: str = _make_jwt({"email": "a@b.com", "preferred_username": "auser", "sub": "sub-1"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.identity_claim == "a@b.com"

    def test_jwt_with_roles_and_scopes(self):
        token: str = _make_jwt({"sub": "user-1", "roles": ["admin"], "scope": "read write"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.roles == ["admin"]
        assert details.scopes == ["read", "write"]

    def test_jwt_name_split_into_first_last(self):
        token: str = _make_jwt({"sub": "user-1", "name": "Jane Doe"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.first_name == "Jane"
        assert details.last_name == "Doe"

    def test_jwt_given_family_name_preferred_over_name(self):
        token: str = _make_jwt({"sub": "user-1", "given_name": "Alice", "family_name": "Smith", "name": "Wrong Name"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()

        assert isinstance(details, JwtUserInfo)
        assert details.first_name == "Alice"
        assert details.last_name == "Smith"

    def test_jwt_keycloak_realm_access_roles(self):
        token: str = _make_jwt({"sub": "user-1", "realm_access": {"roles": ["admin", "editor"]}})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()

        assert isinstance(details, JwtUserInfo)
        assert details.roles == ["admin", "editor"]


# ---------------------------------------------------------------------------
# from_connection — session cookie via WebSocket
# ---------------------------------------------------------------------------
class TestFromConnectionWebSocketCookie:

    def test_websocket_cookie_returns_user_info(self):
        ws = _mock_websocket(cookie_header=f"{SESSION_COOKIE_NAME}=ws-session-abc")
        info: UserInfo = UserManager.extract_user_from_connection(ws)

        assert info.get_user_id()
        assert info.get_user_details() == "ws-session-abc"

    def test_websocket_cookie_with_multiple_cookies(self):
        ws = _mock_websocket(cookie_header=f"other=foo; {SESSION_COOKIE_NAME}=ws-session-xyz; bar=baz")
        info: UserInfo = UserManager.extract_user_from_connection(ws)

        assert info.get_user_details() == "ws-session-xyz"


# ---------------------------------------------------------------------------
# from_connection — JWT via WebSocket
# ---------------------------------------------------------------------------
class TestFromConnectionWebSocketJwt:

    def test_websocket_jwt_returns_user_info(self):
        token: str = _make_jwt({"sub": "ws-jwt-user", "email": "ws@example.com"})
        ws = _mock_websocket(auth_header=f"Bearer {token}")
        info: UserInfo = UserManager.extract_user_from_connection(ws)

        assert info.get_user_id()
        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.email == "ws@example.com"


# ---------------------------------------------------------------------------
# from_connection — priority: cookie > JWT
# ---------------------------------------------------------------------------
class TestFromConnectionPriority:

    def test_cookie_takes_precedence_over_jwt(self):
        token: str = _make_jwt({"sub": "jwt-user"})
        req = _mock_request(
            cookies={SESSION_COOKIE_NAME: "cookie-user"},
            headers={"authorization": f"Bearer {token}"},
        )
        info: UserInfo = UserManager.extract_user_from_connection(req)
        assert info.get_user_details() == "cookie-user"

    def test_websocket_cookie_takes_precedence_over_jwt(self):
        token: str = _make_jwt({"sub": "jwt-user"})
        ws = _mock_websocket(
            cookie_header=f"{SESSION_COOKIE_NAME}=ws-cookie-user",
            auth_header=f"Bearer {token}",
        )
        info: UserInfo = UserManager.extract_user_from_connection(ws)
        assert info.get_user_details() == "ws-cookie-user"


# ---------------------------------------------------------------------------
# from_connection — no credential returns None
# ---------------------------------------------------------------------------
class TestFromConnectionNoCredential:

    def test_no_credentials_returns_none(self):
        req = _mock_request()
        assert UserManager.extract_user_from_connection(req) is None

    def test_non_bearer_scheme_returns_none(self):
        req = _mock_request(headers={"authorization": "Basic dXNlcjpwYXNz"})
        assert UserManager.extract_user_from_connection(req) is None

    def test_invalid_jwt_returns_none(self):
        req = _mock_request(headers={"authorization": "Bearer not.valid.jwt"})
        assert UserManager.extract_user_from_connection(req) is None

    def test_jwt_without_identity_claim_raises(self):
        token: str = _make_jwt({"iss": "some-issuer"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        with pytest.raises(ValueError, match="no usable identity claim"):
            UserManager.extract_user_from_connection(req)

    def test_empty_websocket_returns_none(self):
        ws = _mock_websocket()
        assert UserManager.extract_user_from_connection(ws) is None


# ---------------------------------------------------------------------------
# _from_auth_payload — JWT
# ---------------------------------------------------------------------------
class TestFromAuthPayloadJwt:

    def test_jwt_payload_returns_user_info(self):
        token: str = _make_jwt({"sub": "payload-user", "email": "p@example.com"})
        payload = JwtAuthPayload(method="jwt", token=SecretStr(token))
        info: UserInfo | None = UserManager._from_auth_payload(payload)

        assert info is not None
        assert info.get_user_id()
        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.email == "p@example.com"
        assert details.subject == "payload-user"

    def test_jwt_payload_deterministic_uuid(self):
        token: str = _make_jwt({"sub": "stable-user", "email": "s@example.com"})
        p1 = JwtAuthPayload(method="jwt", token=SecretStr(token))
        p2 = JwtAuthPayload(method="jwt", token=SecretStr(token))

        assert UserManager._from_auth_payload(p1).get_user_id() == \
               UserManager._from_auth_payload(p2).get_user_id()

    def test_jwt_payload_invalid_token_returns_none(self):
        payload = JwtAuthPayload(method="jwt", token=SecretStr("not-a-jwt"))
        assert UserManager._from_auth_payload(payload) is None

    def test_jwt_payload_empty_token_returns_none(self):
        payload = JwtAuthPayload(method="jwt", token=SecretStr(""))
        assert UserManager._from_auth_payload(payload) is None

    def test_jwt_payload_no_identity_claim_raises(self):
        token: str = _make_jwt({"iss": "some-issuer"})
        payload = JwtAuthPayload(method="jwt", token=SecretStr(token))
        with pytest.raises(ValueError, match="no usable identity claim"):
            UserManager._from_auth_payload(payload)


# ---------------------------------------------------------------------------
# _from_auth_payload — API key
# ---------------------------------------------------------------------------
class TestFromAuthPayloadApiKey:

    def test_api_key_payload_returns_user_info(self):
        payload = ApiKeyAuthPayload(method="api_key", token=SecretStr("nvapi-abc123"))
        info: UserInfo | None = UserManager._from_auth_payload(payload)

        assert info is not None
        assert info.get_user_id()
        assert info.get_user_details() == "nvapi-abc123"

    def test_api_key_deterministic_uuid(self):
        p1 = ApiKeyAuthPayload(method="api_key", token=SecretStr("same-key"))
        p2 = ApiKeyAuthPayload(method="api_key", token=SecretStr("same-key"))

        assert UserManager._from_auth_payload(p1).get_user_id() == \
               UserManager._from_auth_payload(p2).get_user_id()

    def test_api_key_empty_token_returns_none(self):
        payload = ApiKeyAuthPayload(method="api_key", token=SecretStr(""))
        assert UserManager._from_auth_payload(payload) is None


# ---------------------------------------------------------------------------
# _from_auth_payload — Basic
# ---------------------------------------------------------------------------
class TestFromAuthPayloadBasic:

    def test_basic_payload_returns_user_info(self):
        payload = BasicAuthPayload(method="basic", username="alice", password=SecretStr("s3cret"))
        info: UserInfo | None = UserManager._from_auth_payload(payload)

        assert info is not None
        assert info.get_user_id()
        details = info.get_user_details()
        assert isinstance(details, BasicUserInfo)
        assert details.username == "alice"

    def test_basic_payload_deterministic_uuid(self):
        p1 = BasicAuthPayload(method="basic", username="bob", password=SecretStr("pass"))
        p2 = BasicAuthPayload(method="basic", username="bob", password=SecretStr("pass"))

        assert UserManager._from_auth_payload(p1).get_user_id() == \
               UserManager._from_auth_payload(p2).get_user_id()

    def test_basic_different_users_different_uuids(self):
        p1 = BasicAuthPayload(method="basic", username="alice", password=SecretStr("pass"))
        p2 = BasicAuthPayload(method="basic", username="bob", password=SecretStr("pass"))

        assert UserManager._from_auth_payload(p1).get_user_id() != \
               UserManager._from_auth_payload(p2).get_user_id()


# ---------------------------------------------------------------------------
# WebSocketMessageHandler._process_auth_message
# ---------------------------------------------------------------------------
class TestHandlerProcessAuthMessage:

    def _make_handler(self):
        from nat.front_ends.fastapi.message_handler import WebSocketMessageHandler
        mock_socket = MagicMock(spec=WebSocket)
        mock_socket.send_json = AsyncMock()
        handler = WebSocketMessageHandler(
            socket=mock_socket,
            session_manager=MagicMock(),
            step_adaptor=MagicMock(),
            worker=MagicMock(),
        )
        return handler

    def _last_sent_payload(self, handler) -> dict:
        """Return the dict passed to the most recent ``_socket.send_json`` call."""
        handler._socket.send_json.assert_awaited_once()
        return handler._socket.send_json.call_args[0][0]

    async def test_jwt_auth_message_sets_user_id(self):
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        token: str = _make_jwt({"sub": "ws-auth-user", "email": "ws@auth.io"})
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=JwtAuthPayload(method="jwt", token=SecretStr(token)),
        )

        assert handler._user_id is None
        await handler._process_auth_message(msg)
        assert handler._user_id is not None
        assert len(handler._user_id) > 0

        sent = self._last_sent_payload(handler)
        assert sent["type"] == "auth_response_message"
        assert sent["status"] == "success"
        assert sent["user_id"] == handler._user_id
        assert sent["payload"] is None

    async def test_api_key_auth_message_sets_user_id(self):
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=ApiKeyAuthPayload(method="api_key", token=SecretStr("nvapi-xyz")),
        )

        await handler._process_auth_message(msg)
        assert handler._user_id is not None
        sent = self._last_sent_payload(handler)
        assert sent["status"] == "success"
        assert sent["user_id"] == handler._user_id

    async def test_basic_auth_message_sets_user_id(self):
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=BasicAuthPayload(method="basic", username="admin", password=SecretStr("pw")),
        )

        await handler._process_auth_message(msg)
        assert handler._user_id is not None
        sent = self._last_sent_payload(handler)
        assert sent["status"] == "success"

    async def test_invalid_jwt_leaves_user_id_none_and_sends_failure(self):
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=JwtAuthPayload(method="jwt", token=SecretStr("bad-token")),
        )

        await handler._process_auth_message(msg)
        assert handler._user_id is None
        sent = self._last_sent_payload(handler)
        assert sent["type"] == "auth_response_message"
        assert sent["status"] == "error"
        assert sent["user_id"] is None
        assert sent["payload"]["code"] == "user_auth_error"

    async def test_api_key_auth_success_response_contains_user_id(self):
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=ApiKeyAuthPayload(method="api_key", token=SecretStr("nvapi-xyz")),
        )

        await handler._process_auth_message(msg)
        sent = self._last_sent_payload(handler)
        assert sent["user_id"] == handler._user_id

    async def test_basic_auth_success_response_contains_user_id(self):
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=BasicAuthPayload(method="basic", username="admin", password=SecretStr("pw")),
        )

        await handler._process_auth_message(msg)
        sent = self._last_sent_payload(handler)
        assert sent["user_id"] == handler._user_id

    async def test_auth_message_user_id_matches_direct_resolution(self):
        """The handler-stored user_id must match a direct _from_auth_payload call."""
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        token: str = _make_jwt({"sub": "consistency-check", "email": "c@c.io"})
        payload = JwtAuthPayload(method="jwt", token=SecretStr(token))
        msg = WebSocketAuthMessage(type="auth_message", payload=payload)

        await handler._process_auth_message(msg)
        direct_info: UserInfo | None = UserManager._from_auth_payload(payload)
        assert handler._user_id == direct_info.get_user_id()

    async def test_user_id_forwarded_to_session(self):
        """After auth message, ``_run_workflow`` must pass ``_user_id`` to the session."""
        handler = self._make_handler()
        handler._user_id = "pre-set-user-id"
        handler._workflow_schema_type = "generate"

        handler._session_manager.session = MagicMock()
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        handler._session_manager.session.return_value = mock_session

        handler._session_manager.get_workflow_single_output_schema = MagicMock(return_value=None)
        handler._session_manager.get_workflow_streaming_output_schema = MagicMock(return_value=None)
        handler._session_manager._context = MagicMock()

        await handler._run_workflow(payload="test input", user_message_id="msg-1")

        handler._session_manager.session.assert_called_once()
        call_kwargs = handler._session_manager.session.call_args.kwargs
        assert call_kwargs["user_id"] == "pre-set-user-id"

    async def test_empty_jwt_token_sends_failure(self):
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=JwtAuthPayload(method="jwt", token=SecretStr("")),
        )

        await handler._process_auth_message(msg)
        assert handler._user_id is None
        sent = self._last_sent_payload(handler)
        assert sent["status"] == "error"

    async def test_empty_api_key_sends_failure(self):
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=ApiKeyAuthPayload(method="api_key", token=SecretStr("")),
        )

        await handler._process_auth_message(msg)
        assert handler._user_id is None
        sent = self._last_sent_payload(handler)
        assert sent["status"] == "error"
