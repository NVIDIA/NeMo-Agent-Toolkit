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

import asyncio
import base64
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr
from pydantic import ValidationError
from starlette.requests import Request
from starlette.websockets import WebSocket

from nat.data_models.api_server import ApiKeyAuthPayload
from nat.data_models.api_server import BasicAuthPayload
from nat.data_models.api_server import JwtAuthPayload
from nat.data_models.user_info import BasicUserInfo
from nat.data_models.user_info import JwtUserInfo
from nat.data_models.user_info import UserInfo
from nat.runtime.session import SESSION_COOKIE_NAME
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


def _mock_websocket(
    cookie_header: str | None = None,
    auth_header: str | None = None,
    api_key_header: str | None = None,
) -> MagicMock:
    """Create a MagicMock that passes ``isinstance(obj, WebSocket)``."""
    raw_headers: list[tuple[bytes, bytes]] = []
    if cookie_header:
        raw_headers.append((b"cookie", cookie_header.encode()))
    if auth_header:
        raw_headers.append((b"authorization", auth_header.encode()))
    if api_key_header:
        raw_headers.append((b"x-api-key", api_key_header.encode()))

    mock = MagicMock(spec=WebSocket)
    mock.scope = {"headers": raw_headers}
    return mock


class TestFromConnectionRequestCookie:
    """extract_user_from_connection resolves a UserInfo from a session cookie on an HTTP Request."""

    def test_session_cookie_returns_user_info(self):
        """Input: Request with nat-session cookie. Asserts UserInfo with matching details is returned."""
        req = _mock_request(cookies={SESSION_COOKIE_NAME: "abc123"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        assert info.get_user_id()
        assert info.get_user_details() == "abc123"

    def test_deterministic_uuid_from_cookie(self):
        """Input: two Requests with the same cookie value. Asserts both produce the same user_id."""
        req1 = _mock_request(cookies={SESSION_COOKIE_NAME: "same-cookie"})
        req2 = _mock_request(cookies={SESSION_COOKIE_NAME: "same-cookie"})

        assert UserManager.extract_user_from_connection(req1).get_user_id() == \
               UserManager.extract_user_from_connection(req2).get_user_id()

    def test_different_cookies_different_uuids(self):
        """Input: two Requests with different cookie values. Asserts they produce different user_ids."""
        req1 = _mock_request(cookies={SESSION_COOKIE_NAME: "cookie-a"})
        req2 = _mock_request(cookies={SESSION_COOKIE_NAME: "cookie-b"})

        assert UserManager.extract_user_from_connection(req1).get_user_id() != \
               UserManager.extract_user_from_connection(req2).get_user_id()


class TestFromConnectionRequestJwt:
    """extract_user_from_connection resolves a UserInfo from a JWT Bearer token on an HTTP Request."""

    def test_jwt_returns_user_info(self):
        """Input: Request with valid JWT. Asserts UserInfo contains decoded email and subject."""
        token: str = _make_jwt({"sub": "user-123", "email": "test@example.com"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        assert info.get_user_id()
        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.email == "test@example.com"
        assert details.subject == "user-123"

    def test_jwt_identity_claim_sub_preferred(self):
        """Input: JWT with email, preferred_username, and sub. Asserts identity_claim is sub."""
        token: str = _make_jwt({"email": "a@b.com", "preferred_username": "auser", "sub": "sub-1"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.identity_claim == "sub-1"

    def test_jwt_with_roles_and_scopes(self):
        """Input: JWT with roles list and space-separated scope string. Asserts both are parsed."""
        token: str = _make_jwt({"sub": "user-1", "roles": ["admin"], "scope": "read write"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.roles == ["admin"]
        assert details.scopes == ["read", "write"]

    def test_jwt_name_split_into_first_last(self):
        """Input: JWT with ``name`` claim "Jane Doe". Asserts given_name="Jane", family_name="Doe"."""
        token: str = _make_jwt({"sub": "user-1", "name": "Jane Doe"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.given_name == "Jane"
        assert details.family_name == "Doe"

    def test_jwt_given_family_name_preferred_over_name(self):
        """Input: JWT with given_name, family_name, and name. Asserts given/family take precedence."""
        token: str = _make_jwt({"sub": "user-1", "given_name": "Alice", "family_name": "Smith", "name": "Wrong Name"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()

        assert isinstance(details, JwtUserInfo)
        assert details.given_name == "Alice"
        assert details.family_name == "Smith"

    def test_jwt_sub_only_returns_user_info(self):
        """Input: JWT with only sub claim (no email). Asserts identity_claim == sub."""
        token: str = _make_jwt({"sub": "sub-only-user"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        assert info.get_user_id()
        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.identity_claim == "sub-only-user"
        assert details.email is None

    def test_jwt_email_only_returns_user_info(self):
        """Input: JWT with only email claim (no sub). Asserts identity_claim == email."""
        token: str = _make_jwt({"email": "emailonly@test.com"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        assert info.get_user_id()
        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.identity_claim == "emailonly@test.com"
        assert details.subject is None

    def test_jwt_keycloak_realm_access_roles(self):
        """Input: JWT with realm_access.roles. Asserts roles extracted from Keycloak structure."""
        token: str = _make_jwt({"sub": "user-1", "realm_access": {"roles": ["admin", "editor"]}})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()

        assert isinstance(details, JwtUserInfo)
        assert details.roles == ["admin", "editor"]


class TestFromConnectionWebSocketCookie:
    """extract_user_from_connection resolves a UserInfo from a session cookie on a WebSocket."""

    def test_websocket_cookie_returns_user_info(self):
        """Input: WebSocket with nat-session cookie header. Asserts UserInfo with matching details."""
        ws = _mock_websocket(cookie_header=f"{SESSION_COOKIE_NAME}=ws-session-abc")
        info: UserInfo = UserManager.extract_user_from_connection(ws)

        assert info.get_user_id()
        assert info.get_user_details() == "ws-session-abc"

    def test_websocket_cookie_with_multiple_cookies(self):
        """Input: WebSocket with multiple cookies in header. Asserts nat-session is correctly extracted."""
        ws = _mock_websocket(cookie_header=f"other=foo; {SESSION_COOKIE_NAME}=ws-session-xyz; bar=baz")
        info: UserInfo = UserManager.extract_user_from_connection(ws)

        assert info.get_user_details() == "ws-session-xyz"


class TestFromConnectionWebSocketJwt:
    """extract_user_from_connection resolves a UserInfo from a JWT Bearer token on a WebSocket."""

    def test_websocket_jwt_returns_user_info(self):
        """Input: WebSocket with Authorization Bearer header. Asserts JwtUserInfo with decoded email."""
        token: str = _make_jwt({"sub": "ws-jwt-user", "email": "ws@example.com"})
        ws = _mock_websocket(auth_header=f"Bearer {token}")
        info: UserInfo = UserManager.extract_user_from_connection(ws)

        assert info.get_user_id()
        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.email == "ws@example.com"


class TestFromConnectionPriority:
    """extract_user_from_connection prefers session cookie over JWT when both are present."""

    def test_cookie_takes_precedence_over_jwt(self):
        """Input: Request with both cookie and JWT. Asserts cookie-based UserInfo is returned."""
        token: str = _make_jwt({"sub": "jwt-user"})
        req = _mock_request(
            cookies={SESSION_COOKIE_NAME: "cookie-user"},
            headers={"authorization": f"Bearer {token}"},
        )
        info: UserInfo = UserManager.extract_user_from_connection(req)
        assert info.get_user_details() == "cookie-user"

    def test_websocket_cookie_takes_precedence_over_jwt(self):
        """Input: WebSocket with both cookie and JWT. Asserts cookie-based UserInfo is returned."""
        token: str = _make_jwt({"sub": "jwt-user"})
        ws = _mock_websocket(
            cookie_header=f"{SESSION_COOKIE_NAME}=ws-cookie-user",
            auth_header=f"Bearer {token}",
        )
        info: UserInfo = UserManager.extract_user_from_connection(ws)
        assert info.get_user_details() == "ws-cookie-user"


class TestFromConnectionNoCredential:
    """extract_user_from_connection with missing or invalid credentials."""

    def test_no_credentials_returns_none(self):
        """Input: Request with no cookies or headers. Asserts returns None."""
        req = _mock_request()
        assert UserManager.extract_user_from_connection(req) is None

    def test_invalid_jwt_raises(self):
        """Input: Request with undecodable Bearer token. Asserts raises ValueError."""
        req = _mock_request(headers={"authorization": "Bearer not.valid.jwt"})
        with pytest.raises(ValueError, match="Failed to decode JWT"):
            UserManager.extract_user_from_connection(req)

    def test_jwt_without_identity_claim_raises(self):
        """Input: Request with JWT containing only ``iss``. Asserts raises ValueError."""
        token: str = _make_jwt({"iss": "some-issuer"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        with pytest.raises(ValueError, match="no usable identity claim"):
            UserManager.extract_user_from_connection(req)

    def test_empty_websocket_returns_none(self):
        """Input: WebSocket with no headers. Asserts returns None."""
        ws = _mock_websocket()
        assert UserManager.extract_user_from_connection(ws) is None


class TestFromAuthPayloadJwt:
    """_from_auth_payload resolves UserInfo from a JwtAuthPayload."""

    def test_jwt_payload_returns_user_info(self):
        """Input: valid JWT payload. Asserts returned UserInfo has decoded email and subject."""
        token: str = _make_jwt({"sub": "payload-user", "email": "p@example.com"})
        payload = JwtAuthPayload(method="jwt", token=SecretStr(token))
        info: UserInfo = UserManager._from_auth_payload(payload)

        assert info.get_user_id()
        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.email == "p@example.com"
        assert details.subject == "payload-user"

    def test_jwt_payload_deterministic_uuid(self):
        """Input: same JWT payload twice. Asserts both produce the same user_id."""
        token: str = _make_jwt({"sub": "stable-user", "email": "s@example.com"})
        p1 = JwtAuthPayload(method="jwt", token=SecretStr(token))
        p2 = JwtAuthPayload(method="jwt", token=SecretStr(token))

        assert UserManager._from_auth_payload(p1).get_user_id() == \
               UserManager._from_auth_payload(p2).get_user_id()

    def test_jwt_payload_invalid_token_raises(self):
        """Input: JWT payload with non-JWT string. Asserts raises ValueError matching "malformed"."""
        payload = JwtAuthPayload(method="jwt", token=SecretStr("not-a-jwt"))
        with pytest.raises(ValueError, match="malformed"):
            UserManager._from_auth_payload(payload)

    def test_jwt_payload_empty_token_raises(self):
        """Input: JWT payload with empty token. Asserts raises ValidationError (min_length=1)."""
        with pytest.raises(ValidationError):
            JwtAuthPayload(method="jwt", token=SecretStr(""))

    def test_jwt_payload_no_identity_claim_raises(self):
        """Input: valid JWT but only iss claim. Asserts raises ValueError matching "no usable identity claim"."""
        token: str = _make_jwt({"iss": "some-issuer"})
        payload = JwtAuthPayload(method="jwt", token=SecretStr(token))
        with pytest.raises(ValueError, match="no usable identity claim"):
            UserManager._from_auth_payload(payload)


class TestFromAuthPayloadApiKey:
    """_from_auth_payload resolves UserInfo from an ApiKeyAuthPayload."""

    def test_api_key_payload_returns_user_info(self):
        """Input: API key payload. Asserts UserInfo details match the token value."""
        payload = ApiKeyAuthPayload(method="api_key", token=SecretStr("nvapi-abc123"))
        info: UserInfo = UserManager._from_auth_payload(payload)

        assert info.get_user_id()
        assert info.get_user_details() == "nvapi-abc123"

    def test_api_key_deterministic_uuid(self):
        """Input: same API key twice. Asserts both produce the same user_id."""
        p1 = ApiKeyAuthPayload(method="api_key", token=SecretStr("same-key"))
        p2 = ApiKeyAuthPayload(method="api_key", token=SecretStr("same-key"))

        assert UserManager._from_auth_payload(p1).get_user_id() == \
               UserManager._from_auth_payload(p2).get_user_id()

    def test_api_key_empty_token_raises(self):
        """Input: API key payload with empty token. Asserts raises ValidationError (min_length=1)."""
        with pytest.raises(ValidationError):
            ApiKeyAuthPayload(method="api_key", token=SecretStr(""))


class TestFromAuthPayloadBasic:
    """_from_auth_payload resolves UserInfo from a BasicAuthPayload."""

    def test_basic_payload_returns_user_info(self):
        """Input: basic auth payload. Asserts UserInfo details is BasicUserInfo with matching username."""
        payload = BasicAuthPayload(method="basic", username="alice", password=SecretStr("s3cret"))
        info: UserInfo = UserManager._from_auth_payload(payload)

        assert info.get_user_id()
        details = info.get_user_details()
        assert isinstance(details, BasicUserInfo)
        assert details.username == "alice"

    def test_basic_payload_deterministic_uuid(self):
        """Input: same basic payload twice. Asserts both produce the same user_id."""
        p1 = BasicAuthPayload(method="basic", username="bob", password=SecretStr("pass"))
        p2 = BasicAuthPayload(method="basic", username="bob", password=SecretStr("pass"))

        assert UserManager._from_auth_payload(p1).get_user_id() == \
               UserManager._from_auth_payload(p2).get_user_id()

    def test_basic_different_users_different_uuids(self):
        """Input: two different basic payloads. Asserts they produce different user_ids."""
        p1 = BasicAuthPayload(method="basic", username="alice", password=SecretStr("pass"))
        p2 = BasicAuthPayload(method="basic", username="bob", password=SecretStr("pass"))

        assert UserManager._from_auth_payload(p1).get_user_id() != \
               UserManager._from_auth_payload(p2).get_user_id()


class TestHandlerProcessAuthMessage:
    """_process_auth_message resolves user identity from WebSocket auth messages and sends responses."""

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
        """Input: valid JWT auth message. Asserts handler._user_id is set and success response sent."""
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
        """Input: API key auth message. Asserts handler._user_id is set and success response sent."""
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
        """Input: basic auth message. Asserts handler._user_id is set and success response sent."""
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
        """Input: malformed JWT auth message. Asserts user_id stays None and error response sent."""
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
        assert sent["payload"]["details"]

    async def test_api_key_auth_success_response_contains_user_id(self):
        """Input: API key auth message. Asserts response user_id matches handler._user_id."""
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
        """Input: basic auth message. Asserts response user_id matches handler._user_id."""
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

    async def test_success_response_payload_is_none(self):
        """Input: valid JWT auth message. Asserts success response payload is None (no error)."""
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        token: str = _make_jwt({"sub": "u", "email": "a@b.com"})
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=JwtAuthPayload(method="jwt", token=SecretStr(token)),
        )

        await handler._process_auth_message(msg)
        sent: dict = self._last_sent_payload(handler)
        assert sent["payload"] is None

    async def test_error_response_user_id_is_none(self):
        """Input: malformed JWT auth message. Asserts error response user_id is None."""
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=JwtAuthPayload(method="jwt", token=SecretStr("bad-token")),
        )

        await handler._process_auth_message(msg)
        sent: dict = self._last_sent_payload(handler)
        assert sent["user_id"] is None

    async def test_error_response_has_details(self):
        """Input: malformed JWT auth message. Asserts error response contains non-empty details string."""
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=JwtAuthPayload(method="jwt", token=SecretStr("bad-token")),
        )

        await handler._process_auth_message(msg)
        sent: dict = self._last_sent_payload(handler)
        assert isinstance(sent["payload"]["details"], str)
        assert len(sent["payload"]["details"]) > 0

    async def test_second_auth_message_overrides_user_id(self):
        """Input: two auth messages for different users. Asserts second overrides first user_id."""
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()

        token_a: str = _make_jwt({"sub": "user-a", "email": "user-a@x.com"})
        msg_a = WebSocketAuthMessage(
            type="auth_message",
            payload=JwtAuthPayload(method="jwt", token=SecretStr(token_a)),
        )
        await handler._process_auth_message(msg_a)
        first_id: str = handler._user_id

        handler._socket.send_json.reset_mock()

        token_b: str = _make_jwt({"sub": "user-b", "email": "user-b@x.com"})
        msg_b = WebSocketAuthMessage(
            type="auth_message",
            payload=JwtAuthPayload(method="jwt", token=SecretStr(token_b)),
        )
        await handler._process_auth_message(msg_b)
        second_id: str = handler._user_id

        assert first_id != second_id
        expected_b: str = UserManager._from_auth_payload(msg_b.payload).get_user_id()
        assert second_id == expected_b

    async def test_auth_then_workflow_passes_user_id(self):
        """Input: auth message then _run_workflow. Asserts session is called with the resolved user_id."""
        from nat.data_models.api_server import WebSocketAuthMessage
        handler = self._make_handler()
        token: str = _make_jwt({"sub": "flow-user", "email": "flow@x.com"})
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=JwtAuthPayload(method="jwt", token=SecretStr(token)),
        )
        await handler._process_auth_message(msg)
        resolved_id: str = handler._user_id

        handler._socket.send_json.reset_mock()
        handler._workflow_schema_type = "generate"
        handler._session_manager.session = MagicMock()
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        handler._session_manager.session.return_value = mock_session
        handler._session_manager.get_workflow_single_output_schema = MagicMock(return_value=None)
        handler._session_manager.get_workflow_streaming_output_schema = MagicMock(return_value=None)
        handler._session_manager._context = MagicMock()

        await handler._run_workflow(payload="hello", user_message_id="m-1")
        call_kwargs: dict = handler._session_manager.session.call_args.kwargs
        assert call_kwargs["user_id"] == resolved_id

    async def test_empty_jwt_token_rejected_at_model_level(self):
        """Input: JWT auth message with empty token. Asserts ValidationError at construction (min_length=1)."""
        with pytest.raises(ValidationError):
            JwtAuthPayload(method="jwt", token=SecretStr(""))

    async def test_empty_api_key_rejected_at_model_level(self):
        """Input: API key auth message with empty token. Asserts ValidationError at construction (min_length=1)."""
        with pytest.raises(ValidationError):
            ApiKeyAuthPayload(method="api_key", token=SecretStr(""))


class TestSessionUserIdResolution:
    """SessionManager.session() resolves user_id from the connection when not explicitly provided."""

    def _make_session_manager(self, *, is_per_user: bool = False):
        """Build a minimal SessionManager with just enough internals for session()."""
        from nat.builder.context import ContextState
        from nat.runtime.session import SessionManager

        sm = object.__new__(SessionManager)
        sm._context_state = ContextState.get()
        sm._is_workflow_per_user = is_per_user
        sm._shared_workflow = MagicMock()
        sm._semaphore = MagicMock()
        sm._context = MagicMock()
        sm._per_user_builders = {}
        sm._per_user_builders_lock = MagicMock()
        return sm

    async def test_user_id_provided_skips_extraction(self):
        """Input: explicit user_id kwarg. Asserts extract_user_from_connection is never called."""
        from unittest.mock import patch

        sm = self._make_session_manager()
        ws = _mock_websocket()

        with patch.object(UserManager, "extract_user_from_connection") as mock_extract:
            async with sm.session(user_id="explicit-id", http_connection=ws) as session:
                assert session._user_id == "explicit-id"
            mock_extract.assert_not_called()

    async def test_websocket_cookie_sets_user_id_in_context(self):
        """Input: WebSocket with session cookie. Asserts session user_id matches cookie-derived UUID."""
        from unittest.mock import patch

        sm = self._make_session_manager()
        ws = _mock_websocket(cookie_header=f"{SESSION_COOKIE_NAME}=cookie-value")

        cookie_info: UserInfo = UserInfo._from_session_cookie("cookie-value")
        expected_id: str = cookie_info.get_user_id()

        with patch.object(UserManager, "extract_user_from_connection", return_value=cookie_info):
            async with sm.session(http_connection=ws) as session:
                assert session._user_id == expected_id

    async def test_request_jwt_sets_user_id_in_context(self):
        """Input: HTTP Request with JWT. Asserts session user_id matches JWT-derived UUID."""
        from unittest.mock import patch

        sm = self._make_session_manager()
        sm.set_metadata_from_http_request = AsyncMock(return_value=(None, None))

        jwt_info: JwtUserInfo = JwtUserInfo(
            email="a@b.com",
            subject="sub-1",
            claims={
                "email": "a@b.com", "sub": "sub-1"
            },
        )
        user_info: UserInfo = UserInfo._from_jwt(jwt_info)
        expected_id: str = user_info.get_user_id()
        req = _mock_request(headers={"authorization": "Bearer fake"})

        with patch.object(UserManager, "extract_user_from_connection", return_value=user_info):
            async with sm.session(http_connection=req) as session:
                assert session._user_id == expected_id

    async def test_no_credential_shared_workflow_user_id_is_none(self):
        """Input: shared workflow, WebSocket with no creds. Asserts session proceeds with user_id=None."""
        from unittest.mock import patch

        sm = self._make_session_manager(is_per_user=False)
        ws = _mock_websocket()

        with patch.object(UserManager, "extract_user_from_connection", return_value=None):
            async with sm.session(http_connection=ws) as session:
                assert session._user_id is None

    async def test_no_credential_per_user_workflow_raises(self):
        """Input: per-user workflow, WebSocket with no creds. Asserts raises ValueError."""
        from unittest.mock import patch

        sm = self._make_session_manager(is_per_user=True)
        ws = _mock_websocket()

        with patch.object(UserManager, "extract_user_from_connection", return_value=None):
            with pytest.raises(ValueError, match="user_id is required for per-user workflow"):
                async with sm.session(http_connection=ws):
                    pass

    async def test_broken_jwt_per_user_workflow_raises(self):
        """Input: per-user workflow, broken JWT. Asserts ValueError propagates from extraction."""
        from unittest.mock import patch

        sm = self._make_session_manager(is_per_user=True)
        ws = _mock_websocket()

        with patch.object(
                UserManager,
                "extract_user_from_connection",
                side_effect=ValueError("Failed to decode JWT"),
        ):
            with pytest.raises(ValueError, match="Failed to decode JWT"):
                async with sm.session(http_connection=ws):
                    pass

    async def test_broken_jwt_shared_workflow_raises(self):
        """Input: shared workflow, broken JWT. Asserts ValueError propagates (fail fast)."""
        from unittest.mock import patch

        sm = self._make_session_manager(is_per_user=False)
        ws = _mock_websocket()

        with patch.object(
                UserManager,
                "extract_user_from_connection",
                side_effect=ValueError("Failed to decode JWT"),
        ):
            with pytest.raises(ValueError, match="Failed to decode JWT"):
                async with sm.session(http_connection=ws):
                    pass


class TestPerUserBuilderUserIdWiring:
    """Per-user workflow builders are keyed by user_id for isolation and reuse."""

    def _make_session_manager(self):
        from nat.builder.context import ContextState
        from nat.runtime.session import PerUserBuilderInfo
        from nat.runtime.session import SessionManager

        sm = object.__new__(SessionManager)
        sm._context_state = ContextState.get()
        sm._is_workflow_per_user = True
        sm._shared_workflow = MagicMock()
        sm._shared_builder = MagicMock()
        sm._semaphore = MagicMock()
        sm._context = MagicMock()
        sm._per_user_builders = {}
        sm._per_user_builders_lock = asyncio.Lock()
        sm._config = MagicMock()
        sm._entry_function = "main"
        sm._max_concurrency = 1
        sm._per_user_session_timeout = MagicMock(total_seconds=MagicMock(return_value=60))
        sm._per_user_session_cleanup_interval = MagicMock(total_seconds=MagicMock(return_value=30))
        sm._shutdown_event = asyncio.Event()
        return sm, PerUserBuilderInfo

    async def test_same_user_id_reuses_builder(self):
        """Input: same user_id twice. Asserts second call returns the same builder and dict has 1 entry."""
        from unittest.mock import patch

        sm, PerUserBuilderInfo = self._make_session_manager()
        mock_builder = MagicMock()
        mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
        mock_builder.__aexit__ = AsyncMock(return_value=False)
        mock_builder.populate_builder = AsyncMock()
        mock_builder.build = AsyncMock(return_value=MagicMock())

        with patch("nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder", return_value=mock_builder):
            _, wf1 = await sm._get_or_create_per_user_builder("user-a")
            _, wf2 = await sm._get_or_create_per_user_builder("user-a")

        assert wf1 is wf2
        assert len(sm._per_user_builders) == 1

    async def test_different_user_ids_create_separate_builders(self):
        """Input: two different user_ids. Asserts dict has 2 entries with distinct workflows."""
        from unittest.mock import patch

        sm, PerUserBuilderInfo = self._make_session_manager()

        def make_builder(*args, **kwargs):
            b = MagicMock()
            b.__aenter__ = AsyncMock(return_value=b)
            b.__aexit__ = AsyncMock(return_value=False)
            b.populate_builder = AsyncMock()
            b.build = AsyncMock(return_value=MagicMock())
            return b

        with patch("nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder", side_effect=make_builder):
            _, wf_a = await sm._get_or_create_per_user_builder("user-a")
            _, wf_b = await sm._get_or_create_per_user_builder("user-b")

        assert len(sm._per_user_builders) == 2
        assert wf_a is not wf_b

    async def test_cleanup_removes_builder_by_user_id(self):
        """Input: inactive builder past timeout. Asserts cleanup removes it from _per_user_builders."""
        import asyncio as _asyncio
        from datetime import datetime
        from datetime import timedelta

        from nat.runtime.session import PerUserBuilderInfo

        sm, _ = self._make_session_manager()
        sm._per_user_session_timeout = timedelta(seconds=1)

        mock_builder = MagicMock()
        mock_builder.__aexit__ = AsyncMock(return_value=False)

        builder_info = PerUserBuilderInfo(
            builder=mock_builder,
            workflow=MagicMock(),
            semaphore=_asyncio.Semaphore(1),
            last_activity=datetime.now() - timedelta(seconds=10),
            ref_count=0,
            lock=_asyncio.Lock(),
        )
        sm._per_user_builders["user-a"] = builder_info

        cleaned: int = await sm._cleanup_inactive_per_user_builders()
        assert cleaned == 1
        assert "user-a" not in sm._per_user_builders


class TestContextVarPropagation:
    """ContextState.user_id context var is set, read, and reset correctly across session boundaries."""

    def test_context_var_set_and_readable(self):
        """Input: set user_id to "test-user". Asserts get() returns "test-user"."""
        from nat.builder.context import ContextState
        state: ContextState = ContextState.get()
        token = state.user_id.set("test-user")
        try:
            assert state.user_id.get() == "test-user"
        finally:
            state.user_id.reset(token)

    def test_context_var_reset_restores_previous(self):
        """Input: set "user-a", then "user-b", then reset. Asserts get() returns "user-a" after reset."""
        from nat.builder.context import ContextState
        state: ContextState = ContextState.get()
        token_a = state.user_id.set("user-a")
        try:
            token_b = state.user_id.set("user-b")
            assert state.user_id.get() == "user-b"
            state.user_id.reset(token_b)
            assert state.user_id.get() == "user-a"
        finally:
            state.user_id.reset(token_a)


class TestGetSessionCookieEdgeCases:
    """_get_session_cookie edge cases for missing or irrelevant cookie headers."""

    def test_websocket_no_cookie_header_returns_none(self):
        """Input: WebSocket with empty headers. Asserts _get_session_cookie returns None."""
        ws = _mock_websocket()
        assert UserManager._get_session_cookie(ws) is None

    def test_websocket_cookie_header_without_nat_session_returns_none(self):
        """Input: WebSocket with cookie header that lacks nat-session. Asserts returns None."""
        ws = _mock_websocket(cookie_header="other=foo; bar=baz")
        assert UserManager._get_session_cookie(ws) is None

    def test_request_empty_cookies_returns_none(self):
        """Input: Request with empty cookies dict. Asserts _get_session_cookie returns None."""
        req = _mock_request(cookies={})
        assert UserManager._get_session_cookie(req) is None


class TestUserInfoFromJwtClaimExtraction:
    """_user_info_from_jwt extracts groups, audience, client_id, exp/iat, name fallbacks."""

    def test_groups_extracted_from_claims(self):
        """Input: claims with groups list. Asserts details.groups == ["g1", "g2"]."""
        token: str = _make_jwt({"sub": "u", "groups": ["g1", "g2"]})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.groups == ["g1", "g2"]

    def test_audience_as_string_wrapped_in_list(self):
        """Input: claims with aud="my-app" (string). Asserts details.audience == ["my-app"]."""
        token: str = _make_jwt({"sub": "u", "aud": "my-app"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.audience == ["my-app"]

    def test_audience_as_list_preserved(self):
        """Input: claims with aud=["a", "b"]. Asserts details.audience == ["a", "b"]."""
        token: str = _make_jwt({"sub": "u", "aud": ["a", "b"]})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.audience == ["a", "b"]

    def test_client_id_from_azp(self):
        """Input: claims with azp="client-1". Asserts details.client_id == "client-1"."""
        token: str = _make_jwt({"sub": "u", "azp": "client-1"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.client_id == "client-1"

    def test_client_id_from_client_id_claim(self):
        """Input: claims with client_id="client-2". Asserts details.client_id == "client-2"."""
        token: str = _make_jwt({"sub": "u", "client_id": "client-2"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.client_id == "client-2"

    def test_client_id_azp_preferred_over_client_id(self):
        """Input: claims with both azp="a" and client_id="b". Asserts details.client_id == "a"."""
        token: str = _make_jwt({"sub": "u", "azp": "a", "client_id": "b"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.client_id == "a"

    def test_exp_iat_as_int(self):
        """Input: claims with exp and iat as integers. Asserts details stores both correctly."""
        token: str = _make_jwt({"sub": "u", "exp": 1700000000, "iat": 1699999000})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.expires_at == 1700000000
        assert details.issued_at == 1699999000

    def test_issuer_extracted(self):
        """Input: claims with iss. Asserts details.issuer == the issuer string."""
        token: str = _make_jwt({"sub": "u", "iss": "https://idp.example.com"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.issuer == "https://idp.example.com"

    def test_name_single_word_given_name_only(self):
        """Input: claims with name="Alice" (single word). Asserts given_name="Alice", family_name is None."""
        token: str = _make_jwt({"sub": "u", "name": "Alice"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.given_name == "Alice"
        assert details.family_name is None

    def test_roles_direct_list(self):
        """Input: claims with roles=["admin"]. Asserts details.roles == ["admin"]."""
        token: str = _make_jwt({"sub": "u", "roles": ["admin"]})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        details = UserManager.extract_user_from_connection(req).get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.roles == ["admin"]


class TestExtractUserFromConnectionEdgeCases:
    """extract_user_from_connection priority and error propagation."""

    def test_cookie_present_jwt_broken_still_returns_cookie_user(self):
        """Input: Request with valid cookie AND broken JWT. Asserts cookie user returned (JWT never evaluated)."""
        req = _mock_request(
            cookies={SESSION_COOKIE_NAME: "abc"},
            headers={"authorization": "Bearer not.valid.jwt"},
        )
        info: UserInfo = UserManager.extract_user_from_connection(req)
        assert info.get_user_details() == "abc"

    def test_no_cookie_broken_jwt_raises(self):
        """Input: Request with no cookie and broken JWT. Asserts raises ValueError."""
        req = _mock_request(headers={"authorization": "Bearer not.valid.jwt"})
        with pytest.raises(ValueError, match="Failed to decode"):
            UserManager.extract_user_from_connection(req)

    def test_websocket_jwt_no_identity_claim_raises(self):
        """Input: WebSocket with JWT containing only iss. Asserts raises ValueError."""
        token: str = _make_jwt({"iss": "x"})
        ws = _mock_websocket(auth_header=f"Bearer {token}")
        with pytest.raises(ValueError, match="no usable identity claim"):
            UserManager.extract_user_from_connection(ws)


class TestFromConnectionRequestBasicAuth:
    """extract_user_from_connection resolves a UserInfo from HTTP Basic Auth."""

    def test_basic_auth_returns_user_info(self):
        """Input: Request with Authorization: Basic header. Asserts UserInfo with BasicUserInfo details."""
        b64: str = base64.b64encode(b"alice:s3cret").decode()
        req = _mock_request(headers={"authorization": f"Basic {b64}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        assert info is not None
        assert info.get_user_id()
        details = info.get_user_details()
        assert isinstance(details, BasicUserInfo)
        assert details.username == "alice"

    def test_basic_auth_deterministic_uuid(self):
        """Input: same Basic auth twice. Asserts both produce the same user_id."""
        b64: str = base64.b64encode(b"alice:s3cret").decode()
        req1 = _mock_request(headers={"authorization": f"Basic {b64}"})
        req2 = _mock_request(headers={"authorization": f"Basic {b64}"})

        assert UserManager.extract_user_from_connection(req1).get_user_id() == \
               UserManager.extract_user_from_connection(req2).get_user_id()

    def test_basic_auth_matches_direct_construction(self):
        """Input: Basic auth via header matches UserInfo(basic_user=...) with same creds."""
        b64: str = base64.b64encode(b"alice:s3cret").decode()
        req = _mock_request(headers={"authorization": f"Basic {b64}"})
        from_connection: UserInfo = UserManager.extract_user_from_connection(req)

        direct: UserInfo = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("s3cret")))
        assert from_connection.get_user_id() == direct.get_user_id()

    def test_basic_auth_invalid_base64_raises(self):
        """Input: Basic auth with invalid base64. Asserts raises ValueError."""
        req = _mock_request(headers={"authorization": "Basic not-valid-base64!!!"})
        with pytest.raises(ValueError, match="Failed to decode Basic auth credential"):
            UserManager.extract_user_from_connection(req)

    def test_basic_auth_no_colon_raises(self):
        """Input: Basic auth with base64 that has no colon. Asserts raises ValueError."""
        b64: str = base64.b64encode(b"nocolon").decode()
        req = _mock_request(headers={"authorization": f"Basic {b64}"})
        with pytest.raises(ValueError, match="colon separator"):
            UserManager.extract_user_from_connection(req)

    def test_basic_auth_empty_username_raises(self):
        """Input: Basic auth with empty username (:password). Asserts raises ValueError."""
        b64: str = base64.b64encode(b":password").decode()
        req = _mock_request(headers={"authorization": f"Basic {b64}"})
        with pytest.raises(ValueError, match="username must not be empty"):
            UserManager.extract_user_from_connection(req)

    def test_basic_auth_websocket(self):
        """Input: WebSocket with Basic auth header. Asserts returns UserInfo with BasicUserInfo."""
        b64: str = base64.b64encode(b"bob:pass123").decode()
        ws = _mock_websocket(auth_header=f"Basic {b64}")
        info: UserInfo = UserManager.extract_user_from_connection(ws)

        assert info is not None
        details = info.get_user_details()
        assert isinstance(details, BasicUserInfo)
        assert details.username == "bob"


class TestFromConnectionRequestApiKey:
    """extract_user_from_connection resolves a UserInfo from a non-JWT Bearer token (API key)."""

    def test_api_key_bearer_returns_user_info(self):
        """Input: Bearer token that is not a JWT (no dots). Asserts treated as API key."""
        req = _mock_request(headers={"authorization": "Bearer sk-my-api-key-123"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        assert info is not None
        assert info.get_user_id()
        assert info.get_user_details() == "sk-my-api-key-123"

    def test_api_key_deterministic_uuid(self):
        """Input: same API key Bearer token twice. Asserts same user_id."""
        req1 = _mock_request(headers={"authorization": "Bearer sk-key-xyz"})
        req2 = _mock_request(headers={"authorization": "Bearer sk-key-xyz"})

        assert UserManager.extract_user_from_connection(req1).get_user_id() == \
               UserManager.extract_user_from_connection(req2).get_user_id()

    def test_api_key_matches_from_api_key_factory(self):
        """Input: API key via Bearer header matches UserInfo._from_api_key with same key."""
        req = _mock_request(headers={"authorization": "Bearer sk-test-key"})
        from_connection: UserInfo = UserManager.extract_user_from_connection(req)
        from_factory: UserInfo = UserInfo._from_api_key("sk-test-key")
        assert from_connection.get_user_id() == from_factory.get_user_id()

    def test_api_key_matches_directly_constructed_user(self):
        """Input: API key via Bearer header matches UserInfo(api_key=...) with same key."""
        req = _mock_request(headers={"authorization": "Bearer sk-direct-key"})
        from_connection: UserInfo = UserManager.extract_user_from_connection(req)
        from_constructor: UserInfo = UserInfo(api_key=SecretStr("sk-direct-key"))
        assert from_connection.get_user_id() == from_constructor.get_user_id()

    def test_api_key_one_dot_treated_as_api_key(self):
        """Input: Bearer token with 1 dot. Asserts treated as API key (not JWT)."""
        req = _mock_request(headers={"authorization": "Bearer prefix.suffix"})
        info: UserInfo = UserManager.extract_user_from_connection(req)
        assert info is not None
        assert info.get_user_details() == "prefix.suffix"

    def test_api_key_websocket(self):
        """Input: WebSocket with non-JWT Bearer token. Asserts treated as API key."""
        ws = _mock_websocket(auth_header="Bearer nvapi-ws-key")
        info: UserInfo = UserManager.extract_user_from_connection(ws)

        assert info is not None
        assert info.get_user_details() == "nvapi-ws-key"


class TestFromConnectionXApiKeyHeader:
    """extract_user_from_connection resolves a UserInfo from an X-API-Key header."""

    def test_x_api_key_header_returns_user_info(self):
        """Input: Request with X-API-Key header only. Asserts treated as API key."""
        req = _mock_request(headers={"x-api-key": "nvapi-header-key"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        assert info is not None
        assert info.get_user_details() == "nvapi-header-key"

    def test_x_api_key_deterministic_uuid(self):
        """Input: Same X-API-Key twice. Asserts same user_id."""
        req1 = _mock_request(headers={"x-api-key": "nvapi-stable"})
        req2 = _mock_request(headers={"x-api-key": "nvapi-stable"})

        assert UserManager.extract_user_from_connection(req1).get_user_id() == \
               UserManager.extract_user_from_connection(req2).get_user_id()

    def test_x_api_key_matches_bearer_api_key(self):
        """Input: Same key via X-API-Key and Bearer. Asserts same user_id."""
        from_x_header: UserInfo = UserManager.extract_user_from_connection(
            _mock_request(headers={"x-api-key": "shared-key"}))
        from_bearer: UserInfo = UserManager.extract_user_from_connection(
            _mock_request(headers={"authorization": "Bearer shared-key"}))

        assert from_x_header.get_user_id() == from_bearer.get_user_id()

    def test_x_api_key_websocket(self):
        """Input: WebSocket with X-API-Key header. Asserts treated as API key."""
        ws = _mock_websocket(api_key_header="nvapi-ws-x-key")
        info: UserInfo = UserManager.extract_user_from_connection(ws)

        assert info is not None
        assert info.get_user_details() == "nvapi-ws-x-key"

    def test_authorization_takes_precedence_over_x_api_key(self):
        """Input: Request with both Authorization Bearer and X-API-Key. Asserts Authorization wins."""
        req = _mock_request(headers={
            "authorization": "Bearer bearer-key",
            "x-api-key": "x-header-key",
        })
        info: UserInfo = UserManager.extract_user_from_connection(req)
        assert info.get_user_details() == "bearer-key"

    def test_x_api_key_used_when_auth_scheme_unsupported(self):
        """Input: Unsupported Authorization scheme + X-API-Key. Asserts X-API-Key is used as fallback."""
        req = _mock_request(headers={
            "authorization": "Digest realm=test",
            "x-api-key": "fallback-key",
        })
        info: UserInfo = UserManager.extract_user_from_connection(req)

        assert info is not None
        assert info.get_user_details() == "fallback-key"


class TestJwtVsApiKeyDiscrimination:
    """Bearer tokens with exactly 2 dots are treated as JWT; all others as API key."""

    def test_three_part_token_treated_as_jwt(self):
        """Input: valid JWT (3 dot-separated parts). Asserts identity_claim from JWT."""
        token: str = _make_jwt({"sub": "jwt-user", "email": "jwt@test.com"})
        req = _mock_request(headers={"authorization": f"Bearer {token}"})
        info: UserInfo = UserManager.extract_user_from_connection(req)

        details = info.get_user_details()
        assert isinstance(details, JwtUserInfo)
        assert details.email == "jwt@test.com"

    def test_no_dot_token_treated_as_api_key(self):
        """Input: Bearer token with no dots. Asserts treated as API key."""
        req = _mock_request(headers={"authorization": "Bearer sk-no-dots"})
        info: UserInfo = UserManager.extract_user_from_connection(req)
        assert info.get_user_details() == "sk-no-dots"

    def test_one_dot_token_treated_as_api_key(self):
        """Input: Bearer token with 1 dot. Asserts treated as API key."""
        req = _mock_request(headers={"authorization": "Bearer one.dot"})
        info: UserInfo = UserManager.extract_user_from_connection(req)
        assert info.get_user_details() == "one.dot"

    def test_three_dot_token_treated_as_api_key(self):
        """Input: Bearer token with 3 dots (not JWT structure). Asserts treated as API key."""
        req = _mock_request(headers={"authorization": "Bearer a.b.c.d"})
        info: UserInfo = UserManager.extract_user_from_connection(req)
        assert info.get_user_details() == "a.b.c.d"

    def test_malformed_jwt_structure_raises(self):
        """Input: 3-part token with invalid base64 payload. Asserts raises ValueError (JWT decode fails)."""
        req = _mock_request(headers={"authorization": "Bearer not.valid.jwt"})
        with pytest.raises(ValueError, match="Failed to decode JWT"):
            UserManager.extract_user_from_connection(req)


class TestResolutionChainPriority:
    """Full resolution chain: cookie > Authorization (JWT / API key / Basic) > X-API-Key."""

    def test_cookie_takes_precedence_over_basic_auth(self):
        """Input: Request with both cookie and Basic auth. Asserts cookie wins."""
        b64: str = base64.b64encode(b"alice:pass").decode()
        req = _mock_request(
            cookies={SESSION_COOKIE_NAME: "cookie-value"},
            headers={"authorization": f"Basic {b64}"},
        )
        info: UserInfo = UserManager.extract_user_from_connection(req)
        assert info.get_user_details() == "cookie-value"

    def test_cookie_takes_precedence_over_api_key(self):
        """Input: Request with both cookie and API key Bearer. Asserts cookie wins."""
        req = _mock_request(
            cookies={SESSION_COOKIE_NAME: "cookie-value"},
            headers={"authorization": "Bearer sk-api-key"},
        )
        info: UserInfo = UserManager.extract_user_from_connection(req)
        assert info.get_user_details() == "cookie-value"

    def test_unknown_scheme_returns_none(self):
        """Input: Request with unsupported auth scheme. Asserts returns None."""
        req = _mock_request(headers={"authorization": "Digest realm=test"})
        assert UserManager.extract_user_from_connection(req) is None
