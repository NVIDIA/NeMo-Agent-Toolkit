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

import asyncio
import base64
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from starlette.websockets import WebSocket
from starlette.websockets import WebSocketDisconnect

from nat.data_models.api_server import ApiKeyAuthPayload
from nat.data_models.api_server import AuthMessageStatus
from nat.data_models.api_server import JwtAuthPayload
from nat.data_models.api_server import OAuthMode
from nat.data_models.api_server import OAuthModePreferencePayload
from nat.data_models.api_server import WebSocketAuthMessage
from nat.data_models.api_server import WebSocketMessageType
from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import FlowState
from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler
from nat.front_ends.fastapi.message_handler import WebSocketMessageHandler


def _make_jwt(claims: dict) -> str:
    """Build a minimal unsigned JWT for exercising the verifier boundary."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}."


def _make_message_handler(
    accepted_identity_credentials=None,
    jwt_validators=None,
    identity_header=None,
) -> tuple[WebSocketMessageHandler, MagicMock, WebSocketAuthenticationFlowHandler]:
    """Build a WebSocketMessageHandler with a mockable socket and a real flow handler."""
    socket = MagicMock(spec=WebSocket)
    session_manager = MagicMock()
    session_manager.get_workflow_single_output_schema.return_value = None
    session_manager.get_workflow_streaming_output_schema.return_value = None
    handler = WebSocketMessageHandler(
        socket=socket,
        session_manager=session_manager,
        step_adaptor=MagicMock(),
        worker=MagicMock(),
        accepted_identity_credentials=accepted_identity_credentials,
        jwt_validators=jwt_validators,
        identity_header=identity_header,
    )
    flow_handler = WebSocketAuthenticationFlowHandler(
        add_flow_cb=AsyncMock(),
        remove_flow_cb=AsyncMock(),
        web_socket_message_handler=handler,
    )
    handler.set_flow_handler(flow_handler)
    return handler, socket, flow_handler


async def test_context_manager_resolves_connection_identity_before_restoration():
    """Connection credentials establish the owner before reconnection is attempted."""
    handler, socket, _ = _make_message_handler()
    user_info = MagicMock()
    user_info.get_user_id.return_value = "user-a"
    restore = AsyncMock()
    handler._restore_execution_state = restore

    with patch(
            "nat.front_ends.fastapi.message_handler.UserManager.extract_user_from_connection_with_verification",
            return_value=user_info,
    ):
        await handler.__aenter__()

    socket.accept.assert_awaited_once()
    assert handler._user_id == "user-a"
    restore.assert_awaited_once()


async def test_context_manager_uses_configured_identity_header():
    """The configured identity header is passed through the verified connection resolver."""
    handler, socket, _ = _make_message_handler(identity_header="X-User-ID")
    user_info = MagicMock()
    user_info.get_user_id.return_value = "resolved-user"

    with patch(
            "nat.front_ends.fastapi.message_handler.UserManager.extract_user_from_connection_with_verification",
            return_value=user_info,
    ) as resolver:
        await handler.__aenter__()

    assert resolver.await_args.kwargs["identity_header"] == "X-User-ID"
    assert handler._user_id == "resolved-user"


async def test_auth_message_cannot_replace_trusted_header_identity():
    """A WebSocket auth message cannot override an upstream-asserted identity."""
    handler, socket, _ = _make_message_handler(identity_header="X-User-ID")
    handler._user_id = "resolved-user"
    msg = WebSocketAuthMessage(
        type=WebSocketMessageType.AUTH_MESSAGE,
        payload=ApiKeyAuthPayload(method="api_key", token="replacement-key"),
    )

    with patch("nat.front_ends.fastapi.message_handler.UserManager.from_auth_payload_with_verification", ) as resolver:
        await handler._process_auth_message(msg)

    resolver.assert_not_called()
    assert handler._user_id == "resolved-user"
    assert socket.send_json.await_args.args[0]["status"] == AuthMessageStatus.ERROR


async def test_context_manager_rejects_disabled_connection_credential_without_restoration():
    """A disabled upgrade credential closes the socket and never attempts state restoration."""
    handler, socket, _ = _make_message_handler(accepted_identity_credentials=["jwt"])
    restore = AsyncMock()
    handler._restore_execution_state = restore

    from nat.runtime.user_manager import IdentityCredentialNotAcceptedError

    with patch(
            "nat.front_ends.fastapi.message_handler.UserManager.extract_user_from_connection_with_verification",
            side_effect=IdentityCredentialNotAcceptedError("Identity credential type 'session_cookie' is not accepted"),
    ):
        await handler.__aenter__()

    restore.assert_not_awaited()
    response = socket.send_json.await_args.args[0]
    assert response["status"] == AuthMessageStatus.ERROR
    socket.close.assert_awaited_once_with(code=1008, reason="Identity credential was rejected")


async def test_context_manager_accepts_verified_connection_jwt_before_restoration():
    """An active verifier result permits identity resolution and owned-state restoration."""
    issuer = "https://identity.example.com"
    jwt_validator = MagicMock()
    jwt_validator.verify = AsyncMock(return_value=MagicMock(active=True))
    handler, socket, _ = _make_message_handler(jwt_validators={issuer: jwt_validator})
    token = _make_jwt({"iss": issuer, "sub": "verified-user"})
    socket.scope = {"headers": [(b"authorization", f"Bearer {token}".encode())]}
    socket.query_params = {"conversation_id": "conversation-a"}
    restore = AsyncMock()
    handler._restore_execution_state = restore

    await handler.__aenter__()

    jwt_validator.verify.assert_awaited_once_with(token)
    assert handler._user_id is not None
    restore.assert_awaited_once()


async def test_context_manager_rejects_unverified_connection_jwt_without_restoration():
    """An inactive verifier result cannot establish identity or restore conversation state."""
    issuer = "https://identity.example.com"
    jwt_validator = MagicMock()
    jwt_validator.verify = AsyncMock(return_value=MagicMock(active=False))
    handler, socket, _ = _make_message_handler(jwt_validators={issuer: jwt_validator})
    token = _make_jwt({"iss": issuer, "sub": "unverified-user"})
    socket.scope = {"headers": [(b"authorization", f"Bearer {token}".encode())]}
    socket.query_params = {"conversation_id": "conversation-a"}
    restore = AsyncMock()
    handler._restore_execution_state = restore

    await handler.__aenter__()

    assert handler._user_id is None
    restore.assert_not_awaited()
    response = socket.send_json.await_args.args[0]
    assert response["status"] == AuthMessageStatus.ERROR
    socket.close.assert_awaited_once_with(code=1008, reason="Identity credential was rejected")


async def test_anonymous_connection_does_not_attempt_conversation_lookup():
    """A conversation ID cannot restore state without a resolved user identity."""
    handler, socket, _ = _make_message_handler()
    socket.query_params = {"conversation_id": "conversation-a"}

    await handler._restore_execution_state()

    handler._worker.get_conversation_handler.assert_not_called()


async def test_successful_auth_message_attempts_owned_restoration_once():
    """Delayed authentication can restore once and cannot retry the lookup."""
    handler, socket, _ = _make_message_handler()
    socket.query_params = {"conversation_id": "conversation-a"}
    handler._worker.get_conversation_handler.return_value = None
    user_info = MagicMock()
    user_info.get_user_id.return_value = "user-a"
    msg = WebSocketAuthMessage(
        type=WebSocketMessageType.AUTH_MESSAGE,
        payload=ApiKeyAuthPayload(method="api_key", token="test-api-key"),
    )

    with patch(
            "nat.front_ends.fastapi.message_handler.UserManager.from_auth_payload_with_verification",
            return_value=user_info,
    ):
        await handler._process_auth_message(msg)
        await handler._process_auth_message(msg)

    handler._worker.get_conversation_handler.assert_called_once_with("user-a", "conversation-a")


async def test_failed_auth_message_does_not_attempt_restoration():
    """A failed identity resolution cannot trigger conversation restoration."""
    handler, socket, _ = _make_message_handler()
    restore = AsyncMock()
    handler._restore_execution_state = restore
    msg = WebSocketAuthMessage(
        type=WebSocketMessageType.AUTH_MESSAGE,
        payload=ApiKeyAuthPayload(method="api_key", token="test-api-key"),
    )

    with patch(
            "nat.front_ends.fastapi.message_handler.UserManager.from_auth_payload_with_verification",
            side_effect=ValueError("invalid credential"),
    ):
        await handler._process_auth_message(msg)

    restore.assert_not_awaited()
    response = socket.send_json.await_args.args[0]
    assert response["status"] == AuthMessageStatus.ERROR


async def test_disabled_auth_message_preserves_existing_identity_and_does_not_restore():
    """A disabled auth message cannot replace identity or trigger restoration."""
    handler, socket, _ = _make_message_handler(accepted_identity_credentials=["jwt"])
    handler._user_id = "existing-user"
    restore = AsyncMock()
    handler._restore_execution_state = restore
    msg = WebSocketAuthMessage(
        type=WebSocketMessageType.AUTH_MESSAGE,
        payload=ApiKeyAuthPayload(method="api_key", token="test-api-key"),
    )

    await handler._process_auth_message(msg)

    assert handler._user_id == "existing-user"
    restore.assert_not_awaited()
    response = socket.send_json.await_args.args[0]
    assert response["status"] == AuthMessageStatus.ERROR
    assert response["payload"]["details"] == "Identity credential type 'api_key' is not accepted"


async def test_rejected_auth_message_jwt_preserves_identity_and_does_not_restore():
    """A rejected JWT auth message cannot replace identity or restore conversation state."""
    issuer = "https://identity.example.com"
    jwt_validator = MagicMock()
    jwt_validator.verify = AsyncMock(return_value=MagicMock(active=False))
    handler, socket, _ = _make_message_handler(jwt_validators={issuer: jwt_validator})
    handler._user_id = "existing-user"
    socket.query_params = {"conversation_id": "conversation-a"}
    restore = AsyncMock()
    handler._restore_execution_state = restore
    token = _make_jwt({"iss": issuer, "sub": "unverified-user"})
    msg = WebSocketAuthMessage(
        type=WebSocketMessageType.AUTH_MESSAGE,
        payload=JwtAuthPayload(method="jwt", token=token),
    )

    await handler._process_auth_message(msg)

    assert handler._user_id == "existing-user"
    restore.assert_not_awaited()
    response = socket.send_json.await_args.args[0]
    assert response["status"] == AuthMessageStatus.ERROR
    assert response["payload"]["details"] == "JWT verification failed"


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


async def test_run_processes_mode_frame_while_preflight_pending():
    """The receive loop processes an oauth_mode_preference frame while preflight auth is still in flight.

    Regression test for the bug where ``run()`` awaited ``_run_preflight_auth()`` BEFORE starting the
    receive loop. Because preflight blocks on the user's OAuth login, the ``oauth_mode_preference`` frame
    sent by the UI on connect was never read during preflight, so the in-flight FlowState kept its default
    REDIRECT mode. With preflight run concurrently, the frame is processed and applied to the pending flow.

    Preflight here stays pending until the test releases it: the old preflight-first ``run()`` would
    never reach ``receive_json`` to process the frame, so ``asyncio.wait_for`` below would time out,
    surfacing the regression as a failure.
    """
    handler, socket, flow_handler = _make_message_handler()

    preflight_started = asyncio.Event()
    release_preflight = asyncio.Event()

    async def _pending_preflight() -> None:
        # Simulate an in-flight preflight OAuth flow awaiting user login until the test releases it.
        flow_handler._current_flow_state = FlowState()
        preflight_started.set()
        await release_preflight.wait()

    handler._run_preflight_auth = _pending_preflight  # type: ignore[method-assign]

    mode_frame = WebSocketAuthMessage(
        type=WebSocketMessageType.AUTH_MESSAGE,
        payload=OAuthModePreferencePayload(method="oauth_mode_preference", mode="popup"),
    ).model_dump()

    frames = iter([mode_frame])

    async def _receive_json() -> dict:
        # Deliver the mode frame only after preflight has begun (its flow is in flight),
        # then release preflight and disconnect to end the loop. If preflight ran first-and-blocking
        # (old code), this coroutine would never be reached and run() would hang until wait_for times out.
        await preflight_started.wait()
        try:
            return next(frames)
        except StopIteration:
            release_preflight.set()
            raise WebSocketDisconnect() from None

    socket.receive_json = _receive_json

    await asyncio.wait_for(handler.run(), timeout=5)

    # The in-flight preflight flow received the popup mode while preflight was still pending.
    assert preflight_started.is_set()
    assert flow_handler._oauth_mode is OAuthMode.POPUP
    assert flow_handler._current_flow_state is not None
    assert flow_handler._current_flow_state.oauth_mode is OAuthMode.POPUP


async def test_run_does_not_cancel_inflight_flow_on_disconnect():
    """An in-flight preflight OAuth flow must survive a WebSocket disconnect.

    Regression test for "Invalid state" on redirect-mode login: in redirect mode the browser
    navigates the tab to the OAuth provider, which closes this WebSocket as a normal step of the
    flow. The flow completes out-of-band via the ``/auth/redirect`` HTTP callback. ``run()`` must
    NOT cancel the in-flight flow when the receive loop exits, otherwise the flow's ``finally``
    removes its ``state`` from ``_outstanding_flows`` before the callback arrives and the callback
    returns "Invalid state. Please restart the authentication process."
    """
    handler, socket, flow_handler = _make_message_handler()

    flow_future: asyncio.Future = asyncio.get_running_loop().create_future()
    cleanup_ran = asyncio.Event()
    disconnected = asyncio.Event()

    async def _redirect_preflight() -> None:
        # Simulate an in-flight OAuth flow: register a current flow, then await the future that the
        # /auth/redirect callback resolves. The finally mirrors the real flow's _remove_flow_cb(state).
        flow_handler._current_flow_state = FlowState()
        try:
            await flow_future
        finally:
            cleanup_ran.set()

    handler._run_preflight_auth = _redirect_preflight  # type: ignore[method-assign]

    async def _receive_json() -> dict:
        # Redirect mode: the browser navigates away immediately, closing the socket.
        disconnected.set()
        raise WebSocketDisconnect()

    socket.receive_json = _receive_json

    run_task = asyncio.create_task(handler.run())
    await asyncio.wait_for(disconnected.wait(), timeout=1)
    # Give run()'s finally a couple of event-loop turns to (incorrectly) cancel, if it would.
    await asyncio.sleep(0.05)

    # The disconnect must NOT have cancelled the flow or triggered its cleanup, and run() must still
    # be waiting for the flow to complete via the HTTP callback.
    assert not flow_future.cancelled()
    assert not cleanup_ran.is_set()
    assert not run_task.done()

    # The provider redirect finally arrives and resolves the flow; only now does it clean up.
    flow_future.set_result("token")
    await asyncio.wait_for(run_task, timeout=1)
    assert cleanup_ran.is_set()
