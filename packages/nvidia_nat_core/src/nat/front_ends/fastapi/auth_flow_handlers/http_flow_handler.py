# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import contextvars
import logging
import secrets
import typing
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass

import pkce
from authlib.common.errors import AuthlibBaseError as OAuthError
from authlib.integrations.httpx_client import AsyncOAuth2Client

from nat.authentication.interfaces import FlowHandlerBase
from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.data_models.api_server import ResponseSerializable
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.interactive_http import StreamOAuthEvent

if typing.TYPE_CHECKING:
    from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import FlowState
    from nat.front_ends.fastapi.execution_store import ExecutionStore

logger = logging.getLogger(__name__)


@dataclass
class _OAuthExecutionContext:
    """Per-execution context for OAuth, stored in a task-local context var."""
    execution_id: str
    store: "ExecutionStore"
    stream_queue: asyncio.Queue[ResponseSerializable | None] | None = None


# Task-local context var so concurrent executions don't race.
_oauth_execution_ctx: contextvars.ContextVar[_OAuthExecutionContext | None] = contextvars.ContextVar(
    "_oauth_execution_ctx", default=None)


class HTTPAuthenticationFlowHandler(FlowHandlerBase):
    """
    HTTP-based authentication flow handler.

    When an execution context is set (via :meth:`set_execution_context`), the
    handler supports the OAuth2 Authorization Code flow by:

    1. Creating the OAuth client and authorization URL.
    2. Registering the flow with the worker's ``_add_flow`` / ``_remove_flow``
       callbacks (same ``FlowState`` as the WebSocket handler).
    3. Publishing ``oauth_required`` to the execution store (and optionally
       pushing a :class:`StreamOAuthEvent` onto a stream queue).
    4. Awaiting ``flow_state.future`` – the background task blocks here until
       the existing ``redirect_uri`` endpoint resolves the future.

    Without an execution context the handler falls back to raising
    ``NotImplementedError`` (preserving existing behaviour).

    The execution context is stored in a :mod:`contextvars` variable so
    concurrent executions sharing the same handler instance do not race.
    """

    def __init__(
        self,
        add_flow_cb: Callable[[str, "FlowState"], Awaitable[None]] | None = None,
        remove_flow_cb: Callable[[str], Awaitable[None]] | None = None,
        auth_timeout_seconds: float = 300.0,
    ) -> None:
        self._add_flow_cb = add_flow_cb
        self._remove_flow_cb = remove_flow_cb
        self._auth_timeout_seconds = auth_timeout_seconds

    # ------------------------------------------------------------------
    # Execution context management (called per-request by the runner)
    # ------------------------------------------------------------------

    @staticmethod
    def set_execution_context(
        execution_id: str,
        store: "ExecutionStore",
        stream_queue: asyncio.Queue[ResponseSerializable | None] | None = None,
    ) -> None:
        """Attach the current execution context so ``authenticate`` can coordinate.

        Uses a :class:`contextvars.ContextVar` so each ``asyncio.Task``
        (i.e. each execution) has its own isolated context.
        """
        _oauth_execution_ctx.set(
            _OAuthExecutionContext(execution_id=execution_id, store=store, stream_queue=stream_queue))

    @staticmethod
    def clear_execution_context() -> None:
        _oauth_execution_ctx.set(None)

    # ------------------------------------------------------------------
    # FlowHandlerBase implementation
    # ------------------------------------------------------------------

    async def authenticate(self, config: AuthProviderBaseConfig, method: AuthFlowType) -> AuthenticatedContext:
        ctx = _oauth_execution_ctx.get()
        # If we have an execution context and the right flow callbacks,
        # handle OAuth2 authorization code.
        if (ctx is not None and self._add_flow_cb is not None and self._remove_flow_cb is not None
                and method == AuthFlowType.OAUTH2_AUTHORIZATION_CODE):
            return await self._handle_oauth2_auth_code_flow(config, ctx)  # type: ignore[arg-type]

        raise NotImplementedError(f"Authentication method '{method}' is not supported by the HTTP frontend."
                                  f" Do you have WebSockets enabled or HTTP interactive mode active?")

    # ------------------------------------------------------------------
    # OAuth2 Authorization Code flow (mirrors WebSocket handler)
    # ------------------------------------------------------------------

    def _create_oauth_client(self, config: OAuth2AuthCodeFlowProviderConfig) -> AsyncOAuth2Client:
        try:
            return AsyncOAuth2Client(
                client_id=config.client_id,
                client_secret=config.client_secret.get_secret_value(),
                redirect_uri=config.redirect_uri,
                scope=" ".join(config.scopes) if config.scopes else None,
                token_endpoint=config.token_url,
                code_challenge_method="S256" if config.use_pkce else None,
                token_endpoint_auth_method=config.token_endpoint_auth_method,
            )
        except (OAuthError, ValueError, TypeError) as e:
            raise RuntimeError(f"Invalid OAuth2 configuration: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to create OAuth2 client: {e}") from e

    def _create_authorization_url(
        self,
        client: AsyncOAuth2Client,
        config: OAuth2AuthCodeFlowProviderConfig,
        state: str,
        verifier: str | None = None,
        challenge: str | None = None,
    ) -> str:
        try:
            authorization_url, _ = client.create_authorization_url(
                config.authorization_url,
                state=state,
                code_verifier=verifier if config.use_pkce else None,
                code_challenge=challenge if config.use_pkce else None,
                **(config.authorization_kwargs or {}),
            )
            return authorization_url
        except (OAuthError, ValueError, TypeError) as e:
            raise RuntimeError(f"Error creating OAuth authorization URL: {e}") from e

    async def _handle_oauth2_auth_code_flow(
        self,
        config: OAuth2AuthCodeFlowProviderConfig,
        ctx: _OAuthExecutionContext,
    ) -> AuthenticatedContext:
        from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import FlowState

        state = secrets.token_urlsafe(16)
        flow_state = FlowState(config=config)
        flow_state.client = self._create_oauth_client(config)

        if config.use_pkce:
            verifier, challenge = pkce.generate_pkce_pair()
            flow_state.verifier = verifier
            flow_state.challenge = challenge

        authorization_url = self._create_authorization_url(
            client=flow_state.client,
            config=config,
            state=state,
            verifier=flow_state.verifier,
            challenge=flow_state.challenge,
        )

        assert self._add_flow_cb is not None
        assert self._remove_flow_cb is not None

        # Register the flow so the redirect_uri endpoint can complete it
        await self._add_flow_cb(state, flow_state)

        try:
            # Publish to execution store
            await ctx.store.set_oauth_required(
                execution_id=ctx.execution_id,
                auth_url=authorization_url,
                oauth_state=state,
            )

            # If streaming, push an SSE event
            if ctx.stream_queue is not None:
                event = StreamOAuthEvent(
                    execution_id=ctx.execution_id,
                    auth_url=authorization_url,
                    oauth_state=state,
                )
                await ctx.stream_queue.put(event)

            # Block until the redirect_uri endpoint resolves the token
            token = await asyncio.wait_for(flow_state.future, timeout=self._auth_timeout_seconds)
        except TimeoutError as exc:
            raise RuntimeError(f"Authentication flow timed out after {self._auth_timeout_seconds} seconds.") from exc
        finally:
            await self._remove_flow_cb(state)

        # Transition back to running
        await ctx.store.set_running(ctx.execution_id)

        return AuthenticatedContext(
            headers={"Authorization": f"Bearer {token['access_token']}"},
            metadata={
                "expires_at": token.get("expires_at"), "raw_token": token
            },
        )
