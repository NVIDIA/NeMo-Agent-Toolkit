# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
import secrets
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field

import pkce
from authlib.integrations.httpx_client import AsyncOAuth2Client

from aiq.authentication.interfaces import FlowHandlerBase
from aiq.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from aiq.data_models.authentication import AuthenticatedContext
from aiq.data_models.authentication import AuthFlowType
from aiq.data_models.interactive import _HumanPromptOAuthConsent
from aiq.front_ends.fastapi.message_handler import WebSocketMessageHandler

logger = logging.getLogger(__name__)


@dataclass
class FlowState:
    future: asyncio.Future = field(default_factory=asyncio.Future, init=False)
    challenge: str | None = None
    verifier: str | None = None
    client: AsyncOAuth2Client | None = None
    config: OAuth2AuthCodeFlowProviderConfig | None = None


class WebSocketAuthenticationFlowHandler(FlowHandlerBase):

    def __init__(self,
                 add_flow_cb: Callable[[str, FlowState], Awaitable[None]],
                 remove_flow_cb: Callable[[str], Awaitable[None]],
                 web_socket_message_handler: WebSocketMessageHandler):

        self._add_flow_cb: Callable[[str, FlowState], Awaitable[None]] = add_flow_cb
        self._remove_flow_cb: Callable[[str], Awaitable[None]] = remove_flow_cb
        self._web_socket_message_handler: WebSocketMessageHandler = web_socket_message_handler

    async def authenticate(self, config: OAuth2AuthCodeFlowProviderConfig,
                           method: AuthFlowType) -> AuthenticatedContext:
        if method == AuthFlowType.OAUTH2_AUTHORIZATION_CODE:
            return await self._handle_oauth2_auth_code_flow(config)

        raise NotImplementedError(f"Authentication method '{method}' is not supported by the websocket frontend.")

    def create_oauth_client(self, config: OAuth2AuthCodeFlowProviderConfig):
        return AsyncOAuth2Client(client_id=config.client_id,
                                 client_secret=config.client_secret,
                                 redirect_uri=config.redirect_uri,
                                 scope=" ".join(config.scopes) if config.scopes else None,
                                 token_endpoint=config.token_url,
                                 code_challenge_method='S256' if config.use_pkce else None,
                                 token_endpoint_auth_method=config.token_endpoint_auth_method)

    async def _handle_oauth2_auth_code_flow(self, config: OAuth2AuthCodeFlowProviderConfig) -> AuthenticatedContext:

        if config.run_local_redirect_server:
            logger.warning("Running a local redirect server is not supported in the WebSocket flow handler. Ignoring "
                           "this setting.")

        state = secrets.token_urlsafe(16)
        flow_state = FlowState(config=config)

        flow_state.client = self.create_oauth_client(config)

        if config.use_pkce:
            verifier, challenge = pkce.generate_pkce_pair()
            flow_state.verifier = verifier
            flow_state.challenge = challenge

        authorization_url, _ = flow_state.client.create_authorization_url(
            config.authorization_url,
            state=state,
            code_verifier=flow_state.verifier if config.use_pkce else None,
            code_challenge=flow_state.challenge if config.use_pkce else None,
            **(config.authorization_kwargs or {})
        )

        await self._add_flow_cb(state, flow_state)
        await self._web_socket_message_handler.create_websocket_message(_HumanPromptOAuthConsent(text=authorization_url)
                                                                        )
        try:
            token = await asyncio.wait_for(flow_state.future, timeout=300)
        except asyncio.TimeoutError:
            raise RuntimeError("Authentication flow timed out after 5 minutes.")
        finally:

            await self._remove_flow_cb(state)

        return AuthenticatedContext(headers={"Authorization": f"Bearer {token['access_token']}"},
                                    metadata={
                                        "expires_at": token.get("expires_at"), "raw_token": token
                                    })
