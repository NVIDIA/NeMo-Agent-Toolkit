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
import secrets
from dataclasses import dataclass
from dataclasses import field

import pkce
from authlib.integrations.httpx_client import AsyncOAuth2Client

from aiq.authentication.interfaces import FlowHandlerBase
from aiq.authentication.oauth2.authorization_code_flow_config import OAuth2AuthorizationCodeFlowConfig
from aiq.data_models.authentication import AuthenticatedContext
from aiq.data_models.authentication import AuthFlowType
from aiq.data_models.interactive import _HumanPromptOAuthConsent
from aiq.front_ends.fastapi.fastapi_front_end_controller import _FastApiFrontEndController


@dataclass
class _FlowState:
    event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    token: dict | None = None
    error: Exception | None = None
    challenge: str | None = None
    verifier: str | None = None


class WebSocketAuthenticationFlowHandler(FlowHandlerBase):
    web_socket = None

    def __init__(self):
        self._flows: dict[str, _FlowState] = {}
        self._configs: dict[str, OAuth2AuthorizationCodeFlowConfig] = {}
        self._server_controller: _FastApiFrontEndController | None = None
        self._server_lock: asyncio.Lock = asyncio.Lock()
        self._active_flows: int = 0
        self._oauth_client = None

    async def authenticate(self, config: OAuth2AuthorizationCodeFlowConfig,
                           method: AuthFlowType) -> AuthenticatedContext:
        if method == AuthFlowType.OAUTH2_AUTHORIZATION_CODE:
            return await self._handle_oauth2_auth_code_flow(config)

        raise NotImplementedError(f"Authentication method '{method}' is not supported by the websocket frontend.")

    def create_oauth_client(self, config: OAuth2AuthorizationCodeFlowConfig):
        if self._oauth_client is None:
            self._oauth_client = AsyncOAuth2Client(client_id=config.client_id,
                                                   client_secret=config.client_secret,
                                                   redirect_uri=config.redirect_uri,
                                                   scope=" ".join(config.scopes) if config.scopes else None,
                                                   token_endpoint=config.token_url,
                                                   code_challenge_method='S256' if config.use_pkce else None)
        return self._oauth_client

    async def _handle_oauth2_auth_code_flow(self, config: OAuth2AuthorizationCodeFlowConfig) -> AuthenticatedContext:
        state = secrets.token_urlsafe(16)
        flow_state = _FlowState()

        client = self.create_oauth_client(config)

        if config.use_pkce:
            verifier, challenge = pkce.generate_pkce_pair()
            flow_state.verifier = verifier
            flow_state.challenge = challenge

        authorization_url, _ = client.create_authorization_url(
            config.authorization_url,
            state=state,
            code_verifier=flow_state.verifier if config.use_pkce else None,
            code_challenge=flow_state.challenge if config.use_pkce else None
        )

        async with self._server_lock:
            self._flows[state] = flow_state
            self._active_flows += 1
            self._configs[state] = config

        if WebSocketAuthenticationFlowHandler.web_socket is None:
            raise RuntimeError("WebSocket instance is not available for handling authentication.")

        await WebSocketAuthenticationFlowHandler.web_socket.message_handler.create_websocket_message(
            _HumanPromptOAuthConsent(text=authorization_url))

        try:
            await asyncio.wait_for(flow_state.event.wait(), timeout=300)
        except asyncio.TimeoutError:
            raise RuntimeError("Authentication flow timed out after 5 minutes.")
        finally:
            async with self._server_lock:
                if state in self._flows:
                    del self._flows[state]
                self._active_flows -= 1

        if flow_state.error:
            raise RuntimeError(f"Authentication failed: {flow_state.error}") from flow_state.error
        if not flow_state.token:
            raise RuntimeError("Authentication failed: Did not receive token.")

        token = flow_state.token

        return AuthenticatedContext(headers={"Authorization": f"Bearer {token['access_token']}"},
                                    metadata={
                                        "expires_at": token.get("expires_at"), "raw_token": token
                                    })
