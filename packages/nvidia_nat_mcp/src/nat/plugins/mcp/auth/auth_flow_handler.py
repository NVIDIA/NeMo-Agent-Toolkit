# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import webbrowser

import click
import pkce
from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import FastAPI

from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.front_ends.console.authentication_flow_handler import ConsoleAuthenticationFlowHandler
from nat.front_ends.console.authentication_flow_handler import _FlowState
from nat.front_ends.fastapi.fastapi_front_end_controller import _FastApiFrontEndController


class MCPAuthenticationFlowHandler(ConsoleAuthenticationFlowHandler):
    """
    Authentication helper for MCP environments.
    A separate flow handler is needed because MCP tool discovery happens before the default auth_callback is available.
    It is used to handle the OAuth2 authorization code flow only during startup.
    """

    def __init__(self):
        super().__init__()
        self._server_controller: _FastApiFrontEndController | None = None
        self._redirect_app: FastAPI | None = None
        self._server_lock = asyncio.Lock()
        self._oauth_client: AsyncOAuth2Client | None = None

    async def authenticate(self, config: AuthProviderBaseConfig, method: AuthFlowType) -> AuthenticatedContext:
        """
        Handle the OAuth2 authorization code flow.
        """
        if method == AuthFlowType.OAUTH2_AUTHORIZATION_CODE:
            if (not isinstance(config, OAuth2AuthCodeFlowProviderConfig)):
                raise ValueError("Requested OAuth2 Authorization Code Flow but passed invalid config")

            return await self._handle_oauth2_auth_code_flow(config)

        raise NotImplementedError(f'Auth method “{method}” not supported')

    async def _handle_oauth2_auth_code_flow(self, cfg: OAuth2AuthCodeFlowProviderConfig) -> AuthenticatedContext:
        state = secrets.token_urlsafe(16)
        flow_state = _FlowState()
        client = self.construct_oauth_client(cfg)

        flow_state.token_url = cfg.token_url
        flow_state.use_pkce = cfg.use_pkce

        # PKCE bits
        if cfg.use_pkce:
            verifier, challenge = pkce.generate_pkce_pair()
            flow_state.verifier = verifier
            flow_state.challenge = challenge

        auth_url, _ = client.create_authorization_url(
            cfg.authorization_url,
            state=state,
            code_verifier=flow_state.verifier if cfg.use_pkce else None,
            code_challenge=flow_state.challenge if cfg.use_pkce else None,
            **(cfg.authorization_kwargs or {})
        )

        async with self._server_lock:
            if self._redirect_app is None:
                self._redirect_app = await self._build_redirect_app()

            await self._start_redirect_server()
            self._flows[state] = flow_state

        click.echo("Your browser has been opened for authentication.")
        webbrowser.open(auth_url)

        try:
            token = await asyncio.wait_for(flow_state.future, timeout=300)
        except asyncio.TimeoutError:
            raise RuntimeError("Authentication timed out (5 min).")
        finally:
            async with self._server_lock:
                self._flows.pop(state, None)
                await self._stop_redirect_server()

        return AuthenticatedContext(
            headers={"Authorization": f"Bearer {token['access_token']}"},
            metadata={
                "expires_at": token.get("expires_at"), "raw_token": token
            },
        )
