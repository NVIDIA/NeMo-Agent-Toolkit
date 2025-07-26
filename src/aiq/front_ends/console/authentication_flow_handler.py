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
import webbrowser
from dataclasses import dataclass
from dataclasses import field

import click
import pkce
from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import FastAPI
from fastapi import Request

from aiq.authentication.interfaces import FlowHandlerBase
from aiq.authentication.oauth2.authorization_code_flow_config import OAuth2AuthorizationCodeFlowConfig
from aiq.data_models.authentication import AuthenticatedContext
from aiq.data_models.authentication import AuthFlowType
from aiq.front_ends.fastapi.fastapi_front_end_controller import _FastApiFrontEndController


@dataclass
class _FlowState:
    event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    token: dict | None = None
    error: Exception | None = None
    challenge: str | None = None
    verifier: str | None = None


class ConsoleAuthenticationFlowHandler(FlowHandlerBase):

    def __init__(self):
        super().__init__()
        self._server_controller: _FastApiFrontEndController | None = None
        self._flows: dict[str, _FlowState] = {}
        self._active_flows = 0
        self._server_lock: asyncio.Lock = asyncio.Lock()
        self._oauth_client = None

    async def authenticate(self, config: OAuth2AuthorizationCodeFlowConfig,
                           method: AuthFlowType) -> AuthenticatedContext:
        if method == AuthFlowType.HTTP_BASIC:
            return ConsoleAuthenticationFlowHandler._handle_http_basic()
        elif method == AuthFlowType.OAUTH2_AUTHORIZATION_CODE:
            return await self._handle_oauth2_auth_code_flow(config)

        raise NotImplementedError(f"Authentication method '{method}' is not supported by the console frontend.")

    def construct_oauth_client(self, config: OAuth2AuthorizationCodeFlowConfig) -> AsyncOAuth2Client:
        client = AsyncOAuth2Client(client_id=config.client_id,
                                   client_secret=config.client_secret,
                                   redirect_uri=config.redirect_uri,
                                   scope=" ".join(config.scopes) if config.scopes else None,
                                   token_endpoint=config.token_url,
                                   token_endpoint_auth_method=config.token_endpoint_auth_method,
                                   code_challenge_method='S256' if config.use_pkce else None)
        self._oauth_client = client
        return client

    @staticmethod
    def _handle_http_basic() -> AuthenticatedContext:
        username = click.prompt("Username", type=str)
        password = click.prompt("Password", type=str, hide_input=True)

        return AuthenticatedContext(headers={"Authorization": f"Bearer {username}:{password}"},
                                    metadata={
                                        "username": username, "password": password
                                    })

    async def _handle_oauth2_auth_code_flow(self, config: OAuth2AuthorizationCodeFlowConfig) -> AuthenticatedContext:
        state = secrets.token_urlsafe(16)
        flow_state = _FlowState()

        client = self.construct_oauth_client(config)

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
            if self._server_controller is None:
                await self._start_redirect_server(config)
            self._flows[state] = flow_state
            self._active_flows += 1

        click.echo("Your browser has been opened to complete the authentication.")
        webbrowser.open(authorization_url)

        try:
            await asyncio.wait_for(flow_state.event.wait(), timeout=300)
        except asyncio.TimeoutError:
            raise RuntimeError("Authentication flow timed out after 5 minutes.")
        finally:
            async with self._server_lock:
                if state in self._flows:
                    del self._flows[state]
                self._active_flows -= 1
                if self._active_flows == 0:
                    await self._stop_redirect_server()

        if flow_state.error:
            raise RuntimeError(f"Authentication failed: {flow_state.error}") from flow_state.error
        if not flow_state.token:
            raise RuntimeError("Authentication failed: Did not receive token.")

        token = flow_state.token

        return AuthenticatedContext(headers={"Authorization": f"Bearer {token['access_token']}"},
                                    metadata={
                                        "expires_at": token.get("expires_at"), "raw_token": token
                                    })

    async def _start_redirect_server(self, config: OAuth2AuthorizationCodeFlowConfig) -> None:
        app = FastAPI()

        @app.get(config.redirect_path)
        async def handle_redirect(request: Request):
            state = request.query_params.get("state")
            if not state or state not in self._flows:
                return "Invalid state. Please restart the authentication process."

            flow_state = self._flows[state]
            verifier = flow_state.verifier

            try:
                flow_state.token = await self._oauth_client.fetch_token(
                    url=config.token_url,
                    authorization_response=str(request.url),
                    code_verifier=verifier if config.use_pkce else None,
                    state=state)
            except Exception as e:
                flow_state.error = e
            finally:
                flow_state.event.set()
            return "Authentication successful! You can close this window."

        controller = _FastApiFrontEndController(app)
        self._server_controller = controller

        asyncio.create_task(controller.start_server(host=config.client_server_host, port=config.client_server_port))
        await asyncio.sleep(1)

    async def _stop_redirect_server(self):
        if self._server_controller:
            await self._server_controller.stop_server()
            self._server_controller = None
