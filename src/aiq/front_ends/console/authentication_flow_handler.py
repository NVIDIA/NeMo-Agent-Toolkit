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
from aiq.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from aiq.data_models.authentication import AuthenticatedContext
from aiq.data_models.authentication import AuthFlowType
from aiq.data_models.authentication import AuthProviderBaseConfig
from aiq.front_ends.fastapi.fastapi_front_end_controller import _FastApiFrontEndController


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class _FlowState:
    future: asyncio.Future = field(default_factory=asyncio.Future, init=False)
    challenge: str | None = None
    verifier: str | None = None


# --------------------------------------------------------------------------- #
# Main handler                                                                #
# --------------------------------------------------------------------------- #
class ConsoleAuthenticationFlowHandler(FlowHandlerBase):
    """
    Authentication helper for CLI / console environments.  Supports:

      • HTTP Basic (username/password)
      • OAuth 2 Authorization‑Code with optional PKCE
    """

    # ----------------------------- lifecycle ----------------------------- #
    def __init__(self) -> None:
        super().__init__()
        self._server_controller: _FastApiFrontEndController | None = None
        self._redirect_app: FastAPI | None = None  # ★ NEW
        self._flows: dict[str, _FlowState] = {}
        self._active_flows = 0
        self._server_lock = asyncio.Lock()
        self._oauth_client: AsyncOAuth2Client | None = None

    # ----------------------------- public API ---------------------------- #
    async def authenticate(
        self,
        config: AuthProviderBaseConfig,
        method: AuthFlowType,
    ) -> AuthenticatedContext:
        if method == AuthFlowType.HTTP_BASIC:
            return self._handle_http_basic()
        if method == AuthFlowType.OAUTH2_AUTHORIZATION_CODE:
            if (not isinstance(config, OAuth2AuthCodeFlowProviderConfig)):
                raise ValueError("Requested OAuth2 Authorization Code Flow but passed invalid config")

            return await self._handle_oauth2_auth_code_flow(config)

        raise NotImplementedError(f"Auth method “{method}” not supported.")

    # --------------------- OAuth2 helper factories ----------------------- #
    def construct_oauth_client(self, cfg: OAuth2AuthCodeFlowProviderConfig) -> AsyncOAuth2Client:
        """
        Separated for easy overriding in tests (to inject ASGITransport).
        """
        client = AsyncOAuth2Client(
            client_id=cfg.client_id,
            client_secret=cfg.client_secret,
            redirect_uri=cfg.redirect_uri,
            scope=" ".join(cfg.scopes) if cfg.scopes else None,
            token_endpoint=cfg.token_url,
            token_endpoint_auth_method=cfg.token_endpoint_auth_method,
            code_challenge_method="S256" if cfg.use_pkce else None,
        )
        self._oauth_client = client
        return client

    # --------------------------- HTTP Basic ------------------------------ #
    @staticmethod
    def _handle_http_basic() -> AuthenticatedContext:
        username = click.prompt("Username", type=str)
        password = click.prompt("Password", type=str, hide_input=True)

        import base64
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("ascii")

        return AuthenticatedContext(
            headers={"Authorization": f"Bearer {encoded_credentials}"},
            metadata={
                "username": username, "password": password
            },
        )

    # --------------------- OAuth2 Authorization‑Code --------------------- #
    async def _handle_oauth2_auth_code_flow(self, cfg: OAuth2AuthCodeFlowProviderConfig) -> AuthenticatedContext:
        state = secrets.token_urlsafe(16)
        flow_state = _FlowState()
        client = self.construct_oauth_client(cfg)

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

        # Register flow + maybe spin up redirect handler
        async with self._server_lock:
            if self._server_controller is None and cfg.run_local_redirect_server:
                await self._start_redirect_server(cfg)
            elif not cfg.run_local_redirect_server and self._redirect_app is None:
                await self._start_redirect_server(cfg)
            self._flows[state] = flow_state
            self._active_flows += 1

        click.echo("Your browser has been opened for authentication.")
        webbrowser.open(auth_url)

        # Wait for the redirect to land
        try:
            token = await asyncio.wait_for(flow_state.future, timeout=300)
        except asyncio.TimeoutError:
            raise RuntimeError("Authentication timed out (5 min).")
        finally:
            async with self._server_lock:
                self._flows.pop(state, None)
                self._active_flows -= 1
                if self._active_flows == 0 and cfg.run_local_redirect_server:
                    await self._stop_redirect_server()
                if self._active_flows == 0 and not cfg.run_local_redirect_server:
                    self._redirect_app = None

        return AuthenticatedContext(
            headers={"Authorization": f"Bearer {token['access_token']}"},
            metadata={
                "expires_at": token.get("expires_at"), "raw_token": token
            },
        )

    # --------------- redirect server / in‑process app -------------------- #
    async def _start_redirect_server(self, cfg: OAuth2AuthCodeFlowProviderConfig) -> None:
        """
        * If cfg.run_redirect_local_server == True → start a uvicorn server (old behaviour).
        * Else → only build the FastAPI app and save it to `self._redirect_app`
                 for in‑process testing with ASGITransport.
        """
        app = FastAPI()

        @app.get(cfg.redirect_path)
        async def handle_redirect(request: Request):
            state = request.query_params.get("state")
            if not state or state not in self._flows:
                return "Invalid state; restart authentication."
            flow_state = self._flows[state]
            try:
                token = await self._oauth_client.fetch_token(  # type: ignore[arg-type]
                    url=cfg.token_url,
                    authorization_response=str(request.url),
                    code_verifier=flow_state.verifier if cfg.use_pkce else None,
                    state=state,
                )
                flow_state.future.set_result(token)
            except Exception as exc:  # noqa: BLE001
                flow_state.future.set_exception(exc)
            return "Authentication successful – you may close this tab."

        # ----------- Option A: in‑process (tests) or external service configured --------------------- #
        if not cfg.run_local_redirect_server:
            # Nothing else to spin up – expose the app for the test.
            self._redirect_app = app
            return

        # ----------- Option B: real uvicorn server -------------------- #
        controller = _FastApiFrontEndController(app)
        self._server_controller = controller
        await self._start_server_task(cfg)
        # Give uvicorn a moment to bind sockets before we return
        await asyncio.sleep(0.3)

    async def _start_server_task(self, cfg: OAuth2AuthCodeFlowProviderConfig) -> None:
        if not self._server_controller:
            raise RuntimeError("Server controller uninitialised.")
        try:
            asyncio.create_task(
                self._server_controller.start_server(host="localhost", port=cfg.local_redirect_server_port))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to start redirect server: {exc}") from exc

    async def _stop_redirect_server(self) -> None:
        if self._server_controller:
            await self._server_controller.stop_server()
            self._server_controller = None

    # ------------------------- test helpers ------------------------------ #
    @property
    def redirect_app(self) -> FastAPI | None:
        """
        In “test‑mode” (run_redirect_local_server=False) the in‑memory FastAPI
        app is exposed so you can mount it on `httpx.ASGITransport`.
        """
        return self._redirect_app
