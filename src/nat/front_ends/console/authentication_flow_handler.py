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
import webbrowser
from dataclasses import dataclass
from dataclasses import field

import click
import httpx
import pkce
from authlib.common.errors import AuthlibBaseError as OAuthError
from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import FastAPI
from fastapi import Request

from nat.authentication.interfaces import FlowHandlerBase
from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.front_ends.fastapi.fastapi_front_end_controller import _FastApiFrontEndController

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class _FlowState:
    future: asyncio.Future = field(default_factory=asyncio.Future, init=False)
    challenge: str | None = None
    verifier: str | None = None
    token_url: str | None = None
    use_pkce: bool | None = None


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
        try:
            client = AsyncOAuth2Client(
                client_id=cfg.client_id,
                client_secret=cfg.client_secret.get_secret_value(),
                redirect_uri=cfg.redirect_uri,
                scope=" ".join(cfg.scopes) if cfg.scopes else None,
                token_endpoint=cfg.token_url,
                token_endpoint_auth_method=cfg.token_endpoint_auth_method,
                code_challenge_method="S256" if cfg.use_pkce else None,
            )
            self._oauth_client = client
            return client
        except (OAuthError, ValueError, TypeError) as e:
            raise RuntimeError(f"Invalid OAuth2 configuration: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to create OAuth2 client: {e}") from e

    def _create_authorization_url(self,
                                  client: AsyncOAuth2Client,
                                  config: OAuth2AuthCodeFlowProviderConfig,
                                  state: str,
                                  verifier: str | None = None,
                                  challenge: str | None = None) -> str:
        """
        Create OAuth authorization URL with proper error handling.

        Args:
            client: The OAuth2 client instance
            config: OAuth2 configuration
            state: OAuth state parameter
            verifier: PKCE verifier (if using PKCE)
            challenge: PKCE challenge (if using PKCE)

        Returns:
            The authorization URL
        """
        try:
            auth_url, _ = client.create_authorization_url(
                config.authorization_url,
                state=state,
                code_verifier=verifier if config.use_pkce else None,
                code_challenge=challenge if config.use_pkce else None,
                **(config.authorization_kwargs or {})
            )
            return auth_url
        except (OAuthError, ValueError, TypeError) as e:
            raise RuntimeError(f"Error creating OAuth authorization URL: {e}") from e

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

        flow_state.token_url = cfg.token_url
        flow_state.use_pkce = cfg.use_pkce

        # PKCE bits
        if cfg.use_pkce:
            verifier, challenge = pkce.generate_pkce_pair()
            flow_state.verifier = verifier
            flow_state.challenge = challenge

        # Create authorization URL using helper function
        auth_url = self._create_authorization_url(client=client,
                                                  config=cfg,
                                                  state=state,
                                                  verifier=flow_state.verifier,
                                                  challenge=flow_state.challenge)

        # Register flow + maybe spin up redirect handler
        async with self._server_lock:
            if (not self._redirect_app):
                self._redirect_app = await self._build_redirect_app()

            await self._start_redirect_server()

            self._flows[state] = flow_state
            self._active_flows += 1

        try:
            webbrowser.open(auth_url)
            click.echo("Your browser has been opened for authentication.")
        except Exception as e:
            logger.error("Browser open failed: %s", e)
            raise RuntimeError(f"Browser open failed: {e}") from e

        # Wait for the redirect to land
        try:
            token = await asyncio.wait_for(flow_state.future, timeout=300)
        except TimeoutError as exc:
            raise RuntimeError("Authentication timed out (5 min).") from exc
        finally:
            async with self._server_lock:
                self._flows.pop(state, None)
                self._active_flows -= 1

                if self._active_flows == 0:
                    await self._stop_redirect_server()

        return AuthenticatedContext(
            headers={"Authorization": f"Bearer {token['access_token']}"},
            metadata={
                "expires_at": token.get("expires_at"), "raw_token": token
            },
        )

    # --------------- redirect server / in‑process app -------------------- #
    async def _build_redirect_app(self) -> FastAPI:
        """
        * If cfg.run_redirect_local_server == True → start a local server.
        * Else → only build the redirect app and save it to `self._redirect_app`
                 for in‑process testing.
        """
        app = FastAPI()

        @app.get("/auth/redirect")
        async def handle_redirect(request: Request):
            state = request.query_params.get("state")
            if not state or state not in self._flows:
                return "Invalid state; restart authentication."
            flow_state = self._flows[state]
            try:
                token = await self._oauth_client.fetch_token(  # type: ignore[arg-type]
                    url=flow_state.token_url,
                    authorization_response=str(request.url),
                    code_verifier=flow_state.verifier if flow_state.use_pkce else None,
                    state=state,
                )
                flow_state.future.set_result(token)
            except OAuthError as e:
                flow_state.future.set_exception(
                    RuntimeError(f"Authorization server rejected request: {e.error} ({e.description})"))
                return "Authentication failed: Authorization server rejected the request. You may close this tab."
            except httpx.HTTPError as e:
                flow_state.future.set_exception(RuntimeError(f"Network error during token fetch: {e}"))
                return "Authentication failed: Network error occurred. You may close this tab."
            except Exception as e:
                flow_state.future.set_exception(RuntimeError(f"Authentication failed: {e}"))
                return "Authentication failed: An unexpected error occurred. You may close this tab."
            return "Authentication successful – you may close this tab."

        return app

    async def _start_redirect_server(self) -> None:
        # If the server is already running, do nothing
        if self._server_controller:
            return
        try:
            if not self._redirect_app:
                raise RuntimeError("Redirect app not built.")

            self._server_controller = _FastApiFrontEndController(self._redirect_app)

            asyncio.create_task(self._server_controller.start_server(host="localhost", port=8000))

            # Give the server a moment to bind sockets before we return
            await asyncio.sleep(0.3)
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
        In test mode (run_redirect_local_server=False) the in‑memory redirect
        app is exposed for testing purposes.
        """
        return self._redirect_app
