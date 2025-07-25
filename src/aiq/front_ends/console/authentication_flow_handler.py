import asyncio
import secrets
import webbrowser
from dataclasses import dataclass
from dataclasses import field

import click
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


class ConsoleAuthenticationFlowHandler(FlowHandlerBase):
    _flows: dict[str, _FlowState] = {}
    _server_controller: _FastApiFrontEndController | None = None
    _server_lock: asyncio.Lock = asyncio.Lock()
    _active_flows: int = 0

    @staticmethod
    async def authenticate(config: OAuth2AuthorizationCodeFlowConfig, method: AuthFlowType) -> AuthenticatedContext:
        if method == AuthFlowType.HTTP_BASIC:
            return ConsoleAuthenticationFlowHandler._handle_http_basic()
        elif method == AuthFlowType.OAUTH2_AUTHORIZATION_CODE:
            return await ConsoleAuthenticationFlowHandler._handle_oauth2_auth_code_flow(config)

        raise NotImplementedError(f"Authentication method '{method}' is not supported by the console frontend.")

    @staticmethod
    def _handle_http_basic() -> AuthenticatedContext:
        username = click.prompt("Username", type=str)
        password = click.prompt("Password", type=str, hide_input=True)

        return AuthenticatedContext(headers={"Authorization": f"Bearer {username}:{password}"},
                                    metadata={
                                        "username": username, "password": password
                                    })

    @staticmethod
    async def _handle_oauth2_auth_code_flow(config: OAuth2AuthorizationCodeFlowConfig) -> AuthenticatedContext:
        state = secrets.token_urlsafe(16)
        flow_state = _FlowState()

        client = AsyncOAuth2Client(
            client_id=config.client_id,
            client_secret=config.client_secret,
            redirect_uri=config.redirect_uri,
            scope=" ".join(config.scope) if config.scope else None,
            token_endpoint=config.token_url,
        )

        authorization_url, _ = client.create_authorization_url(config.authorization_url, state=state)

        async with ConsoleAuthenticationFlowHandler._server_lock:
            if ConsoleAuthenticationFlowHandler._server_controller is None:
                await ConsoleAuthenticationFlowHandler._start_redirect_server(config)
            ConsoleAuthenticationFlowHandler._flows[state] = flow_state
            ConsoleAuthenticationFlowHandler._active_flows += 1

        click.echo("Your browser has been opened to complete the authentication.")
        webbrowser.open(authorization_url)

        try:
            await asyncio.wait_for(flow_state.event.wait(), timeout=300)
        except asyncio.TimeoutError:
            raise RuntimeError("Authentication flow timed out after 5 minutes.")
        finally:
            async with ConsoleAuthenticationFlowHandler._server_lock:
                if state in ConsoleAuthenticationFlowHandler._flows:
                    del ConsoleAuthenticationFlowHandler._flows[state]
                ConsoleAuthenticationFlowHandler._active_flows -= 1
                if ConsoleAuthenticationFlowHandler._active_flows == 0:
                    await ConsoleAuthenticationFlowHandler._stop_redirect_server()

        if flow_state.error:
            raise RuntimeError(f"Authentication failed: {flow_state.error}") from flow_state.error
        if not flow_state.token:
            raise RuntimeError("Authentication failed: Did not receive token.")

        token = flow_state.token

        return AuthenticatedContext(headers={"Authorization": f"Bearer {token['access_token']}"},
                                    metadata={
                                        "expires_at": token.get("expires_at"), "raw_token": token
                                    })

    @staticmethod
    async def _start_redirect_server(config: OAuth2AuthorizationCodeFlowConfig) -> None:
        app = FastAPI()

        @app.get(config.redirect_path)
        async def handle_redirect(request: Request):
            state = request.query_params.get("state")
            if not state or state not in ConsoleAuthenticationFlowHandler._flows:
                return "Invalid state. Please restart the authentication process."

            flow_state = ConsoleAuthenticationFlowHandler._flows[state]

            client = AsyncOAuth2Client(
                client_id=config.client_id,
                client_secret=config.client_secret,
                redirect_uri=config.redirect_uri,
                scope=" ".join(config.scope) if config.scope else None,
                token_endpoint=config.token_url,
            )
            try:
                flow_state.token = await client.fetch_token(url=config.token_url,
                                                            authorization_response=str(request.url))
            except Exception as e:
                flow_state.error = e
            finally:
                flow_state.event.set()
            return "Authentication successful! You can close this window."

        controller = _FastApiFrontEndController(app)
        ConsoleAuthenticationFlowHandler._server_controller = controller

        asyncio.create_task(controller.start_server(host=config.client_server_host, port=config.client_server_port))
        await asyncio.sleep(1)

    @staticmethod
    async def _stop_redirect_server():
        if ConsoleAuthenticationFlowHandler._server_controller:
            await ConsoleAuthenticationFlowHandler._server_controller.stop_server()
            ConsoleAuthenticationFlowHandler._server_controller = None
