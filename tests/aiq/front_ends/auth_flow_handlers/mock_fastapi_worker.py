import asyncio

from fastapi import FastAPI
from fastapi import Request

from aiq.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import FlowState


class MockFastAPIWorker:

    def __init__(self):

        self._outstanding_flows: dict[str, FlowState] = {}
        self._outstanding_flows_lock = asyncio.Lock()
        self._app = None

    @property
    def app(self) -> FastAPI:
        """
        Return the FastAPI application instance.
        """
        return self._app

    async def _add_flow(self, state: str, flow_state: FlowState):
        async with self._outstanding_flows_lock:
            self._outstanding_flows[state] = flow_state

    async def _remove_flow(self, state: str):
        async with self._outstanding_flows_lock:
            del self._outstanding_flows[state]

    async def add_authorization_route(self):

        from fastapi.responses import HTMLResponse

        from aiq.front_ends.fastapi.html_snippets.auth_code_grant_success import AUTH_REDIRECT_SUCCESS_HTML

        app = FastAPI()

        async def redirect_uri(request: Request):
            """
            Handle the redirect URI for OAuth2 authentication.
            Args:
                request: The FastAPI request object containing query parameters.

            Returns:
                HTMLResponse: A response indicating the success of the authentication flow.
            """
            state = request.query_params.get("state")

            async with self._outstanding_flows_lock:
                if not state or state not in self._outstanding_flows:
                    return "Invalid state. Please restart the authentication process."

                flow_state = self._outstanding_flows[state]

            config = flow_state.config
            verifier = flow_state.verifier
            client = flow_state.client

            try:
                res = await client.fetch_token(url=config.token_url,
                                               authorization_response=str(request.url),
                                               code_verifier=verifier,
                                               state=state)
                flow_state.future.set_result(res)
            except Exception as e:
                flow_state.future.set_exception(e)

            return HTMLResponse(content=AUTH_REDIRECT_SUCCESS_HTML,
                                status_code=200,
                                headers={
                                    "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"
                                })

        # Add the redirect URI route
        app.add_api_route(
            path="/auth/redirect",
            endpoint=redirect_uri,
            methods=["GET"],
            description="Handles the authorization code and state returned from the Authorization Code Grant Flow.")

        self._app = app
