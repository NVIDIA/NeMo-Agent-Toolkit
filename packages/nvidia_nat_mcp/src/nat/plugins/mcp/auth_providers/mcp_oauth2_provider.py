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
import logging
import webbrowser
from datetime import timedelta
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from threading import Thread
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlparse

from pydantic import SecretStr

from mcp.client.auth import OAuthClientProvider
from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthClientInformationFull
from mcp.shared.auth import OAuthClientMetadata
from mcp.shared.auth import OAuthToken
from nat.authentication.interfaces import AuthProviderBase
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.plugins.mcp.auth_providers.mcp_oauth2_provider_config import MCPOAuth2ProviderConfig

logger = logging.getLogger(__name__)


class InMemoryTokenStorage(TokenStorage):
    """Simple in-memory token storage implementation for MCP OAuth2 provider."""

    def __init__(self):
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info


class CallbackHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler to capture OAuth callback."""

    def __init__(self, request, client_address, server, callback_data):
        """Initialize with callback data storage."""
        self.callback_data = callback_data
        super().__init__(request, client_address, server)

    def do_GET(self):
        """Handle GET request from OAuth redirect."""
        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)

        if "code" in query_params:
            self.callback_data["authorization_code"] = query_params["code"][0]
            self.callback_data["state"] = query_params.get("state", [None])[0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
            <html>
            <body>
                <h1>Authorization Successful!</h1>
                <p>You can close this window and return to the terminal.</p>
                <script>setTimeout(() => window.close(), 2000);</script>
            </body>
            </html>
            """)
        elif "error" in query_params:
            self.callback_data["error"] = query_params["error"][0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"""
            <html>
            <body>
                <h1>Authorization Failed</h1>
                <p>Error: {query_params["error"][0]}</p>
                <p>You can close this window and return to the terminal.</p>
            </body>
            </html>
            """.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class CallbackServer:
    """Simple server to handle OAuth callbacks."""

    def __init__(self, port: int = 3030):
        self.port = port
        self.server: HTTPServer | None = None
        self.thread: Thread | None = None
        self.callback_data: dict[str, Any] = {}

    def start(self):
        """Start the callback server."""
        self.server = HTTPServer(("localhost", self.port), lambda request, client_address, server: CallbackHandler(
            request, client_address, server, self.callback_data))
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"Callback server started on port {self.port}")

    def stop(self):
        """Stop the callback server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None

    def wait_for_callback(self, timeout: int = 300) -> str:
        """Wait for OAuth callback and return authorization code."""
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            if "authorization_code" in self.callback_data:
                return self.callback_data["authorization_code"]
            if "error" in self.callback_data:
                raise RuntimeError(f"OAuth authorization failed: {self.callback_data['error']}")
            time.sleep(0.1)

        raise RuntimeError("OAuth callback timeout")

    def get_state(self) -> str | None:
        """Get the state parameter from the callback."""
        return self.callback_data.get("state")


class MCPOAuth2Provider(AuthProviderBase[MCPOAuth2ProviderConfig]):
    """MCP OAuth2 authentication provider that wraps MCP-SDK's OAuthClientProvider."""

    def __init__(self, config: MCPOAuth2ProviderConfig):
        super().__init__(config)
        self._oauth_provider: OAuthClientProvider | None = None
        self._server_url: str | None = None

    async def authenticate(self, user_id: str | None = None) -> AuthResult:
        """Authenticate and return auth result."""
        if not self._oauth_provider:
            raise RuntimeError("OAuth provider not initialized. Call get_oauth_provider() first.")

        # Get tokens from OAuth provider
        tokens = await self._oauth_provider.get_tokens()
        if not tokens:
            raise RuntimeError("No valid tokens available. Authentication may have failed.")

        return AuthResult(credentials=[BearerTokenCred(token=SecretStr(tokens.access_token))],
                          token_expires_at=tokens.expires_at,
                          raw=tokens.model_dump())

    def get_oauth_provider(self, server_url: str) -> OAuthClientProvider:
        """Get OAuth provider for MCP client."""
        if self._oauth_provider is None or self._server_url != server_url:
            self._server_url = server_url
            self._oauth_provider = self._create_oauth_provider(server_url)
        return self._oauth_provider

    def _create_oauth_provider(self, server_url: str) -> OAuthClientProvider:
        """Create OAuth provider based on configuration."""
        # Determine redirect URI
        redirect_uri = self.config.redirect_uri or "http://localhost:3030/callback"

        # Create client metadata
        client_metadata_dict = {
            "client_name": self.config.client_name,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_post" if self.config.client_secret else "none",
        }

        # Add client credentials if provided (Option 4: hybrid mode)
        if self.config.client_id:
            client_metadata_dict["client_id"] = self.config.client_id
        if self.config.client_secret:
            client_metadata_dict["client_secret"] = self.config.client_secret

        # Add scopes if provided
        if self.config.scopes:
            client_metadata_dict["scope"] = " ".join(self.config.scopes)

        client_metadata = OAuthClientMetadata.model_validate(client_metadata_dict)

        # Create callback handler
        callback_server = CallbackServer(port=3030)
        callback_server.start()

        async def callback_handler() -> tuple[str, str | None]:
            """Wait for OAuth callback and return auth code and state."""
            logger.info("Waiting for authorization callback...")
            try:
                auth_code = callback_server.wait_for_callback(timeout=300)
                return auth_code, callback_server.get_state()
            finally:
                callback_server.stop()

        async def redirect_handler(authorization_url: str) -> None:
            """Default redirect handler that opens the URL in a browser."""
            logger.info(f"Opening browser for authorization: {authorization_url}")
            webbrowser.open(authorization_url)

        # Create OAuth authentication handler
        oauth_auth = OAuthClientProvider(
            server_url=server_url.replace("/mcp", ""),
            client_metadata=client_metadata,
            storage=InMemoryTokenStorage(),
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )

        return oauth_auth
