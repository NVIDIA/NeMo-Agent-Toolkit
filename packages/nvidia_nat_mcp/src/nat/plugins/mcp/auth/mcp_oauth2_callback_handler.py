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

import logging

from pydantic import SecretStr

from mcp.client.auth import OAuthClientProvider
from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthClientInformationFull
from mcp.shared.auth import OAuthClientMetadata
from mcp.shared.auth import OAuthToken
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.plugins.mcp.auth.mcp_oauth2_provider_config import MCPOAuth2ProviderConfig

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


class MCPOAuth2CallbackHandler:
    """Handles MCP OAuth2 authentication callbacks within NAT framework."""

    def __init__(self):
        self._oauth_providers: dict[str, OAuthClientProvider] = {}

    async def handle_mcp_oauth2_flow(self, config: MCPOAuth2ProviderConfig, server_url: str) -> AuthResult:
        """Handle MCP OAuth2 flow using NAT's redirect handling."""

        # Get or create OAuth provider for this server
        oauth_provider = self._get_oauth_provider(config, server_url)

        # Get tokens
        tokens = await oauth_provider.get_tokens()
        if not tokens:
            raise RuntimeError("MCP OAuth2 authentication failed")

        return AuthResult(credentials=[BearerTokenCred(token=SecretStr(tokens.access_token))],
                          token_expires_at=tokens.expires_at,
                          raw=tokens.model_dump())

    def _get_oauth_provider(self, config: MCPOAuth2ProviderConfig, server_url: str) -> OAuthClientProvider:
        """Get or create OAuth provider for the server."""
        if server_url not in self._oauth_providers:
            self._oauth_providers[server_url] = self._create_oauth_provider(config, server_url)
        return self._oauth_providers[server_url]

    def _create_oauth_provider(self, config: MCPOAuth2ProviderConfig, server_url: str) -> OAuthClientProvider:
        """Create OAuth provider for MCP server."""
        # Create client metadata
        client_metadata = self._create_client_metadata(config)

        # Create OAuth provider (no local server needed)
        oauth_provider = OAuthClientProvider(
            server_url=server_url.replace("/mcp", ""),
            client_metadata=client_metadata,
            storage=InMemoryTokenStorage(),
            redirect_handler=self._nat_redirect_handler,  # Use NAT's redirect
            callback_handler=self._nat_callback_handler,  # Use NAT's callback
        )

        return oauth_provider

    def _create_client_metadata(self, config: MCPOAuth2ProviderConfig) -> OAuthClientMetadata:
        """Create OAuth client metadata from MCP config."""
        redirect_uri = config.redirect_uri or "http://localhost:3030/callback"

        metadata_dict = {
            "client_name": config.client_name,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_post",
        }

        if config.scopes:
            metadata_dict["scope"] = " ".join(config.scopes)

        return OAuthClientMetadata.model_validate(metadata_dict)

    async def _nat_redirect_handler(self, authorization_url: str) -> None:
        """Use NAT's redirect handling instead of webbrowser.open."""
        # This would integrate with NAT's web framework
        # to handle the redirect properly
        logger.info("Opening browser for authorization: %s", authorization_url)
        # For now, we'll use webbrowser as a fallback
        # In the future, this should integrate with NAT's web framework
        import webbrowser
        webbrowser.open(authorization_url)

    async def _nat_callback_handler(self) -> tuple[str, str | None]:
        """Use NAT's callback handling instead of local server."""
        # This would integrate with NAT's web framework
        # to handle the callback properly
        # For now, we'll raise an error indicating this needs framework integration
        raise NotImplementedError("MCP OAuth2 callback handling requires NAT framework integration. "
                                  "This should be implemented in the NAT web framework to handle OAuth2 redirects.")
