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

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.context import Context
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.plugins.mcp.auth.mcp_oauth2_callback_handler import MCPOAuth2CallbackHandler
from nat.plugins.mcp.auth.mcp_oauth2_provider_config import MCPOAuth2ProviderConfig

logger = logging.getLogger(__name__)


class MCPOAuth2Provider(AuthProviderBase[MCPOAuth2ProviderConfig]):
    """MCP OAuth2 authentication provider that delegates to NAT framework."""

    def __init__(self, config: MCPOAuth2ProviderConfig):
        super().__init__(config)
        self._context = Context.get()
        self._mcp_callback_handler = MCPOAuth2CallbackHandler()
        self._server_url: str | None = None

    async def authenticate(self, user_id: str | None = None) -> AuthResult:
        """Authenticate using MCP OAuth2 flow via NAT framework."""

        # Get server URL from context or config
        server_url = self.config.server_url

        # Delegate to NAT framework's auth callback
        auth_callback = self._context.user_auth_callback
        if not auth_callback:
            raise RuntimeError("Authentication callback not set on Context.")

        try:
            # Use MCP-specific flow type
            authenticated_context = await auth_callback(self.config, AuthFlowType.MCP_OAUTH2, server_url=server_url)
            return self._extract_auth_result(authenticated_context)
        except Exception as e:
            raise RuntimeError(f"MCP OAuth2 authentication failed: {e}") from e

    def _extract_auth_result(self, authenticated_context) -> AuthResult:
        """Extract AuthResult from authenticated context."""
        # Extract token and metadata from the context
        auth_header = authenticated_context.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise RuntimeError("Invalid Authorization header")

        token = auth_header.split(" ")[1]

        return AuthResult(
            credentials=[BearerTokenCred(token=SecretStr(token))],
            token_expires_at=authenticated_context.metadata.get("expires_at"),
            raw=authenticated_context.metadata.get("raw_token"),
        )
