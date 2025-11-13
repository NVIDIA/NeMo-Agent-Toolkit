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
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import Credential
from nat.data_models.authentication import HeaderCred
from nat.plugins.mcp.auth.service_account.provider_config import MCPServiceAccountProviderConfig
from nat.plugins.mcp.auth.service_account.token_client import ServiceAccountTokenClient

logger = logging.getLogger(__name__)


class MCPServiceAccountProvider(AuthProviderBase[MCPServiceAccountProviderConfig]):
    """
    MCP service account authentication provider using OAuth2 client credentials.

    Provides headless authentication for MCP clients using service account credentials.
    Supports both standard Bearer tokens and custom token prefix formats.

    Example Configuration:
        authentication:
          my_service:
            _type: mcp_service_account
            client_id: ${SERVICE_ACCOUNT_CLIENT_ID}
            client_secret: ${SERVICE_ACCOUNT_CLIENT_SECRET}
            token_url: https://auth.example.com/oauth/token
            scopes:
              - api.read
              - api.write
            token_prefix: service_account  # Optional
            service_token: ${SERVICE_TOKEN}  # Optional
    """

    def __init__(self, config: MCPServiceAccountProviderConfig, builder=None):
        super().__init__(config)

        # Initialize token client
        self._token_client = ServiceAccountTokenClient(
            client_id=config.client_id,
            client_secret=config.client_secret,
            token_url=config.token_url,
            scopes=" ".join(config.scopes),  # Convert list to space-delimited string for OAuth2
            token_cache_buffer_seconds=config.token_cache_buffer_seconds,
        )

        logger.info("Initialized MCP service account auth provider: "
                    "token_url=%s, scopes=%s, has_service_token=%s",
                    config.token_url,
                    config.scopes,
                    config.service_token is not None)

    async def authenticate(self, user_id: str | None = None, **kwargs) -> AuthResult:
        """
        Authenticate using OAuth2 client credentials flow.

        Note: user_id is ignored for service accounts (non-session-specific).

        Returns:
            AuthResult with HeaderCred objects for service account authentication
        """
        # Get OAuth2 access token (cached if still valid)
        access_token = await self._token_client.get_access_token()

        # Format Authorization header value
        if self.config.token_prefix:
            bearer_token = f"{self.config.token_prefix}:{access_token.get_secret_value()}"
        else:
            # Standard Bearer token (no custom prefix)
            bearer_token = access_token.get_secret_value()

        # Build credentials list using HeaderCred
        credentials: list[Credential] = [HeaderCred(name="Authorization", value=SecretStr(f"Bearer {bearer_token}"))]

        # Add service-specific token if provided
        if self.config.service_token:
            service_token = self.config.service_token.get_secret_value()
            credentials.append(HeaderCred(name=self.config.service_token_header, value=SecretStr(service_token)))

        # Return AuthResult with HeaderCred objects
        return AuthResult(
            credentials=credentials,
            token_expires_at=self._token_client._token_expires_at,
            raw={},
        )
