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
"""OAuth 2.0 Token Introspection verifier implementation for MCP servers."""

import logging

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.provider import TokenVerifier

from nat.authentication.credential_validator.bearer_token_validator import BearerTokenValidator
from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig

logger = logging.getLogger(__name__)


class IntrospectionTokenVerifier(TokenVerifier):
    """Token verifier that delegates token verification to BearerTokenValidator."""

    def __init__(self, config: OAuth2ResourceServerConfig):
        """Create IntrospectionTokenVerifier from OAuth2ResourceServerConfig.

        Args:
            config: OAuth2ResourceServerConfig
        """
        issuer = config.issuer_url
        scopes = config.scopes or []
        audience = config.audience
        jwks_uri = config.jwks_uri
        introspection_endpoint = config.introspection_endpoint
        discovery_url = config.discovery_url
        client_id = config.client_id
        client_secret = config.client_secret

        self._bearer_token_validator = BearerTokenValidator(
            issuer=issuer,
            audience=audience,
            scopes=scopes,
            jwks_uri=jwks_uri,
            introspection_endpoint=introspection_endpoint,
            discovery_url=discovery_url,
            client_id=client_id,
            client_secret=client_secret,
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify token by delegating to BearerTokenValidator.

        Args:
            token: The Bearer token to verify

        Returns:
            AccessToken | None: AccessToken if valid, None if invalid
        """
        validation_result = await self._bearer_token_validator.verify(token)

        if validation_result.active:
            return AccessToken(token=token,
                               expires_at=validation_result.expires_at,
                               scopes=validation_result.scopes or [],
                               client_id=validation_result.client_id or "")
        return None
