# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""FastMCP token verifier backed by NAT's OAuth2 resource-server validator."""

from fastmcp.server.auth import AccessToken
from fastmcp.server.auth import TokenVerifier
from nat.authentication.credential_validator.bearer_token_validator import BearerTokenValidator
from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig


class NATFastMCPTokenVerifier(TokenVerifier):
    """FastMCP token verifier that delegates validation to BearerTokenValidator."""

    def __init__(self, config: OAuth2ResourceServerConfig, *, base_url: str):
        super().__init__(base_url=base_url, required_scopes=config.scopes or [])
        self._bearer_token_validator = BearerTokenValidator(
            issuer=config.issuer_url,
            audience=config.audience,
            scopes=config.scopes,
            jwks_uri=config.jwks_uri,
            introspection_endpoint=config.introspection_endpoint,
            discovery_url=config.discovery_url,
            client_id=config.client_id,
            client_secret=config.client_secret.get_secret_value() if config.client_secret else None,
            client_auth_method=config.client_auth_method,
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify a bearer token and adapt the result to FastMCP's access-token model."""
        validation_result = await self._bearer_token_validator.verify(token)
        if not validation_result.active:
            return None

        claims = {
            "aud": validation_result.audience,
            "sub": validation_result.subject,
            "iss": validation_result.issuer,
            "jti": validation_result.jti,
            "username": validation_result.username,
            "token_type": validation_result.token_type,
            "iat": validation_result.iat,
            "nbf": validation_result.nbf,
        }
        claims = {key: value for key, value in claims.items() if value is not None}

        return AccessToken(
            token=token,
            client_id=validation_result.client_id or "",
            scopes=validation_result.scopes or [],
            expires_at=validation_result.expires_at,
            claims=claims,
        )
