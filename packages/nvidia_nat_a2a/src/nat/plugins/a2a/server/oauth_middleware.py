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
"""OAuth 2.0 token validation middleware for A2A servers."""

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from nat.authentication.credential_validator.bearer_token_validator import BearerTokenValidator
from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig

logger = logging.getLogger(__name__)


class OAuth2ValidationMiddleware(BaseHTTPMiddleware):
    """OAuth2 Bearer token validation middleware for A2A servers.

    Validates Bearer tokens using NAT's BearerTokenValidator which supports:
    - JWT validation via JWKS (RFC 7519)
    - Opaque token validation via introspection (RFC 7662)
    - OIDC discovery
    - Scope and audience enforcement

    The middleware allows public access to the agent card discovery endpoint
    (/.well-known/agent.json) and validates all other A2A requests.
    """

    def __init__(self, app, config: OAuth2ResourceServerConfig):
        """Initialize OAuth2 validation middleware.

        Args:
            app: Starlette application
            config: OAuth2 resource server configuration
        """
        super().__init__(app)

        # Create validator using NAT's BearerTokenValidator
        self.validator = BearerTokenValidator(
            issuer=config.issuer_url,
            audience=config.audience,
            scopes=config.scopes,
            jwks_uri=config.jwks_uri,
            introspection_endpoint=config.introspection_endpoint,
            discovery_url=config.discovery_url,
            client_id=config.client_id,
            client_secret=config.client_secret.get_secret_value() if config.client_secret else None,
        )

        logger.info(
            "OAuth2 validation middleware initialized (issuer=%s, scopes=%s, audience=%s)",
            config.issuer_url,
            config.scopes,
            config.audience,
        )

    async def dispatch(self, request: Request, call_next):
        """Validate OAuth2 Bearer token for all requests except agent card discovery.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response (either error or result from next handler)
        """
        # Public: Agent card discovery (per A2A spec)
        if request.url.path == "/.well-known/agent-card.json":
            logger.debug("Public access to agent card discovery")
            return await call_next(request)

        # Extract Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning("Missing or invalid Authorization header")
            return JSONResponse({
                "error": "unauthorized", "message": "Missing or invalid Bearer token"
            },
                                status_code=401)

        token = auth_header[7:]  # Strip "Bearer "

        # Validate token using NAT's validator
        try:
            result = await self.validator.verify(token)
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return JSONResponse({"error": "invalid_token", "message": "Token validation failed"}, status_code=403)

        # Check if token is active
        if not result.active:
            logger.warning("Token is not active")
            return JSONResponse({"error": "invalid_token", "message": "Token is not active"}, status_code=403)

        # Attach token info to request state for potential use by handlers
        request.state.oauth_user = result.subject
        request.state.oauth_scopes = result.scopes or []
        request.state.oauth_client_id = result.client_id
        request.state.oauth_token_info = result

        logger.debug(
            "Token validated successfully (user=%s, scopes=%s, client=%s)",
            result.subject,
            result.scopes,
            result.client_id,
        )

        return await call_next(request)
