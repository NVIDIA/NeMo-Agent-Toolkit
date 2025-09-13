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
from typing import Any
from typing import overload
from urllib.parse import urlparse

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.provider import TokenVerifier

from nat.authentication.credential_validator.bearer_token_validator import BearerTokenValidator
from nat.data_models.authentication import TokenValidationResult

logger = logging.getLogger(__name__)


class IntrospectionTokenVerifier(TokenVerifier):
    """Token verifier that validates configuration and delegates to BearerTokenValidator.

    This is a thin wrapper that validates configuration parameters and maintains
    the MCP TokenVerifier interface while delegating all token validation to BearerTokenValidator.
    """

    def __init__(self,
                 client_id: str,
                 client_secret: str,
                 introspection_endpoint: str | None = None,
                 issuer: str | None = None,
                 audience: str | None = None,
                 jwks_uri: str | None = None,
                 scopes: list[str] | None = None,
                 timeout: float = 10.0,
                 leeway: int = 30,
                 allowed_algorithms: list[str] | None = None,
                 discovery_url: str | None = None):

        # Store configuration for validation
        self.client_id = client_id
        self.client_secret = client_secret
        self.introspection_endpoint = introspection_endpoint
        self.issuer = issuer
        self.audience = audience
        self.jwks_uri = jwks_uri
        self.scopes = scopes or []
        self.timeout = timeout
        self.leeway = leeway
        self.discovery_url = discovery_url

        # Security defaults - only asymmetric algorithms (RFC 7518)
        self.allowed_algorithms = allowed_algorithms or [
            "RS256",
            "RS384",
            "RS512",  # RSA with SHA
            "ES256",
            "ES384",
            "ES512",  # ECDSA with SHA
        ]

        # Validate configuration - if invalid, don't create BearerTokenValidator
        try:
            self._validate_configuration()
            # Only create BearerTokenValidator if configuration is valid
            self._bearer_token_validator = BearerTokenValidator(client_id=client_id,
                                                                client_secret=client_secret,
                                                                introspection_endpoint=introspection_endpoint,
                                                                issuer=issuer,
                                                                audience=audience,
                                                                jwks_uri=jwks_uri,
                                                                scopes=scopes,
                                                                timeout=timeout,
                                                                leeway=leeway,
                                                                discovery_url=discovery_url)
        except ValueError as e:
            logger.error("Configuration validation failed: %s", e)
            return None

    def _validate_configuration(self) -> None:
        """Validate configuration for RFC 7519/7662 compliance.

        Raises:
            ValueError: If configuration is invalid or incomplete
        """
        # Validate required parameters
        if not self.client_id:
            raise ValueError("client_id is required and must be a non-empty string")
        if not self.client_secret:
            raise ValueError("client_secret is required and must be a non-empty string")

        # Check for at least one validation method
        jwt_possible = self.issuer or self.discovery_url
        introspection_possible = self.introspection_endpoint

        if not jwt_possible and not introspection_possible:
            error_msg = ("Configuration incomplete: Either 'issuer' (for JWT validation) "
                         "or 'introspection_endpoint' (for opaque token validation) must be configured")
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate introspection requirements
        if introspection_possible:
            if not self.client_id or not self.client_secret:
                error_msg = ("client_id and client_secret are required when introspection_endpoint is configured")
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Validate algorithm security
        if "none" in self.allowed_algorithms:
            error_msg = "Algorithm 'none' is never allowed for security reasons (RFC 7518)"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate URLs
        for name, url in [
            ("introspection_endpoint", self.introspection_endpoint),
            ("issuer", self.issuer),
            ("jwks_uri", self.jwks_uri),
            ("discovery_url", self.discovery_url),
        ]:
            if url and not self._is_valid_https_url(url):
                # Check if it's just a non-HTTPS issue or invalid URL format
                try:
                    parsed = urlparse(url)
                    if parsed.scheme and parsed.netloc:
                        # Valid URL format but not HTTPS
                        raise ValueError(f"{name} must use HTTPS: {url}")
                    else:
                        # Invalid URL format
                        raise ValueError(f"{name} must be a valid URL: {url}")
                except Exception:
                    # Invalid URL format
                    raise ValueError(f"{name} must be a valid URL: {url}")

    def _is_valid_https_url(self, url: str) -> bool:
        """Validate URL is HTTPS (with localhost exception).

        Args:
            url: URL to validate

        Returns:
            True if valid HTTPS URL or localhost HTTP, False otherwise
        """
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False

            # HTTPS required except for localhost
            if parsed.scheme == "https":
                return True
            elif parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1"):
                return True
            else:
                return False
        except Exception:
            return False

    @overload
    async def verify_token(self, token: str) -> TokenValidationResult | None:
        ...

    @overload
    async def verify_token(self, token: str) -> AccessToken | None:
        ...

    async def verify_token(self, token: str) -> Any:
        """Verify token by delegating to BearerTokenValidator.

        Args: token: The Bearer token to verify

        Returns: TokenIntrospectionResult if valid, None if invalid or verification fails """

        if self._bearer_token_validator is None:
            logger.error("BearerTokenValidator is not initialized.")
            return None

        try:
            return await self._bearer_token_validator.verify(token)
        except Exception as e:
            logger.debug("Token verification failed: %s", e)
            return None
