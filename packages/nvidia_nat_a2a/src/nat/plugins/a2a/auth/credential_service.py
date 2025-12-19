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
"""Bridge NAT AuthProviderBase to A2A SDK CredentialService."""

import asyncio
import logging

from a2a.client import ClientCallContext
from a2a.client import CredentialService
from a2a.types import AgentCard
from a2a.types import APIKeySecurityScheme
from a2a.types import HTTPAuthSecurityScheme
from a2a.types import OAuth2SecurityScheme
from a2a.types import OpenIdConnectSecurityScheme
from a2a.types import SecurityScheme
from nat.authentication.interfaces import AuthProviderBase
from nat.builder.context import Context
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BasicAuthCred
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.authentication import CookieCred
from nat.data_models.authentication import HeaderCred
from nat.data_models.authentication import QueryCred

logger = logging.getLogger(__name__)


class A2ACredentialService(CredentialService):
    """
    Adapts NAT AuthProviderBase to A2A SDK CredentialService interface.

    This class bridges NAT's authentication system with the A2A SDK's authentication
    mechanism, allowing A2A clients to use NAT's auth providers (API Key, OAuth2, etc.)
    to authenticate with A2A agents.

    The adapter:
    - Calls NAT auth provider to obtain credentials
    - Maps NAT credential types to A2A security scheme requirements
    - Handles token expiration and automatic refresh
    - Supports session-based multi-user authentication

    Args:
        auth_provider: NAT authentication provider instance
        agent_card: Agent card containing security scheme definitions
    """

    def __init__(
        self,
        auth_provider: AuthProviderBase,
        agent_card: AgentCard | None = None,
    ):
        self._auth_provider = auth_provider
        self._agent_card = agent_card
        self._cached_auth_result: AuthResult | None = None
        self._auth_lock = asyncio.Lock()

        # Validate provider compatibility with agent's security requirements
        self._validate_provider_compatibility()

    async def get_credentials(
        self,
        security_scheme_name: str,
        context: ClientCallContext | None,
    ) -> str | None:
        """
        Retrieve credentials for a security scheme.

        This method:
        1. Gets user_id from NAT context
        2. Authenticates via NAT auth provider
        3. Handles token expiration and refresh
        4. Maps credentials to the requested security scheme

        Args:
            security_scheme_name: Name of the security scheme from AgentCard
            context: Client call context with optional session information

        Returns:
            Credential string or None if not available
        """
        # Get user_id from NAT context
        user_id = Context.get().user_id

        # Authenticate and get credentials from NAT provider
        auth_result = await self._authenticate(user_id)

        if not auth_result:
            logger.warning("Authentication failed, no credentials available")
            return None

        # Map NAT credentials to A2A format based on security scheme
        credential = self._extract_credential_for_scheme(auth_result, security_scheme_name)

        if credential:
            logger.debug(
                "Successfully retrieved credentials for scheme '%s'",
                security_scheme_name,
            )
        else:
            logger.warning(
                "No compatible credentials found for scheme '%s'",
                security_scheme_name,
            )

        return credential

    async def _authenticate(self, user_id: str | None) -> AuthResult | None:
        """
        Authenticate and get credentials from NAT auth provider.

        Handles token expiration by triggering re-authentication if needed.
        Uses a lock to prevent concurrent authentication requests and race conditions.

        Args:
            user_id: User identifier for authentication

        Returns:
            AuthResult with credentials or None on failure
        """
        try:
            # Fast path: check cache without lock
            auth_result = self._cached_auth_result
            if auth_result and not auth_result.is_expired():
                return auth_result

            # Acquire lock to serialize authentication attempts
            async with self._auth_lock:
                # Double-check: another coroutine may have refreshed while we waited for lock
                auth_result = self._cached_auth_result
                if auth_result and not auth_result.is_expired():
                    logger.debug("Credentials were refreshed by another coroutine while waiting for lock")
                    return auth_result

                # Log if we're refreshing expired credentials
                if auth_result and auth_result.is_expired():
                    logger.info("Cached credentials expired, re-authenticating")

                # Call NAT auth provider (provider is responsible for token refresh/validity)
                auth_result = await self._auth_provider.authenticate(user_id=user_id)

                # Cache the result while holding the lock
                self._cached_auth_result = auth_result

                # Warn if provider returned expired credentials (provider bug)
                if auth_result and auth_result.is_expired():
                    logger.warning("Auth provider returned already-expired credentials. "
                                   "This may indicate a bug in the auth provider's token refresh logic.")

                return auth_result

        except Exception as e:
            logger.error("Authentication failed: %s", e, exc_info=True)
            return None

    def _extract_credential_for_scheme(self, auth_result: AuthResult, security_scheme_name: str) -> str | None:
        """
        Extract appropriate credential based on security scheme type.

        Maps NAT credential types to A2A security scheme requirements:
        - BearerTokenCred -> OAuth2, OIDC, HTTP Bearer
        - HeaderCred -> API Key in header
        - QueryCred -> API Key in query
        - CookieCred -> API Key in cookie
        - BasicAuthCred -> HTTP Basic

        Args:
            auth_result: Authentication result containing credentials
            security_scheme_name: Name of the security scheme

        Returns:
            Credential string or None
        """
        # Get scheme definition from agent card
        scheme_def = self._get_scheme_definition(security_scheme_name)

        # Try to match NAT credentials to security scheme
        for cred in auth_result.credentials:
            # Check compatibility and extract credential value
            credential_value = None

            if isinstance(cred, BearerTokenCred) and self._is_bearer_compatible(scheme_def):
                credential_value = cred.token.get_secret_value()
            elif isinstance(cred, HeaderCred) and self._is_header_compatible(scheme_def, cred.name):
                credential_value = cred.value.get_secret_value()
            elif isinstance(cred, QueryCred) and self._is_query_compatible(scheme_def, cred.name):
                credential_value = cred.value.get_secret_value()
            elif isinstance(cred, CookieCred) and self._is_cookie_compatible(scheme_def, cred.name):
                credential_value = cred.value.get_secret_value()
            elif isinstance(cred, BasicAuthCred) and self._is_basic_compatible(scheme_def):
                # For HTTP Basic, encode username:password as base64
                import base64

                username = cred.username.get_secret_value()
                password = cred.password.get_secret_value()
                credentials = f"{username}:{password}"
                credential_value = base64.b64encode(credentials.encode()).decode()

            if credential_value:
                return credential_value

        return None

    def _get_scheme_definition(self, scheme_name: str) -> SecurityScheme | None:
        """
        Get security scheme definition from agent card.

        Args:
            scheme_name: Name of the security scheme

        Returns:
            SecurityScheme definition or None
        """
        if not self._agent_card or not self._agent_card.security_schemes:
            return None
        return self._agent_card.security_schemes.get(scheme_name)

    def _validate_provider_compatibility(self) -> None:
        """
        Validate that the auth provider type is compatible with agent's security schemes.

        This performs early validation at connection time to fail fast if there's a
        configuration mismatch between the NAT auth provider and the A2A agent's
        security requirements.

        Raises:
            ValueError: If the provider is incompatible with all required security schemes
        """
        if not self._agent_card or not self._agent_card.security_schemes:
            # No security schemes defined, nothing to validate
            logger.debug("No security schemes defined in agent card, skipping validation")
            return

        provider_type = type(self._auth_provider).__name__
        schemes = self._agent_card.security_schemes

        logger.info("Validating auth provider '%s' against agent security schemes: %s",
                    provider_type,
                    list(schemes.keys()))

        # Check if provider type is compatible with at least one security scheme
        compatible_schemes = []
        incompatible_schemes = []

        for scheme_name, scheme in schemes.items():
            is_compatible = self._is_provider_compatible_with_scheme(scheme)
            if is_compatible:
                compatible_schemes.append(scheme_name)
            else:
                incompatible_schemes.append((scheme_name, type(scheme.root).__name__))

        if not compatible_schemes:
            # Provider is not compatible with any security scheme
            scheme_details = ", ".join(f"{name} ({scheme_type})" for name, scheme_type in incompatible_schemes)
            raise ValueError(f"Auth provider '{provider_type}' is not compatible with agent's "
                             f"security requirements. Agent requires: {scheme_details}")

        logger.info("Auth provider '%s' is compatible with schemes: %s", provider_type, compatible_schemes)

    def _is_provider_compatible_with_scheme(self, scheme: SecurityScheme) -> bool:
        """
        Check if the current auth provider can satisfy a security scheme.

        Args:
            scheme: Security scheme from agent card

        Returns:
            True if provider is compatible with the scheme
        """
        provider_type = type(self._auth_provider).__name__

        # OAuth2/OIDC schemes require OAuth2 providers
        if isinstance(scheme.root, OAuth2SecurityScheme | OpenIdConnectSecurityScheme):
            return "OAuth2" in provider_type

        # API Key schemes (can be in header, query, or cookie)
        if isinstance(scheme.root, APIKeySecurityScheme):
            return "APIKey" in provider_type

        # HTTP Auth schemes (Basic or Bearer)
        if isinstance(scheme.root, HTTPAuthSecurityScheme):
            scheme_lower = scheme.root.scheme.lower()
            if scheme_lower == "basic":
                return "HTTPBasic" in provider_type or "BasicAuth" in provider_type
            elif scheme_lower == "bearer":
                # Bearer can be satisfied by OAuth2 or API Key providers
                return "OAuth2" in provider_type or "APIKey" in provider_type

        # Unknown or unsupported scheme type
        logger.warning("Unknown security scheme type: %s", type(scheme.root).__name__)
        return False

    @staticmethod
    def _is_bearer_compatible(scheme_def: SecurityScheme | None) -> bool:
        """
        Check if security scheme accepts Bearer tokens.

        Bearer tokens are compatible with:
        - OAuth2SecurityScheme
        - OpenIdConnectSecurityScheme
        - HTTPAuthSecurityScheme with scheme='bearer'

        Args:
            scheme_def: Security scheme definition

        Returns:
            True if Bearer token is compatible
        """
        if not scheme_def:
            return False

        # Check for OAuth2 or OIDC schemes
        if isinstance(scheme_def.root, OAuth2SecurityScheme | OpenIdConnectSecurityScheme):
            return True

        # Check for HTTP Bearer scheme
        if isinstance(scheme_def.root, HTTPAuthSecurityScheme):
            return scheme_def.root.scheme.lower() == "bearer"

        return False

    @staticmethod
    def _is_header_compatible(scheme_def: SecurityScheme | None, header_name: str) -> bool:
        """
        Check if security scheme accepts header-based API keys.

        Args:
            scheme_def: Security scheme definition
            header_name: Name of the header containing the credential

        Returns:
            True if header credential is compatible
        """
        if not scheme_def:
            return False

        # Check for API Key in header
        if isinstance(scheme_def.root, APIKeySecurityScheme):
            if scheme_def.root.in_ == "header":
                # Match header name (case-insensitive)
                return scheme_def.root.name.lower() == header_name.lower()

        return False

    @staticmethod
    def _is_query_compatible(scheme_def: SecurityScheme | None, param_name: str) -> bool:
        """
        Check if security scheme accepts query parameter API keys.

        Args:
            scheme_def: Security scheme definition
            param_name: Name of the query parameter

        Returns:
            True if query credential is compatible
        """
        if not scheme_def:
            return False

        # Check for API Key in query
        if isinstance(scheme_def.root, APIKeySecurityScheme):
            if scheme_def.root.in_ == "query":
                return scheme_def.root.name == param_name

        return False

    @staticmethod
    def _is_cookie_compatible(scheme_def: SecurityScheme | None, cookie_name: str) -> bool:
        """
        Check if security scheme accepts cookie-based API keys.

        Args:
            scheme_def: Security scheme definition
            cookie_name: Name of the cookie

        Returns:
            True if cookie credential is compatible
        """
        if not scheme_def:
            return False

        # Check for API Key in cookie
        if isinstance(scheme_def.root, APIKeySecurityScheme):
            if scheme_def.root.in_ == "cookie":
                return scheme_def.root.name == cookie_name

        return False

    @staticmethod
    def _is_basic_compatible(scheme_def: SecurityScheme | None) -> bool:
        """
        Check if security scheme accepts HTTP Basic authentication.

        Args:
            scheme_def: Security scheme definition

        Returns:
            True if Basic auth is compatible
        """
        if not scheme_def:
            return False

        # Check for HTTP Basic scheme
        if isinstance(scheme_def.root, HTTPAuthSecurityScheme):
            return scheme_def.root.scheme.lower() == "basic"

        return False
