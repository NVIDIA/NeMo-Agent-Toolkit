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
from urllib.parse import urljoin
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl

from nat.authentication.interfaces import AuthProviderBase
from nat.data_models.authentication import AuthReason
from nat.data_models.authentication import AuthRequest
from nat.data_models.authentication import AuthResult
from nat.plugins.mcp.auth.mcp_oauth2_provider_config import MCPOAuth2ProviderConfig

logger = logging.getLogger(__name__)


class OAuth2Endpoints(BaseModel):
    """OAuth2 endpoints discovered from MCP server."""
    authorization_url: HttpUrl = Field(..., description="OAuth2 authorization endpoint URL")
    token_url: HttpUrl = Field(..., description="OAuth2 token endpoint URL")
    registration_url: HttpUrl | None = Field(default=None, description="OAuth2 client registration endpoint URL")


class OAuth2Credentials(BaseModel):
    """OAuth2 client credentials from registration."""
    client_id: str = Field(..., description="OAuth2 client identifier")
    client_secret: str | None = Field(default=None, description="OAuth2 client secret")


class DiscoverOAuth2Endpoints:
    """
    Endpoint discovery utility that follows the MCP-SDK flow:

    1) If a 401 provided a WWW-Authenticate header with a resource_metadata hint (RFC 9728),
    fetch that Protected Resource Metadata.
    2) Otherwise, fetch the RS's well-known Protected Resource Metadata at:
    /.well-known/oauth-protected-resource (relative to the server's base).
    3) If the protected resource metadata lists authorization_servers (issuers), pick the first issuer
    and perform path-aware RFC 8414 / OIDC discovery against it:
        - /.well-known/oauth-authorization-server{path}
        - /.well-known/oauth-authorization-server
        - /.well-known/openid-configuration{path}
        - {issuer}/.well-known/openid-configuration
    Return the first doc that yields both authorization_endpoint and token_endpoint.
    4) If the protected resource metadata directly embeds endpoints (non-standard), use them.
    5) If no 9728 info leads anywhere, fall back to path-aware 8414 / OIDC using the MCP server URL.
    """

    def __init__(self, config: MCPOAuth2ProviderConfig):
        self.config = config
        self._cached_endpoints: OAuth2Endpoints | None = None

    async def discover(self, reason: AuthReason, www_authenticate: str | None) -> tuple[OAuth2Endpoints, bool]:
        """
        Discover OAuth2 endpoints. Returns (endpoints, changed), where 'changed' is True
        iff the selected endpoints differ from the cached ones.
        """
        # Fast path: reuse cache when not a 401 retry
        if reason != AuthReason.RETRY_AFTER_401 and self._cached_endpoints is not None:
            return self._cached_endpoints, False

        # Default to server URL if unable to discover issuer
        issuer: str = str(self.config.server_url)
        endpoints: OAuth2Endpoints | None = None

        # 1) 401 hint (RFC 9728) if present
        if reason == AuthReason.RETRY_AFTER_401 and www_authenticate:
            hint_url = self._extract_from_www_authenticate_header(www_authenticate)
            if hint_url:
                logger.info("Using RFC 9728 resource_metadata hint: %s", hint_url)
                issuer, endpoints = await self._fetch_protected_resource_metadata(hint_url)

        # 2) If no endpoints yet, try RS protected resource metadata well-known
        if endpoints is None:
            pr_url = urljoin(self._authorization_base_url(), "/.well-known/oauth-protected-resource")
            try:
                logger.debug("Fetching protected resource metadata: %s", pr_url)
                issuer2, endpoints2 = await self._fetch_protected_resource_metadata(pr_url)
                # prefer newly learned issuer/endpoints if provided
                issuer = issuer2 or issuer
                endpoints = endpoints2 or endpoints
            except Exception as e:
                logger.debug("Protected resource metadata not available: %s", e)

        # 3) Choose: direct endpoints from 9728 vs issuer-based 8414/OIDC
        if endpoints is None:
            endpoints = await self._discover_via_issuer_or_base(issuer)

        if endpoints is None:
            raise RuntimeError("Could not discover OAuth2 endpoints from MCP server")

        changed = (self._cached_endpoints is None
                   or endpoints.authorization_url != self._cached_endpoints.authorization_url
                   or endpoints.token_url != self._cached_endpoints.token_url
                   or endpoints.registration_url != self._cached_endpoints.registration_url)

        self._cached_endpoints = endpoints
        logger.info("OAuth2 endpoints selected: %s", self._cached_endpoints)
        return self._cached_endpoints, changed

    # ---------------------------
    # Helpers
    # ---------------------------
    def _authorization_base_url(self) -> str:
        """
        Derive a base URL for RS well-known discovery: scheme://host
        (mirrors MCP-SDK's use of an 'authorization base URL').
        """
        parsed = urlparse(str(self.config.server_url))
        return f"{parsed.scheme}://{parsed.netloc}"

    def _extract_from_www_authenticate_header(self, www_authenticate: str) -> str | None:
        """Extract resource_metadata URL from WWW-Authenticate (robust parsing)."""
        import re

        if not www_authenticate:
            return None
        # resource_metadata="url" | 'url' | url  (case-insensitive; stop on space/comma/semicolon)
        pattern = r'(?i)\bresource_metadata\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|([^\s,;]+))'
        match = re.search(pattern, www_authenticate)
        if match:
            url = next((g for g in match.groups() if g), None)
            if url:
                logger.debug("Extracted resource_metadata URL: %s", url)
                return url
        return None

    async def _fetch_protected_resource_metadata(self, url: str) -> tuple[str | None, OAuth2Endpoints | None]:
        """
        Fetch RFC 9728 Protected Resource Metadata.
        Returns (issuer, endpoints) where:
          - issuer: first authorization server URL (if provided)
          - endpoints: if the doc directly contains endpoints (non-standard), they’re returned
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        # Standard RFC 9728: authorization_servers (list of issuers)
        issuers = data.get("authorization_servers")
        if isinstance(issuers, list) and issuers:
            issuer = str(issuers[0])
            return issuer, None

        # Non-standard nested object with endpoints
        if isinstance(data.get("authorization_server"), dict):
            as_obj = data["authorization_server"]
            auth_url = as_obj.get("authorization_endpoint")
            token_url = as_obj.get("token_endpoint")
            registration_url = as_obj.get("registration_endpoint")
            if auth_url and token_url:
                return None, OAuth2Endpoints(
                    authorization_url=auth_url,
                    token_url=token_url,
                    registration_url=registration_url,
                )

        # Nothing usable
        return None, None

    def _parse_oauth_metadata(self, data: dict) -> OAuth2Endpoints | None:
        """Extract endpoints from an OAuth/OIDC metadata document."""
        auth_url = data.get("authorization_endpoint")
        token_url = data.get("token_endpoint")
        registration_url = data.get("registration_endpoint")
        if auth_url and token_url:
            return OAuth2Endpoints(
                authorization_url=auth_url,
                token_url=token_url,
                registration_url=registration_url,
            )
        return None

    async def _discover_via_issuer_or_base(self, base_or_issuer: str) -> OAuth2Endpoints | None:
        """
        Perform path-aware RFC 8414 / OIDC discovery given an issuer or base URL.
        """
        urls = self._build_path_aware_discovery_urls(base_or_issuer)
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in urls:
                try:
                    response = await client.get(url)
                    if response.status_code != 200:
                        continue
                    data = response.json()
                    endpoints = self._parse_oauth_metadata(data)
                    if endpoints:
                        logger.info("Discovered OAuth2 endpoints from %s", url)
                        return endpoints
                except Exception as e:
                    logger.debug("Discovery failed at %s: %s", url, e)
        return None

    def _build_path_aware_discovery_urls(self, base_or_issuer: str) -> list[str]:
        """
        Build candidate URLs in the same order MCP-SDK uses:
          - /.well-known/oauth-authorization-server{path}
          - /.well-known/oauth-authorization-server
          - /.well-known/openid-configuration{path}
          - {base_or_issuer}/.well-known/openid-configuration
        """
        parsed = urlparse(base_or_issuer)
        base = f"{parsed.scheme}://{parsed.netloc}"
        path = (parsed.path or "").rstrip("/")
        urls: list[str] = []

        if path and path != "":
            urls.append(urljoin(base, f"/.well-known/oauth-authorization-server{path}"))
        urls.append(urljoin(base, "/.well-known/oauth-authorization-server"))

        if path and path != "":
            urls.append(urljoin(base, f"/.well-known/openid-configuration{path}"))
        urls.append(base_or_issuer.rstrip("/") + "/.well-known/openid-configuration")

        return urls


class DynamicClientRegistration:
    """Dynamic client registration utility."""

    def __init__(self, config: MCPOAuth2ProviderConfig):
        self.config = config

    async def register(self, endpoints: OAuth2Endpoints) -> OAuth2Credentials:
        """
        Register OAuth2 client with the AS using Dynamic Client Registration (RFC 7591).

        Notes:
        - Omits optional fields rather than sending empty strings/lists (many ASes reject those).
        """
        if not endpoints or not endpoints.registration_url:
            raise RuntimeError("No registration endpoint found in discovered OAuth2 metadata")

        registration_url = str(endpoints.registration_url)

        # Build client metadata; include optional fields only when meaningful.
        client_metadata: dict[str, object] = {
            "client_name": self.config.client_name,
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        }
        if self.config.redirect_uri:
            client_metadata["redirect_uris"] = [str(self.config.redirect_uri)]
        if self.config.scopes:
            client_metadata["scope"] = " ".join(self.config.scopes)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                registration_url,
                json=client_metadata,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception as e:
                # Provide a clearer error if the AS returned non-JSON or invalid JSON
                raise RuntimeError(f"Registration response was not valid JSON from {registration_url}") from e

        # Extract credentials
        client_id = data.get("client_id")
        client_secret = data.get("client_secret")

        if not isinstance(client_id, str) or not client_id:
            raise RuntimeError("No client_id received from registration")

        logger.info("Successfully registered OAuth2 client: %s", client_id)
        return OAuth2Credentials(client_id=client_id,
                                 client_secret=client_secret if isinstance(client_secret, str) else None)


class MCPOAuth2Provider(AuthProviderBase[MCPOAuth2ProviderConfig]):
    """MCP OAuth2 authentication provider that delegates to NAT framework."""

    def __init__(self, config: MCPOAuth2ProviderConfig):
        super().__init__(config)

        # Discovery
        self._discoverer = DiscoverOAuth2Endpoints(config)
        self._cached_endpoints: OAuth2Endpoints | None = None

        # Client registration
        self._registrar = DynamicClientRegistration(config)
        self._cached_credentials: OAuth2Credentials | None = None

        # For the OAuth2 flow
        self._auth_code_provider = None
        self._auth_code_key = None

    async def authenticate(self, user_id: str | None = None, auth_request: AuthRequest | None = None) -> AuthResult:
        """
        Authenticate using MCP OAuth2 flow via NAT framework.
        1. Dynamic endpoints discovery (RFC9728 + RFC 8414 + OIDC)
        2. Client registration (RFC7591)
        3. Use NAT's standard OAuth2 flow (OAuth2AuthCodeFlowProvider)
        """
        if not auth_request:
            raise RuntimeError("Auth request is required")

        # Discover OAuth2 endpoints
        self._cached_endpoints, endpoints_changed = await self._discoverer.discover(reason=auth_request.reason,
                                                                                    www_authenticate=auth_request.www_authenticate)
        if endpoints_changed:
            self._cached_credentials = None  # invalidate credentials tied to old AS

        if endpoints_changed:
            logger.info("OAuth2 endpoints: %s", self._cached_endpoints)

        # Client registration
        if not self._cached_credentials:
            if self.config.client_id:
                # Manual registration mode
                self._cached_credentials = OAuth2Credentials(
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret,
                )
                logger.info("Using manual client_id: %s", self._cached_credentials.client_id)
            else:
                if not self.config.enable_dynamic_registration:
                    raise RuntimeError(
                        "Dynamic registration is not enabled and no client_id/client_secret were provided")
            if not self._cached_endpoints:

                # Dynamic registration mode requires registration endpoint
                self._cached_credentials = await self._registrar.register(self._cached_endpoints)
                logger.info("Registered OAuth2 client: %s", self._cached_credentials.client_id)

        # Use NAT's standard OAuth2 flow
        if auth_request.reason == AuthReason.RETRY_AFTER_401:
            # force fresh delegate (clears in-mem token cache)
            self._auth_code_provider = None
            # preserve other fields, just normalize reason & inject user_id
            auth_request = auth_request.model_copy(update={
                "reason": AuthReason.NORMAL, "user_id": user_id, "www_authenticate": None
            })
        else:
            # back-compat: propagate user_id if provided but not set in the request
            if user_id is not None and auth_request.user_id is None:
                auth_request = auth_request.model_copy(update={"user_id": user_id})

        return await self._perform_oauth2_flow(auth_request=auth_request)

    async def _perform_oauth2_flow(self, auth_request: AuthRequest | None = None) -> AuthResult:
        from nat.authentication.oauth2.oauth2_auth_code_flow_provider import OAuth2AuthCodeFlowProvider
        from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig

        # This helper is only for non-401 flows
        if auth_request and auth_request.reason == AuthReason.RETRY_AFTER_401:
            raise RuntimeError("_perform_oauth2_flow should not be called for RETRY_AFTER_401")

        endpoints = self._cached_endpoints
        credentials = self._cached_credentials
        if not endpoints or not credentials:
            raise RuntimeError("OAuth2 flow called before discovery/registration")

        # Build a key so we only (re)create the delegate when material config changes
        key = (
            str(endpoints.authorization_url),
            str(endpoints.token_url),
            str(endpoints.registration_url) if endpoints.registration_url else None,
            credentials.client_id,
            credentials.client_secret or None,
            tuple(self.config.scopes or []),
            str(self.config.redirect_uri) if self.config.redirect_uri else None,
            getattr(self.config, "token_endpoint_auth_method", None),
            bool(self.config.use_pkce),
        )

        # (Re)build the delegate if needed
        if self._auth_code_provider is None or self._auth_code_key != key:
            if not self.config.redirect_uri:
                raise RuntimeError("Redirect URI is not set")

            oauth2_config = OAuth2AuthCodeFlowProviderConfig(
                client_id=credentials.client_id,
                client_secret=credentials.client_secret or "",
                authorization_url=str(endpoints.authorization_url),
                token_url=str(endpoints.token_url),
                token_endpoint_auth_method=getattr(self.config, "token_endpoint_auth_method", None),
                redirect_uri=str(self.config.redirect_uri) if self.config.redirect_uri else "",
                scopes=self.config.scopes or [],
                use_pkce=bool(self.config.use_pkce),
            )

            self._auth_code_provider = OAuth2AuthCodeFlowProvider(oauth2_config)
            self._auth_code_key = key

        # Let the delegate handle per-user cache + refresh
        user_id = self._resolve_user_id(user_id=None, auth_request=auth_request)
        return await self._auth_code_provider.authenticate(user_id=user_id)
