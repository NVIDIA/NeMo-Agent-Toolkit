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

from mcp.shared.auth import OAuthClientInformationFull
from mcp.shared.auth import OAuthClientMetadata
from mcp.shared.auth import OAuthMetadata
from mcp.shared.auth import ProtectedResourceMetadata
from nat.authentication.interfaces import AuthProviderBase
from nat.data_models.authentication import AuthReason
from nat.data_models.authentication import AuthRequest
from nat.data_models.authentication import AuthResult
from nat.plugins.mcp.auth.auth_provider_config import MCPOAuth2ProviderConfig

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
    MCP-SDK parity discovery flow:
      1) If 401 + WWW-Authenticate has resource_metadata (RFC 9728), fetch it.
      2) Else fetch RS well-known /.well-known/oauth-protected-resource.
      3) If PR metadata lists authorization_servers, pick first as issuer.
      4) Do path-aware RFC 8414 / OIDC discovery against issuer (or server base).
    """

    def __init__(self, config: MCPOAuth2ProviderConfig):
        self.config = config
        self._cached_endpoints: OAuth2Endpoints | None = None
        self._last_oauth_scopes: list[str] | None = None

    async def discover(self, reason: AuthReason, www_authenticate: str | None) -> tuple[OAuth2Endpoints, bool]:
        """
        Discover OAuth2 endpoints from MCP server.

        Args:
            reason: The reason for the discovery.
            www_authenticate: The WWW-Authenticate header from a 401 response.

        Returns:
            A tuple of OAuth2Endpoints and a boolean indicating if the endpoints have changed.
        """
        # Fast path: reuse cache when not a 401 retry
        if reason != AuthReason.RETRY_AFTER_401 and self._cached_endpoints is not None:
            return self._cached_endpoints, False

        issuer: str = str(self.config.server_url)  # default to server URL
        endpoints: OAuth2Endpoints | None = None

        # 1) 401 hint (RFC 9728) if present
        if reason == AuthReason.RETRY_AFTER_401 and www_authenticate:
            hint_url = self._extract_from_www_authenticate_header(www_authenticate)
            if hint_url:
                logger.info("Using RFC 9728 resource_metadata hint: %s", hint_url)
                issuer_hint = await self._fetch_pr_issuer(hint_url)
                if issuer_hint:
                    issuer = issuer_hint

        # 2) Try RS protected resource well-known if we still only have default issuer
        if issuer == str(self.config.server_url):
            pr_url = urljoin(self._authorization_base_url(), "/.well-known/oauth-protected-resource")
            try:
                logger.debug("Fetching protected resource metadata: %s", pr_url)
                issuer2 = await self._fetch_pr_issuer(pr_url)
                if issuer2:
                    issuer = issuer2
            except Exception as e:
                logger.debug("Protected resource metadata not available: %s", e)

        # 3) Path-aware RFC 8414 / OIDC discovery using issuer (or server base)
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

    # --------------------------- helpers ---------------------------
    def _authorization_base_url(self) -> str:
        """Get the authorization base URL from the MCP server URL."""
        p = urlparse(str(self.config.server_url))
        return f"{p.scheme}://{p.netloc}"

    def _extract_from_www_authenticate_header(self, hdr: str) -> str | None:
        """Extract the resource_metadata URL from the WWW-Authenticate header."""
        import re

        if not hdr:
            return None
        # resource_metadata="url" | 'url' | url (case-insensitive; stop on space/comma/semicolon)
        m = re.search(r'(?i)\bresource_metadata\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|([^\s,;]+))', hdr)
        if not m:
            return None
        url = next((g for g in m.groups() if g), None)
        if url:
            logger.debug("Extracted resource_metadata URL: %s", url)
        return url

    async def _fetch_pr_issuer(self, url: str) -> str | None:
        """Fetch RFC 9728 Protected Resource Metadata and return the first issuer (authorization_server)."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            body = await resp.aread()
        try:
            pr = ProtectedResourceMetadata.model_validate_json(body)
        except Exception as e:
            logger.debug("Invalid ProtectedResourceMetadata at %s: %s", url, e)
            return None
        if pr.authorization_servers:
            return str(pr.authorization_servers[0])
        return None

    async def _discover_via_issuer_or_base(self, base_or_issuer: str) -> OAuth2Endpoints | None:
        """Perform path-aware RFC 8414 / OIDC discovery given an issuer or base URL."""
        urls = self._build_path_aware_discovery_urls(base_or_issuer)
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in urls:
                try:
                    resp = await client.get(url, headers={"Accept": "application/json"})
                    if resp.status_code != 200:
                        continue
                    body = await resp.aread()
                    try:
                        meta = OAuthMetadata.model_validate_json(body)
                    except Exception as e:
                        logger.debug("Invalid OAuthMetadata at %s: %s", url, e)
                        continue
                    if meta.authorization_endpoint and meta.token_endpoint:
                        logger.info("Discovered OAuth2 endpoints from %s", url)
                        # this is bit of a hack to get the scopes supported by the auth server
                        self._last_oauth_scopes = meta.scopes_supported
                        return OAuth2Endpoints(
                            authorization_url=str(meta.authorization_endpoint),
                            token_url=str(meta.token_endpoint),
                            registration_url=str(meta.registration_endpoint) if meta.registration_endpoint else None,
                        )
                except Exception as e:
                    logger.debug("Discovery failed at %s: %s", url, e)
        return None

    def _build_path_aware_discovery_urls(self, base_or_issuer: str) -> list[str]:
        """Build path-aware discovery URLs."""
        p = urlparse(base_or_issuer)
        base = f"{p.scheme}://{p.netloc}"
        path = (p.path or "").rstrip("/")
        urls: list[str] = []
        if path:
            urls.append(urljoin(base, f"/.well-known/oauth-authorization-server{path}"))
        urls.append(urljoin(base, "/.well-known/oauth-authorization-server"))
        if path:
            urls.append(urljoin(base, f"/.well-known/openid-configuration{path}"))
        urls.append(base_or_issuer.rstrip("/") + "/.well-known/openid-configuration")
        return urls

    def scopes_supported(self) -> list[str] | None:
        """Get the last OAuth scopes discovered from the AS."""
        return self._last_oauth_scopes


class DynamicClientRegistration:
    """Dynamic client registration utility."""

    def __init__(self, config: MCPOAuth2ProviderConfig):
        self.config = config

    def _authorization_base_url(self) -> str:
        """Get the authorization base URL from the MCP server URL."""
        p = urlparse(str(self.config.server_url))
        return f"{p.scheme}://{p.netloc}"

    async def register(self, endpoints: OAuth2Endpoints, scopes: list[str] | None) -> OAuth2Credentials:
        """Register an OAuth2 client with the Authorization Server using OIDC client registration."""
        # Fallback to /register if metadata didn't provide an endpoint
        registration_url = (str(endpoints.registration_url) if endpoints.registration_url else urljoin(
            self._authorization_base_url(), "/register"))

        metadata = OAuthClientMetadata(
            redirect_uris=[self.config.redirect_uri],
            token_endpoint_auth_method=(getattr(self.config, "token_endpoint_auth_method", None)
                                        or "client_secret_post"),
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=" ".join(scopes) if scopes else None,
            client_name=self.config.client_name or None,
        )
        payload = metadata.model_dump(by_alias=True, mode="json", exclude_none=True)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                registration_url,
                json=payload,
                headers={
                    "Content-Type": "application/json", "Accept": "application/json"
                },
            )
            resp.raise_for_status()
            body = await resp.aread()

        try:
            info = OAuthClientInformationFull.model_validate_json(body)
        except Exception as e:
            raise RuntimeError(
                f"Registration response was not valid OAuthClientInformation from {registration_url}") from e

        if not info.client_id:
            raise RuntimeError("No client_id received from registration")

        logger.info("Successfully registered OAuth2 client: %s", info.client_id)
        return OAuth2Credentials(client_id=info.client_id, client_secret=info.client_secret)


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

    async def authenticate(self, user_id: str | None = None) -> AuthResult:
        """
        Authenticate using MCP OAuth2 flow via NAT framework.
        1. Dynamic endpoints discovery (RFC9728 + RFC 8414 + OIDC)
        2. Client registration (RFC7591)
        3. Use NAT's standard OAuth2 flow (OAuth2AuthCodeFlowProvider)
        """
        auth_request = self.config.auth_request
        if not auth_request:
            auth_request = AuthRequest(reason=AuthReason.NORMAL)

        if auth_request.reason != AuthReason.RETRY_AFTER_401:
            # auth provider is expected to be setup via 401, till that time we return empty auth result
            if not self._auth_code_provider:
                return AuthResult(credentials=[], token_expires_at=None, raw={})

        await self._discover_and_register(auth_request)
        # Use NAT's standard OAuth2 flow
        if auth_request.reason == AuthReason.RETRY_AFTER_401:
            # force fresh delegate (clears in-mem token cache)
            self._auth_code_provider = None
            # preserve other fields, just normalize reason & inject user_id
            auth_request = auth_request.model_copy(update={
                "reason": AuthReason.NORMAL, "user_id": user_id, "www_authenticate": None
            })
        # back-compat: propagate user_id if provided but not set in the request
        elif user_id is not None and auth_request.user_id is None:
            auth_request = auth_request.model_copy(update={"user_id": user_id})

        # Perform the OAuth2 flow without lock
        return await self._perform_oauth2_flow(auth_request=auth_request)

    async def _discover_and_register(self, auth_request: AuthRequest):
        """
        Discover OAuth2 endpoints and register an OAuth2 client with the Authorization Server
        using OIDC client registration.
        """
        # Discover OAuth2 endpoints
        self._cached_endpoints, endpoints_changed = await self._discoverer.discover(reason=auth_request.reason,
                                                                                    www_authenticate=auth_request.www_authenticate)
        if endpoints_changed:
            logger.info("OAuth2 endpoints: %s", self._cached_endpoints)
            self._cached_credentials = None  # invalidate credentials tied to old AS
        effective_scopes = self._effective_scopes()

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
                # Dynamic registration mode requires registration endpoint
                self._cached_credentials = await self._registrar.register(self._cached_endpoints, effective_scopes)
                logger.info("Registered OAuth2 client: %s", self._cached_credentials.client_id)

    def _effective_scopes(self) -> list[str] | None:
        """
        Prefer caller-provided scopes; otherwise fall back to AS-advertised scopes_supported.
        """
        return self.config.scopes or self._discoverer.scopes_supported()

    async def _build_oauth2_delegate(self):
        """Build NAT OAuth2 provider and delegate auth token acquisition and refresh to it"""
        from nat.authentication.oauth2.oauth2_auth_code_flow_provider import OAuth2AuthCodeFlowProvider
        from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig

        endpoints = self._cached_endpoints
        credentials = self._cached_credentials

        if self._auth_code_provider is None:
            oauth2_config = OAuth2AuthCodeFlowProviderConfig(
                client_id=credentials.client_id,
                client_secret=credentials.client_secret or "",
                authorization_url=str(endpoints.authorization_url),
                token_url=str(endpoints.token_url),
                token_endpoint_auth_method=getattr(self.config, "token_endpoint_auth_method", None),
                redirect_uri=str(self.config.redirect_uri) if self.config.redirect_uri else "",
                scopes=self._effective_scopes() or [],
                use_pkce=bool(self.config.use_pkce),
            )

            self._auth_code_provider = OAuth2AuthCodeFlowProvider(oauth2_config)

    async def _perform_oauth2_flow(self, auth_request: AuthRequest | None = None) -> AuthResult:
        """Perform the OAuth2 flow using NAT OAuth2 provider."""
        # This helper is only for non-401 flows
        if auth_request and auth_request.reason == AuthReason.RETRY_AFTER_401:
            raise RuntimeError("_perform_oauth2_flow should not be called for RETRY_AFTER_401")

        if not self._cached_endpoints or not self._cached_credentials:
            raise RuntimeError("OAuth2 flow called before discovery/registration")

        # (Re)build the delegate if needed
        await self._build_oauth2_delegate()
        # Let the delegate handle per-user cache + refresh
        return await self._auth_code_provider.authenticate()
