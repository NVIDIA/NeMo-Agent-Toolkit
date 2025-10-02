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
from collections.abc import Awaitable
from collections.abc import Callable
from urllib.parse import urljoin
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl
from pydantic import TypeAdapter

from mcp.shared.auth import OAuthClientInformationFull
from mcp.shared.auth import OAuthClientMetadata
from mcp.shared.auth import OAuthMetadata
from mcp.shared.auth import ProtectedResourceMetadata
from nat.authentication.interfaces import AuthenticatedContext
from nat.authentication.interfaces import AuthFlowType
from nat.authentication.interfaces import AuthProviderBase
from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.data_models.authentication import AuthResult
from nat.plugins.mcp.auth.auth_flow_handler import MCPAuthenticationFlowHandler
from nat.plugins.mcp.auth.auth_provider_config import MCPOAuth2ProviderConfig

logger = logging.getLogger(__name__)


class OAuth2Endpoints(BaseModel):
    """OAuth2 endpoints discovered from MCP server."""
    authorization_url: HttpUrl = Field(..., description="OAuth2 authorization endpoint URL")
    token_url: HttpUrl = Field(..., description="OAuth2 token endpoint URL")
    registration_url: HttpUrl | None = Field(default=None, description="OAuth2 client registration endpoint URL")
    scopes: list[str] | None = Field(default=None, description="OAuth2 scopes to be used for the authentication")


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

        self._flow_handler: MCPAuthenticationFlowHandler = MCPAuthenticationFlowHandler()

    async def discover(self, response: httpx.Response | None = None) -> tuple[OAuth2Endpoints, bool]:
        """
        Discover OAuth2 endpoints from MCP server.

        Args:
            reason: The reason for the discovery.
            www_authenticate: The WWW-Authenticate header from a 401 response.

        Returns:
            A tuple of OAuth2Endpoints and a boolean indicating if the endpoints have changed.
        """
        is_401_retry = response is not None and response.status_code == 401
        # Fast path: reuse cache when not a 401 retry
        if not is_401_retry and self._cached_endpoints is not None:
            return self._cached_endpoints, False

        issuer: str = str(self.config.server_url)  # default to server URL
        endpoints: OAuth2Endpoints | None = None

        # 1) 401 hint (RFC 9728) if present
        if is_401_retry and response:
            www_authenticate = response.headers.get("WWW-Authenticate")
            if www_authenticate:
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

        changed = (self._cached_endpoints is None or endpoints.model_dump() != self._cached_endpoints.model_dump())
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
                    resp = await client.get(url, follow_redirects=True, headers={"Accept": "application/json"})
                    if resp.status_code != 200:
                        continue

                    # Check content type before attempting JSON parsing
                    content_type = resp.headers.get("content-type", "").lower()
                    if "application/json" not in content_type:
                        logger.info(
                            "Discovery endpoint %s returned non-JSON content type: %s. "
                            "This may indicate the endpoint doesn't support discovery or requires authentication.",
                            url,
                            content_type)
                        # If it's HTML, log a more helpful message
                        if "text/html" in content_type:
                            logger.info("The endpoint appears to be returning an HTML page instead of OAuth metadata. "
                                        "This often means:")
                            logger.info("1. The OAuth discovery endpoint doesn't exist at this URL")
                            logger.info("2. The server requires authentication before providing discovery metadata")
                            logger.info("3. The URL is pointing to a web application instead of an OAuth server")
                        continue

                    body = await resp.aread()

                    try:
                        meta = OAuthMetadata.model_validate_json(body)
                    except Exception as e:
                        logger.debug("Invalid OAuthMetadata at %s: %s", url, e)
                        continue
                    if meta.authorization_endpoint and meta.token_endpoint:
                        logger.info("Discovered OAuth2 endpoints from %s", url)
                        # Convert AnyHttpUrl to HttpUrl using TypeAdapter
                        http_url_adapter = TypeAdapter(HttpUrl)
                        return OAuth2Endpoints(
                            authorization_url=http_url_adapter.validate_python(str(meta.authorization_endpoint)),
                            token_url=http_url_adapter.validate_python(str(meta.token_endpoint)),
                            registration_url=http_url_adapter.validate_python(str(meta.registration_endpoint))
                            if meta.registration_endpoint else None,
                            scopes=meta.scopes_supported,
                        )
                except Exception as e:
                    logger.debug("Discovery failed at %s: %s", url, e)

        # If we get here, all discovery URLs failed
        logger.info("OAuth discovery failed for all attempted URLs.")
        logger.info("Attempted URLs: %s", urls)
        return None

    def _build_path_aware_discovery_urls(self, base_or_issuer: str) -> list[str]:
        """Build path-aware discovery URLs."""
        p = urlparse(base_or_issuer)
        base = f"{p.scheme}://{p.netloc}"
        path = (p.path or "").rstrip("/")
        urls: list[str] = []
        if path:
            # this is the specified by the MCP spec
            urls.append(urljoin(base, f".well-known/oauth-protected-resource{path}"))
            # this is fallback for backward compatibility
            urls.append(urljoin(base, f"{path}/.well-known/oauth-authorization-server"))
        urls.append(urljoin(base, "/.well-known/oauth-authorization-server"))
        if path:
            # this is the specified by the MCP spec
            urls.append(urljoin(base, f".well-known/openid-configuration{path}"))
            # this is fallback for backward compatibility
            urls.append(urljoin(base, f"{path}/.well-known/openid-configuration"))
        urls.append(base_or_issuer.rstrip("/") + "/.well-known/openid-configuration")
        return urls


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

    def __init__(self, config: MCPOAuth2ProviderConfig, builder=None):
        super().__init__(config)
        self._builder = builder

        # Discovery
        self._discoverer = DiscoverOAuth2Endpoints(config)
        self._cached_endpoints: OAuth2Endpoints | None = None

        # Client registration
        self._registrar = DynamicClientRegistration(config)
        self._cached_credentials: OAuth2Credentials | None = None

        # For the OAuth2 flow
        self._auth_code_provider = None
        self._flow_handler = MCPAuthenticationFlowHandler()

        self._auth_callback = None

        # Initialize token storage
        self._token_storage = None
        self._token_storage_object_store_name = None

        if self.config.token_storage_object_store:
            # Store object store name, will be resolved later when builder context is available
            self._token_storage_object_store_name = self.config.token_storage_object_store
            logger.info(f"Configured to use object store '{self._token_storage_object_store_name}' for token storage")
        else:
            # Default: use in-memory token storage
            from .token_storage import InMemoryTokenStorage
            self._token_storage = InMemoryTokenStorage()

    def _set_custom_auth_callback(self,
                                  auth_callback: Callable[[OAuth2AuthCodeFlowProviderConfig, AuthFlowType],
                                                          Awaitable[AuthenticatedContext]]):
        """Set the custom authentication callback."""
        if not self._auth_callback:
            logger.info("Using custom authentication callback")
            self._auth_callback = auth_callback
            if self._auth_code_provider:
                self._auth_code_provider._set_custom_auth_callback(self._auth_callback)  # type: ignore[arg-type]

    async def authenticate(self, user_id: str | None = None, **kwargs) -> AuthResult:
        """
        Authenticate using MCP OAuth2 flow via NAT framework.

        If response is provided in kwargs (typically from a 401), performs:
        1. Dynamic endpoints discovery (RFC9728 + RFC 8414 + OIDC)
        2. Client registration (RFC7591)
        3. Authentication

        Otherwise, performs standard authentication flow.
        """
        if not user_id:
            # MCP tool calls cannot be made without an authorized user
            raise RuntimeError("User is not authorized to call the tool")

        response = kwargs.get('response')
        if response and response.status_code == 401:
            await self._discover_and_register(response=response)

        return await self._nat_oauth2_authenticate(user_id=user_id)

    @property
    def _effective_scopes(self) -> list[str]:
        """Get the effective scopes to be used for the authentication."""
        return self.config.scopes or (self._cached_endpoints.scopes if self._cached_endpoints else []) or []

    async def _discover_and_register(self, response: httpx.Response | None = None):
        """
        Discover OAuth2 endpoints and register an OAuth2 client with the Authorization Server
        using OIDC client registration.
        """
        # Discover OAuth2 endpoints
        self._cached_endpoints, endpoints_changed = await self._discoverer.discover(response=response)
        if endpoints_changed:
            logger.info("OAuth2 endpoints: %s", self._cached_endpoints)
            self._cached_credentials = None  # invalidate credentials tied to old AS
            self._auth_code_provider = None
        effective_scopes = self._effective_scopes

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

    async def _nat_oauth2_authenticate(self, user_id: str | None = None) -> AuthResult:
        """Perform the OAuth2 flow using MCP-specific authentication flow handler."""
        from nat.authentication.oauth2.oauth2_auth_code_flow_provider import OAuth2AuthCodeFlowProvider

        if not self._cached_endpoints or not self._cached_credentials:
            # if discovery is yet to to be done return empty auth result
            return AuthResult(credentials=[], token_expires_at=None, raw={})

        endpoints = self._cached_endpoints
        credentials = self._cached_credentials

        # Resolve object store reference if needed
        if self._token_storage_object_store_name and not self._token_storage:
            try:
                if not self._builder:
                    raise RuntimeError("Builder not available for resolving object store")
                object_store = await self._builder.get_object_store_client(self._token_storage_object_store_name)
                from .token_storage import ObjectStoreTokenStorage
                self._token_storage = ObjectStoreTokenStorage(object_store)
                logger.info(f"Initialized token storage with object store '{self._token_storage_object_store_name}'")
            except Exception as e:
                logger.warning(
                    f"Failed to resolve object store '{self._token_storage_object_store_name}' for token storage: {e}. "
                    "Falling back to in-memory storage.")
                from .token_storage import InMemoryTokenStorage
                self._token_storage = InMemoryTokenStorage()

        # Build the OAuth2 provider if not already built
        if self._auth_code_provider is None:
            scopes = self._effective_scopes
            oauth2_config = OAuth2AuthCodeFlowProviderConfig(
                client_id=credentials.client_id,
                client_secret=credentials.client_secret or "",
                authorization_url=str(endpoints.authorization_url),
                token_url=str(endpoints.token_url),
                token_endpoint_auth_method=getattr(self.config, "token_endpoint_auth_method", None),
                redirect_uri=str(self.config.redirect_uri) if self.config.redirect_uri else "",
                scopes=scopes,
                use_pkce=bool(self.config.use_pkce),
                authorization_kwargs={"resource": str(self.config.server_url)})
            self._auth_code_provider = OAuth2AuthCodeFlowProvider(oauth2_config, token_storage=self._token_storage)

            # Use MCP-specific authentication method if available
            if hasattr(self._auth_code_provider, "_set_custom_auth_callback"):
                callback = self._auth_callback or self._flow_handler.authenticate
                self._auth_code_provider._set_custom_auth_callback(callback)  # type: ignore[arg-type]

        # Auth code provider is responsible for per-user cache + refresh
        return await self._auth_code_provider.authenticate(user_id=user_id)
