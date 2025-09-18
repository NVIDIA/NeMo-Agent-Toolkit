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

import json
import logging
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.jose import JsonWebKey
from authlib.jose import KeySet
from authlib.jose import jwt

from nat.data_models.authentication import TokenValidationResult

logger = logging.getLogger(__name__)


class BearerTokenValidator:
    """Bearer token validator supporting JWT and opaque tokens.

    Implements RFC 7519 (JWT) and RFC 7662 (Token Introspection) standards.
    """

    def __init__(
        self,
        introspection_endpoint: str | None = None,
        issuer: str | None = None,
        audience: str | None = None,
        jwks_uri: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        scopes: list[str] | None = None,
        timeout: float = 10.0,
        leeway: int = 60,
        discovery_url: str | None = None,
    ):
        """
        Args:
            introspection_endpoint: OAuth 2.0 introspection URL (required to validate opaque tokens).
            issuer: Expected token issuer (`iss`); recommended for policy, not required for JWT signature validity.
            audience: Expected token audience (`aud`); recommended for policy, not required for JWT signature validity.
            jwks_uri: JWKS URL with public keys to verify asymmetric JWTs; optional if using discovery.
            client_id: OAuth 2.0 client ID for authenticating to the introspection endpoint.
            client_secret: OAuth 2.0 client secret for authenticating to the introspection endpoint.
            scopes: Optional authorization scopes to check after validation; not required for token validity.
            timeout: HTTP request timeout for discovery/JWKS/introspection (default: 10.0s).
            leeway: Clock-skew allowance for `exp`/`nbf`/`iat` checks (default: 60s).
            discovery_url: OIDC/OAuth metadata URL to auto-discover `jwks_uri` and `introspection_endpoint`.
        """
        # Configuration parameters
        self.introspection_endpoint = introspection_endpoint
        self.issuer = issuer
        self.audience = audience
        self.jwks_uri = jwks_uri
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = scopes
        self.timeout = timeout
        self.leeway = leeway
        self.discovery_url = discovery_url

        # Validate configuration
        self._validate_configuration()

        # HTTPS validation for configured URLs
        if self.discovery_url:
            self._require_https(self.discovery_url, "discovery_url")
        if self.jwks_uri:
            self._require_https(self.jwks_uri, "jwks_uri")
        if self.introspection_endpoint:
            self._require_https(self.introspection_endpoint, "introspection_endpoint")

        # Caches for performance with TTL
        # JWKS cache: uri -> {keyset, cache_expires_at}
        self._jwks_cache: dict[str, dict[str, Any]] = {}
        # OIDC config cache: url -> {config, cache_expires_at}
        self._oidc_config_cache: dict[str, dict[str, Any]] = {}
        # Positive introspection result cache: token_prefix -> {result, cache_expires_at}
        self._introspection_cache: dict[str, dict[str, Any]] = {}

        # Cache TTL settings
        self._jwks_cache_ttl = 900  # 15 minutes
        self._discovery_cache_ttl = 900  # 15 minutes

    def _validate_configuration(self) -> None:
        """Validate that at least one token verification method is configured."""

        jwt_possible = self.jwks_uri or self.discovery_url or self.issuer
        introspection_possible = self.introspection_endpoint and self.client_id and self.client_secret

        if not jwt_possible and not introspection_possible:
            raise ValueError("No valid token verification method configured. "
                             "Either provide JWT verification (jwks_uri, discovery_url, or issuer for derived JWKS) "
                             "or introspection (introspection_endpoint with client_id and client_secret)")

    async def verify(self, token: str) -> TokenValidationResult:
        """Validate bearer token per RFC 7519 (JWT) and RFC 7662 (Introspection).

        Args:
            token: Bearer token to validate

        Returns:
            TokenValidationResult
        """
        if not token or not isinstance(token, str):
            return TokenValidationResult(client_id="", token_type="bearer", active=False)

        if token.startswith("Bearer "):
            token = token[7:]

        if not token:
            return TokenValidationResult(client_id="", token_type="bearer", active=False)

        try:
            if token.count(".") == 2:
                return await self._verify_jwt_token(token)
            elif (self.introspection_endpoint and self.client_id and self.client_secret):
                return await self._verify_opaque_token(token)
            else:
                return TokenValidationResult(client_id="", token_type="bearer", active=False)
        except Exception:
            return TokenValidationResult(client_id="", token_type="bearer", active=False)

    def _is_jwt_token(self, token: str) -> bool:
        """Check if token has JWT structure."""
        return token.count(".") == 2

    async def _verify_jwt_token(self, token: str) -> TokenValidationResult:
        """Verify JWT token.

        Args:
            token: JWT token to verify

        Returns:
            TokenValidationResult
        """
        jwks_uri = await self._resolve_jwks_uri()
        keyset = await self._fetch_jwks(jwks_uri)

        claims = jwt.decode(
            token,
            keyset,
            claims_options={
                "exp": {
                    "essential": True, "leeway": self.leeway
                },
                "nbf": {
                    "essential": False, "leeway": self.leeway
                },
                "iat": {
                    "essential": False, "leeway": self.leeway
                },
            },
        )
        claims.validate(leeway=self.leeway)

        issuer = claims.get("iss")
        subject = claims.get("sub")
        audience = self._extract_audience_from_claims(claims)
        scopes = claims.get("scope") or claims.get("scp")
        scopes = (scopes.split() if isinstance(scopes, str) else scopes) or None

        self._check_jwt_policies(issuer, audience, scopes)

        return TokenValidationResult(
            client_id=claims.get("azp") or claims.get("client_id") or subject,
            expires_at=claims.get("exp"),
            audience=audience,
            subject=subject,
            issuer=issuer,
            token_type="at+jwt",
            nbf=claims.get("nbf"),
            iat=claims.get("iat"),
            jti=claims.get("jti"),
            scopes=scopes,
            active=True,
        )

    async def _verify_opaque_token(self, token: str) -> TokenValidationResult:
        """Verify opaque token via RFC 7662 introspection.

        Args:
            token: Opaque token to verify

        Returns:
            TokenValidationResult
        """

        cache_key = token[:10] if len(token) >= 10 else token

        # Check cache first
        cache_entry = self._introspection_cache.get(cache_key)
        if cache_entry:
            cached_result = cache_entry["result"]
            cache_expires_at = cache_entry["cache_expires_at"]
            now = int(time.time())

            # Use cached result if not expired
            if now < cache_expires_at:
                return cached_result
            else:
                del self._introspection_cache[cache_key]

        try:
            async with AsyncOAuth2Client(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    timeout=httpx.Timeout(self.timeout),
            ) as oauth_client:
                introspection_response = await oauth_client.introspect_token(
                    self.introspection_endpoint,
                    token,
                    token_type_hint="access_token",
                )

                # Check if token is active
                if not introspection_response.get("active", False):
                    raise ValueError("Token is inactive")

                # Extract claims
                client_id = introspection_response.get("client_id")
                username = introspection_response.get("username")
                token_type = introspection_response.get("token_type", "opaque")
                expires_at = introspection_response.get("exp")
                not_before = introspection_response.get("nbf")
                issued_at = introspection_response.get("iat")
                subject = introspection_response.get("sub")
                audience = self._extract_audience_from_introspection(introspection_response)
                issuer = introspection_response.get("iss")
                jwt_id = introspection_response.get("jti")

                # Parse scopes
                scope_value = introspection_response.get("scope")
                scopes = None
                if scope_value and isinstance(scope_value, str):
                    scopes = scope_value.split()
                elif isinstance(scope_value, list):
                    scopes = scope_value

                # Check expiration and not-before with leeway
                if self._is_expired(expires_at):
                    raise ValueError("Token is expired")

                # Check not-before claim with leeway
                if not_before and self._is_not_yet_valid(not_before):
                    raise ValueError("Token is not yet valid")

                # Apply opaque token policy checks
                self._check_opaque_policies(issuer, audience, scopes)

                result = TokenValidationResult(
                    client_id=client_id,
                    username=username,
                    token_type=token_type,
                    expires_at=expires_at,
                    audience=audience,
                    subject=subject,
                    issuer=issuer,
                    jti=jwt_id,
                    scopes=scopes,
                    active=True,
                    nbf=not_before,
                    iat=issued_at,
                )

                # Cache positive result with TTL based on token expiration
                if expires_at:
                    cache_expires_at = min(expires_at, int(time.time()) + 300)  # Max 5 minutes
                    self._introspection_cache[cache_key] = {"result": result, "cache_expires_at": cache_expires_at}

                return result

        except (ValueError, TypeError, KeyError, httpx.HTTPError) as e:
            raise ValueError(f"Introspection failed: {e}") from e

    async def _resolve_jwks_uri(self) -> str:
        """Resolve JWKS URI using configuration priority: jwks_uri → discovery → issuer.

        Returns:
            JWKS URI string
        """

        if self.jwks_uri:
            return self.jwks_uri

        if self.discovery_url:
            try:
                config = await self._get_oidc_configuration(self.discovery_url)
                jwks = config.get("jwks_uri")
                if isinstance(jwks, str) and jwks:
                    self._require_https(jwks, "jwks_uri")
                    return jwks
            except Exception as e:
                raise ValueError(f"Failed to get JWKS URI from discovery: {e}") from e

        if self.issuer:
            jwks = f"{self.issuer.rstrip('/')}/.well-known/jwks.json"
            self._require_https(jwks, "jwks_uri")
            return jwks

        raise ValueError("No JWKS URI available - no jwks_uri, discovery_url, or issuer configured")

    async def _get_oidc_configuration(self, discovery_url: str) -> dict[str, Any]:
        """Get OIDC configuration.

        Args:
            discovery_url: OIDC discovery URL

        Returns:
            OIDC configuration dict
        """

        # Check cache first
        cache_entry = self._oidc_config_cache.get(discovery_url)
        if cache_entry:
            config = cache_entry["config"]
            cache_expires_at = cache_entry["cache_expires_at"]
            now = int(time.time())

            if now < cache_expires_at:
                return config
            else:
                # Remove expired entry
                del self._oidc_config_cache[discovery_url]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(discovery_url)
                response.raise_for_status()
                config = response.json()

                if not isinstance(config, dict):
                    logger.warning("OIDC discovery returned non-dict; not caching")
                    return config

                jwks_uri = config.get("jwks_uri")
                if jwks_uri is not None and not isinstance(jwks_uri, str):
                    logger.warning("OIDC discovery jwks_uri is not a string; not caching")
                    return config

                # Cache with TTL
                cache_expires_at = int(time.time()) + self._discovery_cache_ttl
                self._oidc_config_cache[discovery_url] = {"config": config, "cache_expires_at": cache_expires_at}
                return config

        except httpx.HTTPError as e:
            raise ValueError(f"OIDC discovery failed: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid OIDC discovery response: {e}") from e

    async def _fetch_jwks(self, jwks_uri: str) -> KeySet:
        """Fetch JWKS from URI.

        Args:
            jwks_uri: JWKS endpoint URI

        Returns:
            KeySet for token verification
        """

        # Check cache first
        cache_entry = self._jwks_cache.get(jwks_uri)
        if cache_entry:
            keyset = cache_entry["keyset"]
            cache_expires_at = cache_entry["cache_expires_at"]
            now = int(time.time())

            if now < cache_expires_at:
                return keyset
            else:
                # Remove expired entry
                del self._jwks_cache[jwks_uri]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(jwks_uri)
            response.raise_for_status()
            jwks_data = response.json()

        keys = jwks_data.get("keys", [])
        if not keys:
            raise ValueError("JWKS contains no keys")

        keyset = KeySet([JsonWebKey.import_key(k) for k in keys if isinstance(k, dict)])
        if not keyset:
            raise ValueError("JWKS contains no valid keys")

        # Cache keyset with TTL
        cache_expires_at = int(time.time()) + self._jwks_cache_ttl
        self._jwks_cache[jwks_uri] = {"keyset": keyset, "cache_expires_at": cache_expires_at}
        return keyset

    def _extract_audience_from_claims(self, claims: dict[str, Any]) -> list[str] | None:
        """Extract audience from JWT claims.

        Args:
            claims: JWT claims dict

        Returns:
            List of audience values
        """

        audience = claims.get("aud")
        if isinstance(audience, str):
            return [audience]
        elif isinstance(audience, list):
            filtered = [aud for aud in audience if isinstance(aud, str)]
            return filtered if filtered else None
        return None

    def _extract_audience_from_introspection(self, response: dict[str, Any]) -> list[str] | None:
        """Extract audience from introspection response.

        Args:
            response: Introspection response dict

        Returns:
            List of audience values
        """

        audience = response.get("aud")
        if isinstance(audience, str):
            return [audience]
        elif isinstance(audience, list):
            filtered = [aud for aud in audience if isinstance(aud, str)]
            return filtered if filtered else None
        return None

    def _require_https(self, url: str, url_description: str) -> None:
        """Enforce HTTPS requirement.

        Args:
            url: URL to validate
            url_description: Description for error messages
        """

        if url.startswith("https://"):
            return
        parsed_url = urlparse(url)
        if parsed_url.hostname in ("localhost", "127.0.0.1", "::1"):
            return
        raise ValueError(f"{url_description} must use HTTPS: {url}")

    def _check_jwt_policies(self,
                            issuer_claim: str | None,
                            audience_claim: list[str] | None,
                            token_scopes: list[str] | None) -> None:
        """Check JWT token against configured policies.

        Args:
            issuer_claim: Issuer from JWT token
            audience_claim: Audience list from JWT token
            token_scopes: Scopes from JWT token
        """
        # Check issuer policy
        if self.issuer and issuer_claim != self.issuer:
            raise ValueError(f"JWT issuer '{issuer_claim}' does not match expected issuer '{self.issuer}'")

        # Check audience policy
        if self.audience:
            if not audience_claim or self.audience not in audience_claim:
                raise ValueError(f"JWT audience {audience_claim} does not contain required audience '{self.audience}'")

        # Check scope policy
        if self.scopes:
            if not token_scopes:
                raise ValueError(f"JWT has no scopes but required scopes: {self.scopes}")

            token_scope_set = set(token_scopes)
            required_scope_set = set(self.scopes)

            if not required_scope_set.issubset(token_scope_set):
                missing_scopes = required_scope_set - token_scope_set
                raise ValueError(
                    f"JWT missing required scopes: {sorted(missing_scopes)} (has: {sorted(token_scope_set)})")

    def _check_opaque_policies(self,
                               issuer_claim: str | None,
                               audience_claim: list[str] | None,
                               token_scopes: list[str] | None) -> None:
        """Check opaque token against configured policies.

        Args:
            issuer_claim: Issuer from introspection response
            audience_claim: Audience list from introspection response
            token_scopes: Scopes from introspection response
        """
        # Check issuer policy
        if self.issuer and issuer_claim != self.issuer:
            raise ValueError(f"Opaque token issuer '{issuer_claim}' does not match expected issuer '{self.issuer}'")

        # Check audience policy
        if self.audience:
            if not audience_claim or self.audience not in audience_claim:
                raise ValueError(
                    f"Opaque token audience {audience_claim} does not contain required audience '{self.audience}'")

        # Check scope policy
        if self.scopes:
            if not token_scopes:
                raise ValueError(f"Opaque token has no scopes but required scopes: {self.scopes}")

            token_scope_set = set(token_scopes)
            required_scope_set = set(self.scopes)

            if not required_scope_set.issubset(token_scope_set):
                missing_scopes = required_scope_set - token_scope_set
                raise ValueError(
                    f"Opaque token missing required scopes: {sorted(missing_scopes)} (has: {sorted(token_scope_set)})")

    def _is_expired(self, exp: int | None, leeway: int | None = None) -> bool:
        """Check if timestamp is expired considering leeway.

        Args:
            exp: Expiration timestamp
            leeway: Clock skew allowance

        Returns:
            True if expired
        """

        if exp is None:
            return False
        leeway = leeway or self.leeway
        now = int(time.time())
        return now > (exp + leeway)

    def _is_not_yet_valid(self, nbf: int | None, leeway: int | None = None) -> bool:
        """Check if timestamp is not yet valid considering leeway.

        Args:
            nbf: Not-before timestamp
            leeway: Clock skew allowance

        Returns:
            True if not yet valid
        """

        if nbf is None:
            return False
        leeway = leeway or self.leeway
        now = int(time.time())
        return now < (nbf - leeway)
