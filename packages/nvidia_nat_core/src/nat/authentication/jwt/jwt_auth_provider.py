# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named JWT identity verification provider."""

from nat.authentication.credential_validator.bearer_token_validator import BearerTokenValidator
from nat.authentication.interfaces import AuthProviderBase
from nat.authentication.jwt.jwt_auth_provider_config import JwtAuthProviderConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import TokenValidationResult


class JwtAuthProvider(AuthProviderBase[JwtAuthProviderConfig]):
    """Validate inbound JWTs using a named issuer policy."""

    def __init__(self, config: JwtAuthProviderConfig) -> None:
        super().__init__(config)
        self._validator = BearerTokenValidator(
            issuer=config.issuer_url,
            audience=config.audience,
            jwks_uri=config.jwks_uri,
            scopes=config.scopes,
            timeout=config.timeout,
            leeway=config.leeway,
        )

    async def verify(self, token: str) -> TokenValidationResult:
        """Verify a JWT against this provider's configured trust policy."""
        return await self._validator.verify(token)

    @property
    def validator(self) -> BearerTokenValidator:
        """Return the cached validator used by this named provider."""
        return self._validator

    async def authenticate(self, user_id: str | None = None, **kwargs) -> AuthResult:
        """Validate the supplied ``token`` through the authentication-provider interface."""
        token = kwargs.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError("JWT authentication requires a token")

        result = await self.verify(token)
        if not result.active:
            raise ValueError("JWT verification failed")
        return AuthResult(raw=result.model_dump())
