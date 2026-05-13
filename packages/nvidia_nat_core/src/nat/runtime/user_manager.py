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
"""Runtime credential resolver that auto-detects identity source and creates UserInfo."""

from __future__ import annotations

import base64
import logging
import typing
from http.cookies import SimpleCookie

from fastapi import WebSocket
from pydantic import SecretStr
from starlette.requests import Request

from nat.authentication.jwt_utils import decode_jwt_claims_unverified
from nat.data_models.api_server import ApiKeyAuthPayload
from nat.data_models.api_server import AuthPayload
from nat.data_models.api_server import BasicAuthPayload
from nat.data_models.api_server import JwtAuthPayload
from nat.data_models.user_info import BasicUserInfo
from nat.data_models.user_info import JwtUserInfo
from nat.data_models.user_info import UserInfo

logger = logging.getLogger(__name__)


class UserManager:
    """Stateless resolver that creates ``UserInfo`` from HTTP/WebSocket connections."""

    @classmethod
    def extract_user_from_connection(cls, connection: Request | WebSocket) -> UserInfo | None:
        """Resolve an HTTP/WebSocket connection into a ``UserInfo``.

        Args:
            connection: The incoming Starlette ``Request`` or ``WebSocket``.

        Returns:
            A fully populated ``UserInfo``, or ``None`` if no credential
            is present on the connection.

        Raises:
            ValueError: If a credential is found but cannot be resolved
                to a valid user identity.
        """
        cookie: str | None = cls._get_session_cookie(connection)
        if cookie:
            return cls._user_info_from_session_cookie(cookie)

        auth_header: str | None = cls._get_auth_header(connection)
        if auth_header:
            resolved: UserInfo | None = cls._resolve_from_auth_header(auth_header)
            if resolved is not None:
                return resolved

        api_key: str | None = cls._get_api_key_header(connection)
        if api_key:
            return UserInfo._from_api_key(api_key)

        return None

    @classmethod
    def _resolve_from_auth_header(cls, auth_header: str) -> UserInfo | None:
        """Parse an ``Authorization`` header and resolve identity by scheme.

        Args:
            auth_header: Raw header value (e.g. ``Bearer <token>`` or ``Basic <b64>``).

        Returns:
            A ``UserInfo`` if the header contains a recognised scheme with a
            non-empty credential, or ``None`` if the header is malformed or
            uses an unsupported scheme.

        Raises:
            ValueError: If a credential is present but cannot be decoded
                (e.g. invalid JWT structure, malformed base64).
        """
        parts: list[str] = auth_header.strip().split(maxsplit=1)
        if len(parts) != 2:
            return None

        scheme: str = parts[0].lower()
        credential: str = parts[1]
        if not credential:
            return None

        if scheme == "bearer":
            if credential.count(".") == 2:
                claims: dict[str, typing.Any] = decode_jwt_claims_unverified(credential)
                return cls._user_info_from_jwt(claims)
            return UserInfo._from_api_key(credential)

        if scheme == "basic":
            return cls._user_info_from_basic_auth(credential)

        return None

    @staticmethod
    def _from_auth_payload(payload: AuthPayload) -> UserInfo:
        """Resolve a ``UserInfo`` from a WebSocket auth message payload.

        This is an identity resolver, not an authenticator.  JWTs are decoded
        with ``verify_signature=False`` to extract identity claims; API keys and
        basic credentials are mapped directly.  Clients should verify and
        authenticate credentials (e.g. via JWKS, OAuth flows, or other auth
        middleware) before sending them over a WebSocket auth message.

        Args:
            payload: Discriminated union of JWT, API key, or basic auth credentials.

        Returns:
            A ``UserInfo`` with a deterministic user ID.

        Raises:
            ValueError: If the payload cannot be resolved to a valid user identity.
        """
        if isinstance(payload, JwtAuthPayload):
            raw_token: str = payload.token.get_secret_value()
            claims: dict[str, typing.Any] = decode_jwt_claims_unverified(raw_token)
            return UserManager._user_info_from_jwt(claims)

        if isinstance(payload, ApiKeyAuthPayload):
            token_value: str = payload.token.get_secret_value()
            if not token_value:
                raise ValueError("API key token is empty")
            return UserInfo._from_api_key(token_value)

        if isinstance(payload, BasicAuthPayload):
            return UserInfo(basic_user=BasicUserInfo(
                username=payload.username,
                password=payload.password,
            ))

        typing.assert_never(payload)

    @staticmethod
    def _get_session_cookie(connection: Request | WebSocket) -> str | None:
        """Extract the ``nat-session`` cookie value from a Request or WebSocket."""
        from nat.runtime.session import SESSION_COOKIE_NAME

        if isinstance(connection, Request):
            cookies: dict[str, str] = dict(connection.cookies) if connection.cookies else {}
            return cookies.get(SESSION_COOKIE_NAME)

        if isinstance(connection, WebSocket) and hasattr(connection, "scope") and "headers" in connection.scope:
            for name, value in connection.scope.get("headers", []):
                try:
                    name_str: str = name.decode("utf-8").lower()
                    value_str: str = value.decode("utf-8")
                except Exception:
                    logger.debug("Failed to decode WebSocket header, skipping", exc_info=True)
                    continue

                if name_str == "cookie":
                    for key, morsel in SimpleCookie(value_str).items():
                        if key == SESSION_COOKIE_NAME:
                            return morsel.value
        return None

    @staticmethod
    def _get_api_key_header(connection: Request | WebSocket) -> str | None:
        """Extract the ``X-API-Key`` header value from a connection."""
        if isinstance(connection, Request):
            return connection.headers.get("x-api-key")
        if isinstance(connection, WebSocket) and hasattr(connection, "scope") and "headers" in connection.scope:
            for name, value in connection.scope.get("headers", []):
                try:
                    name_str: str = name.decode("utf-8").lower()
                    value_str: str = value.decode("utf-8")
                except Exception:
                    continue
                if name_str == "x-api-key":
                    return value_str
        return None

    @staticmethod
    def _get_auth_header(connection: Request | WebSocket) -> str | None:
        """Extract the raw ``Authorization`` header value from a connection."""
        if isinstance(connection, Request):
            return connection.headers.get("authorization")
        if isinstance(connection, WebSocket) and hasattr(connection, "scope") and "headers" in connection.scope:
            for name, value in connection.scope.get("headers", []):
                try:
                    name_str: str = name.decode("utf-8").lower()
                    value_str: str = value.decode("utf-8")
                except Exception:
                    continue
                if name_str == "authorization":
                    return value_str
        return None

    @staticmethod
    def _user_info_from_session_cookie(cookie_value: str) -> UserInfo:
        """Build a ``UserInfo`` from a session cookie value."""
        return UserInfo._from_session_cookie(cookie_value)

    @staticmethod
    def _user_info_from_jwt(claims: dict[str, typing.Any]) -> UserInfo:
        """Build a ``UserInfo`` from decoded JWT claims.

        Registered claims (``sub``, ``iss``, ``aud``, ``exp``, ``iat``) follow
        RFC 7519.  Identity claims (``email``, ``preferred_username``, ``name``)
        follow OpenID Connect Core 1.0 Section 5.1.  ``sub`` is preferred as
        the stable identifier per RFC 7519 Section 4.1.2.

        Raises:
            ValueError: If the JWT contains no usable identity claim.
        """
        has_identity: bool = any(
            isinstance(claims.get(k), str) and claims.get(k, "").strip()
            for k in ("sub", "email", "preferred_username"))
        if not has_identity:
            raise ValueError("JWT contains no usable identity claim (sub, email, preferred_username)")

        given_name: str | None = (claims.get("given_name") if isinstance(claims.get("given_name"), str) else None)
        family_name: str | None = (claims.get("family_name") if isinstance(claims.get("family_name"), str) else None)
        if not given_name and not family_name:
            raw_name: typing.Any = claims.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                name_parts: list[str] = raw_name.strip().split(maxsplit=1)
                given_name = name_parts[0]
                family_name = name_parts[1] if len(name_parts) > 1 else None

        raw_scope: typing.Any = claims.get("scope")
        scopes: list[str] = raw_scope.split() if isinstance(raw_scope, str) else []

        raw_roles: typing.Any = claims.get("roles")
        if not isinstance(raw_roles, list):
            realm_access: typing.Any = claims.get("realm_access")
            if isinstance(realm_access, dict):
                raw_roles = realm_access.get("roles")
        roles: list[str] = raw_roles if isinstance(raw_roles, list) else []

        raw_groups: typing.Any = claims.get("groups")
        groups: list[str] = raw_groups if isinstance(raw_groups, list) else []

        raw_aud: typing.Any = claims.get("aud")
        audience: list[str] | None = None
        if isinstance(raw_aud, list):
            audience = raw_aud
        elif isinstance(raw_aud, str):
            audience = [raw_aud]

        jwt_info: JwtUserInfo = JwtUserInfo(
            given_name=given_name,
            family_name=family_name,
            email=claims.get("email") if isinstance(claims.get("email"), str) else None,
            preferred_username=(claims.get("preferred_username")
                                if isinstance(claims.get("preferred_username"), str) else None),
            roles=roles,
            groups=groups,
            scopes=scopes,
            issuer=claims.get("iss") if isinstance(claims.get("iss"), str) else None,
            subject=claims.get("sub") if isinstance(claims.get("sub"), str) else None,
            audience=audience,
            expires_at=claims.get("exp") if isinstance(claims.get("exp"), int) else None,
            issued_at=claims.get("iat") if isinstance(claims.get("iat"), int) else None,
            client_id=(claims.get("azp") or claims.get("client_id")
                       if isinstance(claims.get("azp"), str) or isinstance(claims.get("client_id"), str) else None),
            claims=claims,
        )
        return UserInfo._from_jwt(jwt_info)

    @staticmethod
    def _user_info_from_basic_auth(b64_credential: str) -> UserInfo:
        """Build a ``UserInfo`` from a base64-encoded Basic Auth credential.

        Args:
            b64_credential: The base64-encoded ``username:password`` string.

        Raises:
            ValueError: If the credential cannot be decoded or is malformed.
        """
        try:
            decoded: str = base64.b64decode(b64_credential).decode("utf-8")
        except Exception as exc:
            raise ValueError(f"Failed to decode Basic auth credential: {exc}") from exc

        if ":" not in decoded:
            raise ValueError("Basic auth credential must contain a colon separator (username:password)")

        username: str
        password: str
        username, _, password = decoded.partition(":")
        if not username:
            raise ValueError("Basic auth username must not be empty")

        return UserInfo(basic_user=BasicUserInfo(username=username, password=SecretStr(password)))
