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

import logging
import typing
from http.cookies import SimpleCookie

import jwt
from fastapi import WebSocket
from starlette.requests import Request

from nat.data_models.api_server import ApiKeyAuthPayload
from nat.data_models.api_server import AuthPayload
from nat.data_models.api_server import BasicAuthPayload
from nat.data_models.api_server import JwtAuthPayload
from nat.data_models.authentication import HeaderAuthScheme
from nat.data_models.user_info import BasicUserInfo
from nat.data_models.user_info import JwtUserInfo
from nat.data_models.user_info import UserInfo

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME: str = "nat-session"


class UserManager:
    """Stateless resolver that creates ``UserInfo`` from HTTP/WebSocket connections.

    All methods are class or static methods — no instance state is held.
    """

    @classmethod
    def extract_user_from_connection(cls, connection: Request | WebSocket) -> UserInfo | None:
        """Resolve an HTTP/WebSocket connection into a ``UserInfo``.

        Checks for a ``nat-session`` cookie first, then falls back to a
        JWT Bearer token in the ``Authorization`` header.

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

        claims: dict[str, typing.Any] | None = cls._get_jwt_claims(connection)
        if claims is not None:
            return cls._user_info_from_jwt(claims)

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
            if not raw_token or raw_token.count(".") != 2:
                raise ValueError("JWT token is empty or malformed (expected 3 dot-separated parts)")
            try:
                claims: dict[str, typing.Any] = jwt.decode(raw_token, options={"verify_signature": False})
            except Exception as exc:
                raise ValueError(f"Failed to decode JWT token: {exc}") from exc
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

        raise ValueError(f"Unsupported auth payload type: {type(payload).__name__}")

    @staticmethod
    def _get_session_cookie(connection: Request | WebSocket) -> str | None:
        """Extract the ``nat-session`` cookie value from a Request or WebSocket."""
        if isinstance(connection, Request):
            cookies: dict[str, str] = dict(connection.cookies) if connection.cookies else {}
            return cookies.get(SESSION_COOKIE_NAME)

        if isinstance(connection, WebSocket) and hasattr(connection, "scope") and "headers" in connection.scope:
            for name, value in connection.scope.get("headers", []):
                try:
                    name_str: str = name.decode("utf-8").lower()
                    value_str: str = value.decode("utf-8")
                except Exception:
                    continue
                if name_str == "cookie":
                    for key, morsel in SimpleCookie(value_str).items():
                        if key == SESSION_COOKIE_NAME:
                            return morsel.value
        return None

    @staticmethod
    def _get_jwt_claims(connection: Request | WebSocket) -> dict[str, typing.Any] | None:
        """Extract and decode a JWT Bearer token from the ``Authorization`` header.

        Args:
            connection: The incoming Starlette ``Request`` or ``WebSocket``.

        Returns:
            The decoded claims dict, or ``None`` if no ``Authorization``
            header is present.

        Raises:
            ValueError: If an ``Authorization`` header is present but the
                token cannot be decoded into valid JWT claims.
        """
        auth: str | None = None

        if isinstance(connection, Request):
            auth = connection.headers.get("authorization")
        elif isinstance(connection, WebSocket) and hasattr(connection, "scope") and "headers" in connection.scope:
            for name, value in connection.scope.get("headers", []):
                try:
                    name_str: str = name.decode("utf-8").lower()
                    value_str: str = value.decode("utf-8")
                except Exception:
                    continue
                if name_str == "authorization":
                    auth = value_str
                    break

        if not auth:
            return None

        parts: list[str] = auth.strip().split(maxsplit=1)
        if len(parts) != 2 or parts[0].lower() != HeaderAuthScheme.BEARER.lower():
            return None

        token: str = parts[1]
        if not token or token.count(".") != 2:
            raise ValueError("Bearer token is empty or malformed (expected 3 dot-separated parts)")

        try:
            claims: dict[str, typing.Any] = jwt.decode(token, options={"verify_signature": False})
        except Exception as exc:
            raise ValueError(f"Failed to decode JWT from Authorization header: {exc}") from exc

        return claims

    @staticmethod
    def _user_info_from_session_cookie(cookie_value: str) -> UserInfo:
        """Build a ``UserInfo`` from a session cookie value."""
        return UserInfo._from_session_cookie(cookie_value)

    @staticmethod
    def _user_info_from_jwt(claims: dict[str, typing.Any]) -> UserInfo:
        """Build a ``UserInfo`` from decoded JWT claims.

        Raises:
            ValueError: If the JWT contains no usable identity claim.
        """
        has_identity: bool = any(
            isinstance(claims.get(k), str) and claims.get(k, "").strip()
            for k in ("email", "preferred_username", "sub"))
        if not has_identity:
            raise ValueError("JWT contains no usable identity claim (email, preferred_username, sub)")

        first_name: str | None = (claims.get("given_name") if isinstance(claims.get("given_name"), str) else None)
        last_name: str | None = (claims.get("family_name") if isinstance(claims.get("family_name"), str) else None)
        if not first_name and not last_name:
            raw_name: typing.Any = claims.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                name_parts: list[str] = raw_name.strip().split(maxsplit=1)
                first_name = name_parts[0]
                last_name = name_parts[1] if len(name_parts) > 1 else None

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
            first_name=first_name,
            last_name=last_name,
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
