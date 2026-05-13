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
"""Lightweight JWT extraction and decoding utilities for identity resolution (RFC 7519)."""

from __future__ import annotations

import typing

import jwt
from fastapi import WebSocket
from starlette.requests import Request

from nat.data_models.authentication import HeaderAuthScheme


def extract_bearer_token(connection: Request | WebSocket, *, header: str = "authorization") -> str | None:
    """Extract the raw Bearer token string from an HTTP header.

    Args:
        connection: The incoming Starlette ``Request`` or ``WebSocket``.
        header: Header name to read the Bearer token from (case-insensitive).

    Returns:
        The raw token string, or ``None`` if no valid Bearer token is present.
    """
    auth: str | None = None
    header_lower: str = header.lower()

    if isinstance(connection, Request):
        auth = connection.headers.get(header_lower)
    elif isinstance(connection, WebSocket) and hasattr(connection, "scope") and "headers" in connection.scope:
        for name, value in connection.scope.get("headers", []):
            try:
                name_str: str = name.decode("utf-8").lower()
                value_str: str = value.decode("utf-8")
            except Exception:
                continue
            if name_str == header_lower:
                auth = value_str
                break

    if not auth:
        return None

    parts: list[str] = auth.strip().split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != HeaderAuthScheme.BEARER.lower():
        return None

    return parts[1] or None


def decode_jwt_claims_unverified(token: str) -> dict[str, typing.Any]:
    """Decode JWT claims without signature verification (RFC 7519 Section 7.2).

    Intended for identity extraction only — callers are responsible for
    authenticating/verifying tokens via JWKS, OAuth flows, or other
    auth middleware before trusting the claims.

    Args:
        token: A raw JWT string (three dot-separated parts per RFC 7519 Section 3).

    Returns:
        The decoded claims dictionary.

    Raises:
        ValueError: If the token is empty, structurally malformed,
            or cannot be decoded.
    """
    if not token or token.count(".") != 2:
        raise ValueError("JWT token is empty or malformed (expected 3 dot-separated parts)")

    try:
        claims: dict[str, typing.Any] = jwt.decode(token, options={"verify_signature": False})
    except Exception as exc:
        raise ValueError(f"Failed to decode JWT: {exc}") from exc

    return claims
