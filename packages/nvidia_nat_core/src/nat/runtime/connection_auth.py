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
"""Helpers for extracting auth headers, cookies, and resolving user_id from HTTP/WebSocket connections."""

import typing
from http.cookies import SimpleCookie

import jwt
from fastapi import WebSocket
from starlette.requests import HTTPConnection
from starlette.requests import Request


def get_auth_and_cookies_from_connection(connection: HTTPConnection, ) -> tuple[str | None, dict[str, str]]:
    """
    Extract Authorization header value and cookies dict from Request or WebSocket.

    Returns:
        (auth_header_value, cookies_dict). auth_header_value is the raw header
        (e.g. "Bearer <token>"). cookies_dict has cookie names as keys.
    """
    if isinstance(connection, Request):
        auth = connection.headers.get("authorization") or connection.headers.get("Authorization")
        cookies = dict(connection.cookies) if connection.cookies else {}
        return (auth, cookies)
    # WebSocket: ASGI scope["headers"] is a list of (bytes, bytes), no .cookies/.headers API
    if isinstance(connection, WebSocket) and hasattr(connection, "scope") and "headers" in connection.scope:
        auth = None
        cookie_header = None
        for name, value in connection.scope.get("headers", []):
            try:
                name_str = name.decode("utf-8").lower()
                value_str = value.decode("utf-8")
            except Exception:
                continue
            if name_str == "authorization":
                auth = value_str
            elif name_str == "cookie":
                cookie_header = value_str
        cookies = {}
        if cookie_header:
            for key, morsel in SimpleCookie(cookie_header).items():
                cookies[key] = morsel.value
        return (auth, cookies)
    return (None, {})


def decode_jwt_payload_unverified(token: str) -> dict[str, typing.Any] | None:
    """
    Decode JWT payload without verification (PyJWT).
    Used only to extract user identity claims (name, email, sub) for routing.
    """
    if not token or token.count(".") != 2:
        return None
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def resolve_user_id(
    auth_header_value: str | None,
    cookies: dict[str, str],
) -> str | None:
    """
    Resolve user_id: 1) nat-session cookie (preserves existing behavior),
    2) from JWT in Authorization header (name/email/sub) when cookie is not set.
    """
    if cookies.get("nat-session"):
        return cookies.get("nat-session")
    if auth_header_value:
        parts = auth_header_value.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            payload = decode_jwt_payload_unverified(parts[1])
            if payload:
                for claim in ("name", "email", "preferred_username", "sub"):
                    val = payload.get(claim)
                    if val and isinstance(val, str) and val.strip():
                        return val.strip()
    return None
