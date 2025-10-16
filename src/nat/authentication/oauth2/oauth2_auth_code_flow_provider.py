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
from datetime import UTC
from datetime import datetime

import httpx
from authlib.integrations.httpx_client import OAuth2Client as AuthlibOAuth2Client
from pydantic import SecretStr

from nat.authentication.interfaces import AuthProviderBase
from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.builder.context import Context
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred

logger = logging.getLogger(__name__)


class OAuth2AuthCodeFlowProvider(AuthProviderBase[OAuth2AuthCodeFlowProviderConfig]):

    def __init__(self, config: OAuth2AuthCodeFlowProviderConfig, token_storage=None):
        super().__init__(config)
        self._auth_callback = None
        # Always use token storage - defaults to in-memory if not provided
        if token_storage is None:
            from nat.plugins.mcp.auth.token_storage import InMemoryTokenStorage
            self._token_storage = InMemoryTokenStorage()
        else:
            self._token_storage = token_storage

    async def _attempt_token_refresh(self, user_id: str, auth_result: AuthResult) -> AuthResult | None:
        refresh_token = auth_result.raw.get("refresh_token")
        if not isinstance(refresh_token, str):
            return None

        try:
            with AuthlibOAuth2Client(
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret,
            ) as client:
                new_token_data = client.refresh_token(self.config.token_url, refresh_token=refresh_token)

                expires_at_ts = new_token_data.get("expires_at")
                new_expires_at = datetime.fromtimestamp(expires_at_ts, tz=UTC) if expires_at_ts else None

            new_auth_result = AuthResult(
                credentials=[BearerTokenCred(token=SecretStr(new_token_data["access_token"]))],
                token_expires_at=new_expires_at,
                raw=new_token_data,
            )

            await self._token_storage.store(user_id, new_auth_result)
        except httpx.HTTPStatusError:
            return None
        except httpx.RequestError:
            return None
        except Exception:
            # On any other failure, we'll fall back to the full auth flow.
            return None

        return new_auth_result

    def _set_custom_auth_callback(self,
                                  auth_callback: Callable[[OAuth2AuthCodeFlowProviderConfig, AuthFlowType],
                                                          Awaitable[AuthenticatedContext]]):
        self._auth_callback = auth_callback

    async def authenticate(self, user_id: str | None = None, **kwargs) -> AuthResult:
        context = Context.get()
        if user_id is None and hasattr(context, "metadata") and hasattr(
                context.metadata, "cookies") and context.metadata.cookies is not None:
            session_id = context.metadata.cookies.get("nat-session", None)
            if not session_id:
                raise RuntimeError("Authentication failed. No session ID found. Cannot identify user.")

            user_id = session_id

        if user_id:
            # Try to retrieve from token storage
            auth_result = await self._token_storage.retrieve(user_id)

            if auth_result:
                if not auth_result.is_expired():
                    return auth_result

                refreshed_auth_result = await self._attempt_token_refresh(user_id, auth_result)
                if refreshed_auth_result:
                    return refreshed_auth_result

        # Try getting callback from the context if that's not set, use the default callback
        try:
            auth_callback = Context.get().user_auth_callback
        except RuntimeError:
            auth_callback = self._auth_callback

        if not auth_callback:
            raise RuntimeError("Authentication callback not set on Context.")

        try:
            authenticated_context = await auth_callback(self.config, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)
        except Exception as e:
            raise RuntimeError(f"Authentication callback failed: {e}") from e

        headers = authenticated_context.headers or {}
        auth_header = headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise RuntimeError("Invalid Authorization header")

        token = auth_header.split(" ")[1]

        # Safely access metadata
        metadata = authenticated_context.metadata or {}
        auth_result = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr(token))],
            token_expires_at=metadata.get("expires_at"),
            raw=metadata.get("raw_token") or {},
        )

        if user_id:
            await self._token_storage.store(user_id, auth_result)

        return auth_result
