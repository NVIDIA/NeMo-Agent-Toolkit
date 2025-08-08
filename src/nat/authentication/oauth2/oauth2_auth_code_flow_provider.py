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

from datetime import datetime
from datetime import timezone

from authlib.integrations.httpx_client import OAuth2Client as AuthlibOAuth2Client
from pydantic import SecretStr

from aiq.authentication.interfaces import AuthProviderBase
from aiq.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from aiq.builder.context import Context
from aiq.data_models.authentication import AuthFlowType
from aiq.data_models.authentication import AuthResult
from aiq.data_models.authentication import BearerTokenCred


class OAuth2AuthCodeFlowProvider(AuthProviderBase[OAuth2AuthCodeFlowProviderConfig]):

    def __init__(self, config: OAuth2AuthCodeFlowProviderConfig):
        super().__init__(config)
        self._authenticated_tokens: dict[str, AuthResult] = {}
        self._context = Context.get()

    async def _attempt_token_refresh(self, user_id: str, auth_result: AuthResult) -> AuthResult | None:
        refresh_token = auth_result.raw.get("refresh_token")
        if not isinstance(refresh_token, str):
            return None

        with AuthlibOAuth2Client(
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
        ) as client:
            try:
                new_token_data = client.refresh_token(self.config.token_url, refresh_token=refresh_token)
            except Exception:
                # On any failure, we'll fall back to the full auth flow.
                return None

        expires_at_ts = new_token_data.get("expires_at")
        new_expires_at = datetime.fromtimestamp(expires_at_ts, tz=timezone.utc) if expires_at_ts else None

        new_auth_result = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr(new_token_data["access_token"]))],
            token_expires_at=new_expires_at,
            raw=new_token_data,
        )

        self._authenticated_tokens[user_id] = new_auth_result

        return new_auth_result

    async def authenticate(self, user_id: str | None = None) -> AuthResult:
        if user_id is None and hasattr(Context.get(), "metadata") and hasattr(
                Context.get().metadata, "cookies") and Context.get().metadata.cookies is not None:
            session_id = Context.get().metadata.cookies.get("aiqtoolkit-session", None)
            if not session_id:
                raise RuntimeError("Authentication failed. No session ID found. Cannot identify user.")

            user_id = session_id

        if user_id and user_id in self._authenticated_tokens:
            auth_result = self._authenticated_tokens[user_id]
            if not auth_result.is_expired():
                return auth_result

            refreshed_auth_result = await self._attempt_token_refresh(user_id, auth_result)
            if refreshed_auth_result:
                return refreshed_auth_result

        auth_callback = self._context.user_auth_callback
        if not auth_callback:
            raise RuntimeError("Authentication callback not set on AIQContext.")

        try:
            authenticated_context = await auth_callback(self.config, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)
        except Exception as e:
            raise RuntimeError(f"Authentication callback failed: {e}") from e

        auth_header = authenticated_context.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise RuntimeError("Invalid Authorization header")

        token = auth_header.split(" ")[1]

        auth_result = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr(token))],
            token_expires_at=authenticated_context.metadata.get("expires_at"),
            raw=authenticated_context.metadata.get("raw_token"),
        )

        if user_id:
            self._authenticated_tokens[user_id] = auth_result

        return auth_result
