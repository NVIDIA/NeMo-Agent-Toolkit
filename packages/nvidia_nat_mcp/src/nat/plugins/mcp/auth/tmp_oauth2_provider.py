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

# This is a temporary OAuth2 provider that is used to authenticate the MCP server.
# This is intended to be used until the NAT OAuth2 provider is updated to support MCP server authentication


from pydantic import SecretStr

from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.authentication import AuthResult
from nat.authentication.interfaces import AuthProviderBase
from nat.plugins.mcp.auth.mcp_flow_handler import MCPAuthenticationFlowHandler

class TmpOAuth2AuthCodeFlowProvider(AuthProviderBase[OAuth2AuthCodeFlowProviderConfig]):

    def __init__(self, config: OAuth2AuthCodeFlowProviderConfig):
        super().__init__(config)
        self._authenticated_tokens: dict[str, AuthResult] = {}
        self._flow_handler = MCPAuthenticationFlowHandler()
        self._auth_callback  = None

        def authenticate(self, user_id: str | None = None) -> AuthResult:

            # if tokens exist for the user just return them
            if user_id and user_id in self._authenticated_tokens:
                return self._authenticated_tokens[user_id]

            authenticated_context: AuthenticatedContext = self._flow_handler.authenticate(self.config, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)

            auth_header = authenticated_context.headers.get("Authorization", "")
            if not auth_header:
                raise RuntimeError("Invalid Authorization header")

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
