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

from pydantic import SecretStr

from aiq.authentication.interfaces import AuthenticationClientBase
from aiq.builder.context import AIQContext
from aiq.data_models.authentication import AuthenticatedContext
from aiq.data_models.authentication import AuthenticationBaseConfig
from aiq.data_models.authentication import AuthFlowType
from aiq.data_models.authentication import AuthResult
from aiq.data_models.authentication import BasicAuthCred
from aiq.data_models.authentication import BearerTokenCred


class HTTPBasicAuthExchanger(AuthenticationClientBase):
    """
    Abstract base class for HTTP Basic Authentication exchangers.
    """

    def __init__(self, config: AuthenticationBaseConfig):
        """
        Initialize the HTTP Basic Auth Exchanger with the given configuration.
        Args:
            config: Configuration for the authentication process.
        """
        super().__init__(config)
        self._authenticated_tokens: dict[str, AuthResult] = {}
        self._context = AIQContext.get()

    async def authenticate(self, user_id: str | None = None) -> AuthResult:
        """
        Performs simple HTTP Authentication using the provided user ID.
        Args:
            user_id: User identifier for whom to perform authentication.

        Returns:
            AuthenticatedContext: The context containing authentication headers.
        """
        if user_id and user_id in self._authenticated_tokens:
            return self._authenticated_tokens[user_id]

        auth_callback = self._context.user_auth_callback

        try:
            auth_context: AuthenticatedContext = await auth_callback(self.config, AuthFlowType.HTTP_BASIC)
        except RuntimeError as e:
            raise RuntimeError(f"Authentication callback failed: {str(e)}. Did you forget to set a "
                               f"callback handler for your frontend?") from e

        basic_auth_credentials = BasicAuthCred(username=SecretStr(auth_context.metadata.get("username", "")),
                                               password=SecretStr(auth_context.metadata.get("password", "")))

        # Get the auth token from the headers of auth context
        bearer_token = auth_context.headers.get("Authorization", "").split(" ")[-1]
        if not bearer_token:
            raise RuntimeError("Authentication failed: No Authorization header found in the response.")

        bearer_token_cred = BearerTokenCred(token=SecretStr(bearer_token), scheme="Basic")

        auth_result = AuthResult(credentials=[basic_auth_credentials, bearer_token_cred])

        self._authenticated_tokens[user_id] = auth_result

        return auth_result
