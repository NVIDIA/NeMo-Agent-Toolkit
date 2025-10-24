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

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.context import Context
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BasicAuthCred
from nat.data_models.authentication import BearerTokenCred


class HTTPBasicAuthProvider(AuthProviderBase):
    """
    Abstract base class for HTTP Basic Authentication exchangers.
    """

    def __init__(self, config: AuthProviderBaseConfig):
        """
        Initialize the HTTP Basic Auth Exchanger with the given configuration.
        """
        super().__init__(config)

        self._authenticated_tokens: dict[str, AuthResult] = {}

    async def authenticate(self, user_id: str | None = None, **kwargs) -> AuthResult:
        """
        Performs simple HTTP Authentication using the provided user ID.
        """

        context = Context.get()

        if user_id is None and hasattr(context, "metadata") and hasattr(
                context.metadata, "cookies") and context.metadata.cookies is not None:
            session_id = context.metadata.cookies.get("nat-session", None)
            if not session_id:
                raise RuntimeError("Authentication failed. No session ID found. Cannot identify user.")

            user_id = session_id

        if user_id and user_id in self._authenticated_tokens:
            return self._authenticated_tokens[user_id]

        auth_callback = context.user_auth_callback

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
