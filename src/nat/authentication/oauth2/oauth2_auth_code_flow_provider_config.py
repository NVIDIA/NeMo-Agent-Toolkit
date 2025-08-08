# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pydantic import Field

from aiq.data_models.authentication import AuthProviderBaseConfig


class OAuth2AuthCodeFlowProviderConfig(AuthProviderBaseConfig, name="oauth2_auth_code_flow"):

    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    client_secret: str = Field(description="The secret associated with the client_id.")
    authorization_url: str = Field(description="The authorization URL for OAuth 2.0 authentication.")
    token_url: str = Field(description="The token URL for OAuth 2.0 authentication.")
    token_endpoint_auth_method: str | None = Field(
        description=("The authentication method for the token endpoint. "
                     "Usually one of `client_secret_post` or `client_secret_basic`."),
        default=None)
    redirect_uri: str = Field(description="The redirect URI for OAuth 2.0 authentication. Must match the registered "
                              "redirect URI with the OAuth provider.")
    scopes: list[str] = Field(description="The scopes for OAuth 2.0 authentication.", default_factory=list)
    use_pkce: bool = Field(default=False,
                           description="Whether to use PKCE (Proof Key for Code Exchange) in the OAuth 2.0 flow.")

    authorization_kwargs: dict[str, str] | None = Field(description=("Additional keyword arguments for the "
                                                                     "authorization request."),
                                                        default=None)
