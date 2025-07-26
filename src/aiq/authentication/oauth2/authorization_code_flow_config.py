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
from pydantic.config import ConfigDict

from aiq.data_models.authentication import AuthenticationBaseConfig


class OAuth2AuthorizationCodeFlowConfig(AuthenticationBaseConfig, name="oauth2_authorization_code"):

    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    client_secret: str = Field(description="The secret associated with the client_id.")
    authorization_url: str = Field(description="The authorization URL for OAuth 2.0 authentication.")
    token_url: str = Field(description="The token URL for OAuth 2.0 authentication.")
    token_endpoint_auth_method: str | None = Field(description="The authentication method for the token endpoint.",
                                                   default=None)
    scopes: list[str] = Field(description="The space-delimited scopes for OAuth 2.0 authentication.",
                              default_factory=list)

    # Configuration for the local server that handles the redirect
    client_server_host: str = Field(default="localhost", description="Host for the local redirect server.")
    client_server_port: int = Field(default=8000, description="Port for the local redirect server.")
    redirect_path: str = Field(default="/auth/redirect",
                               description="Path for the local redirect server to handle the callback.")
    use_pkce: bool = Field(default=False,
                           description="Whether to use PKCE (Proof Key for Code Exchange) in the OAuth 2.0 flow.")

    @property
    def redirect_uri(self) -> str:
        return f"http://{self.client_server_host}:{self.client_server_port}{self.redirect_path}"
