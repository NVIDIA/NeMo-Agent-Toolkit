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

import secrets
from datetime import datetime
from enum import Enum

from pydantic import Field

from aiq.builder.authentication import AuthenticationProviderInfo
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_authentication_provider
from aiq.data_models.authentication import AuthenticationBaseConfig


class ConsentPromptMode(str, Enum):
    BROWSER = "browser"
    FRONTEND = "frontend"


class AuthCodeGrantConfig(AuthenticationBaseConfig, name="oauth2_authorization_code_grant"):
    """
    OAuth 2.0 authorization code grant authentication configuration model.
    """
    client_server_url: str = Field(description="The base url of the API server instance. "
                                   "This is needed to properly construct the redirect uri i.e: http://localhost:8000")
    authorization_url: str = Field(description="The base url to the authorization server in which authorization "
                                   "request are made to receive access codes..")
    authorization_token_url: str = Field(
        description="The base url to the authorization token server in which access codes "
        "are exchanged for access tokens.")
    consent_prompt_mode: ConsentPromptMode = Field(
        default=ConsentPromptMode.BROWSER,
        description="Specifies how the application handles the OAuth 2.0 consent prompt. "
        "Options are 'browser' to open the system's default browser for login, "
        "or 'frontend' to store the login url retrievable via POST /auth/prompt-uri")
    consent_prompt_key: str = Field(description="The key used to retrieve the consent prompt location header, "
                                    " triggering the browser to complete the OAuth process from the front end.",
                                    frozen=True)
    client_secret: str = Field(description="The client secret for OAuth 2.0 authentication.")
    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    audience: str = Field(description="The audience for OAuth 2.0 authentication.")
    scope: list[str] = Field(description="The scope for OAuth 2.0 authentication.")
    state: str = Field(default=secrets.token_urlsafe(nbytes=16),
                       description="A URL-safe base64 format 16 byte random string",
                       frozen=True)
    access_token: str | None = Field(default=None, description="The access token for OAuth 2.0 authentication.")
    access_token_expires_in: datetime | None = Field(default=None,
                                                     description="Expiry time of the access token in seconds.")
    refresh_token: str | None = Field(
        default=None, description="The refresh token for OAuth 2.0 authentication used to obtain a new access token.")
    consent_prompt_location_url: str | None = Field(
        default=None,
        description="302 redirect Location header to which the client will be redirected to the consent prompt.")


@register_authentication_provider(config_type=AuthCodeGrantConfig)
async def oauth2_authorization_code_grant(authentication_provider: AuthCodeGrantConfig, builder: Builder):

    yield AuthenticationProviderInfo(config=authentication_provider,
                                     description="OAuth 2.0 authorization code grant authentication provider.")
