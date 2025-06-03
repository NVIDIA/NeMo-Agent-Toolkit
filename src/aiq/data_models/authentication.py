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
import typing
from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .common import BaseModelRegistryTag
from .common import TypedBaseModel


class AuthenticationBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    pass


class ExecutionMode(str, Enum):
    CONSOLE = "console"
    SERVER = "server"


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ConsentPromptMode(str, Enum):
    BROWSER = "browser"
    FRONTEND = "frontend"


class AuthenticationEndpoint(str, Enum):
    REDIRECT_URI = "/redirect"
    PROMPT_REDIRECT_URI = "/prompt-uri"


class PromptRedirectRequest(BaseModel):
    consent_prompt_key: str = Field(description="The key used to retrieve the consent prompt 302 redirect, "
                                    " triggering the browser to complete the OAuth process from the front end.")


class OAuth2AuthQueryParams(BaseModel):
    """
    OAuth 2.0 authorization request query parameters model.
    """
    audience: str = Field(description="The audience for OAuth 2.0 authentication.")
    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    state: str = Field(description="A URL-safe base64 format 16 byte random string")
    scope: str = Field(description="The scope for OAuth 2.0 authentication.")
    redirect_uri: str = Field(description="Registered redirect uri.")
    response_type: str = Field(description="Type of response the client expects from the authorization server.")
    prompt: str = Field(description="Specifies what type of user consent prompt")


class AccessCodeTokenRequest(BaseModel):
    """
    OAuth 2.0 request body for exchanging access codes for access tokens.
    """
    model_config = ConfigDict(extra="forbid")
    grant_type: str = Field(default="authorization_code", description="Authorization flow identifier.", frozen=True)
    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    client_secret: str = Field(description="The client secret for OAuth 2.0 authentication.")
    code: str = Field(description="Authorization code to be exchanged for an access token.")
    redirect_uri: str = Field(description="Registered redirect uri.")


class RefreshTokenRequest(BaseModel):
    """
    OAuth 2.0 request body for exchanging refresh tokens for access tokens.
    """
    model_config = ConfigDict(extra="forbid")
    grant_type: str = Field(default="refresh_token", description="Authorization flow identifier.", frozen=True)
    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    client_secret: str = Field(description="The client secret for OAuth 2.0 authentication.")
    refresh_token: str = Field(
        description="The refresh token for OAuth 2.0 authentication used to obtain a new access token.")


class OAuth2Config(AuthenticationBaseConfig):
    """
    OAuth 2.0 authentication configuration model.
    """
    type: typing.Literal["oauth2"] = Field(alias="_type", default="oauth2", description="OAuth 2.0 Config.")
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
        "or 'polling' to store the login url retrievable via GET /auth/prompt-uri")
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


class APIKeyConfig(AuthenticationBaseConfig):
    """
    API Key authentication configuration model.
    """
    type: typing.Literal["api_key"] = Field(alias="_type", default="api_key", description="API Key Config.")
    api_key: str = Field(description="The API key for authentication.")
    header_name: str = Field(
        description="The HTTP header corresponding to the API provider. i.e. 'Authorization', X-API-Key.")
    header_prefix: str = Field(
        description="The HTTP header prefix corresponding to the API provider. i.e 'Bearer', 'JWT'.")


AuthenticationProvider = typing.Annotated[OAuth2Config | APIKeyConfig, Field(discriminator="type")]
