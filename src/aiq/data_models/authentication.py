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

import typing
from enum import Enum

import httpx
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .common import BaseModelRegistryTag
from .common import TypedBaseModel


class AuthenticationBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    pass


AuthenticationBaseConfigT = typing.TypeVar("AuthenticationBaseConfigT", bound=AuthenticationBaseConfig)


class CredentialLocation(str, Enum):
    HEADER = "header"
    QUERY = "query"
    COOKIE = "cookie"
    BODY = "body"


class AuthFlowType(str, Enum):
    API_KEY = "api_key"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"
    OAUTH2_AUTHORIZATION_CODE = "oauth2_authorization_code"
    OAUTH2_PASSWORD = "oauth2_password"
    OAUTH2_DEVICE_CODE = "oauth2_device_code"
    HTTP_BASIC = "http_basic"
    NONE = "none"


class AuthenticatedContext(BaseModel):
    """
    Represents the context used to authenticate an API request. This can include
    headers, query parameters, cookies, or other metadata used for authentication.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    headers: dict[str, str] | httpx.Headers | None = Field(default=None,
                                                           description="HTTP headers used for authentication.")
    query_params: dict[str, str] | httpx.QueryParams | None = Field(
        default=None, description="Query parameters used for authentication.")
    cookies: dict[str, str] | httpx.Cookies | None = Field(default=None, description="Cookies used for authentication.")
    body: dict[str, str] | None = Field(default=None, description="Authenticated Body value, if applicable.")


class HeaderAuthScheme(str, Enum):
    BEARER = "Bearer"
    X_API_KEY = "X-API-Key"
    BASIC = "Basic"
    CUSTOM = "custom"


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


class OAuth2AuthorizationQueryParams(BaseModel):
    """
    Base Query Params for initiating authorization request (used in redirect-based flows).
    """
    model_config = ConfigDict(extra="forbid")
    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    audience: str = Field(description="The resource server the token is intended for.")
    state: str = Field(description="Opaque value to maintain state between request and callback.")
    redirect_uri: str = Field(description="Registered redirect URI for callback.")
    response_type: str = Field(description="The type of response expected, e.g., 'code' or 'token'.")
    scope: str = Field(description="Space-delimited list of requested scopes.")
    prompt: str = Field(description="Type of consent prompt to show the user.")
    code_challenge: str | None = Field(description="PKCE code challenge.", default=None)
    code_challenge_method: str | None = Field(description="PKCE challenge method (e.g., 'S256').", default=None)


class OAuth2TokenRequest(BaseModel):
    """
    Base OAuth 2.0 request body for exchanging access codes for access tokens.
    """
    model_config = ConfigDict(extra="forbid")
    grant_type: str = Field(
        description="OAuth 2.0 grant type identifier (e.g., 'authorization_code', 'client_credentials').")
    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    client_secret: str = Field(description="The client secret for OAuth 2.0 authentication.")
    redirect_uri: str = Field(description="Registered redirect uri.")
    code: str = Field(
        description="Authorization code to be exchanged for an access token (if using authorization_code flow). "
        "NOTE: Field must be named code or a HTTP 400 status error will be thrown. ")
    refresh_token: str | None = Field(description="Refresh token (if using refresh_token flow).", default=None)
    device_code: str | None = Field(description="Device code (if using device_code flow).", default=None)
    username: str | None = Field(description="Resource owner username (if using password flow).", default=None)
    password: str | None = Field(description="Resource owner password (if using password flow).", default=None)
    scope: str | None = Field(description="Optional scopes for client_credentials/password flows.", default=None)
    code_verifier: str | None = Field(description="PKCE code verifier (for authorization_code + PKCE).", default=None)


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
