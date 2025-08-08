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
from datetime import datetime
from datetime import timezone
from enum import Enum

import httpx
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr

from nat.data_models.common import BaseModelRegistryTag
from nat.data_models.common import TypedBaseModel


class AuthProviderBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    """
    Base configuration for authentication providers.
    """

    # Default, forbid extra fields to prevent unexpected behavior or miss typed options
    model_config = ConfigDict(extra="forbid")


AuthProviderBaseConfigT = typing.TypeVar("AuthProviderBaseConfigT", bound=AuthProviderBaseConfig)


class CredentialLocation(str, Enum):
    """
    Enum representing the location of credentials in an HTTP request.
    """
    HEADER = "header"
    QUERY = "query"
    COOKIE = "cookie"
    BODY = "body"


class AuthFlowType(str, Enum):
    """
    Enum representing different types of authentication flows.
    """
    API_KEY = "api_key"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"
    OAUTH2_AUTHORIZATION_CODE = "oauth2_auth_code_flow"
    OAUTH2_PASSWORD = "oauth2_password"
    OAUTH2_DEVICE_CODE = "oauth2_device_code"
    HTTP_BASIC = "http_basic"
    NONE = "none"


class AuthenticatedContext(BaseModel):
    """
    Represents an authenticated context for making requests.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    headers: dict[str, str] | httpx.Headers | None = Field(default=None,
                                                           description="HTTP headers used for authentication.")
    query_params: dict[str, str] | httpx.QueryParams | None = Field(
        default=None, description="Query parameters used for authentication.")
    cookies: dict[str, str] | httpx.Cookies | None = Field(default=None, description="Cookies used for authentication.")
    body: dict[str, str] | None = Field(default=None, description="Authenticated Body value, if applicable.")
    metadata: dict[str, typing.Any] | None = Field(default=None, description="Additional metadata for the request.")


class HeaderAuthScheme(str, Enum):
    """
    Enum representing different header authentication schemes.
    """
    BEARER = "Bearer"
    X_API_KEY = "X-API-Key"
    BASIC = "Basic"
    CUSTOM = "Custom"


class HTTPMethod(str, Enum):
    """
    Enum representing HTTP methods used in requests.
    """
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class CredentialKind(str, Enum):
    """
    Enum representing different kinds of credentials used for authentication.
    """
    HEADER = "header"
    QUERY = "query"
    COOKIE = "cookie"
    BASIC = "basic_auth"
    BEARER = "bearer_token"


class _CredBase(BaseModel):
    """
    Base class for credentials used in authentication.
    """
    kind: CredentialKind
    model_config = ConfigDict(extra="forbid")


class HeaderCred(_CredBase):
    """
    Represents a credential that is sent in the HTTP header.
    """
    kind: typing.Literal[CredentialKind.HEADER] = CredentialKind.HEADER
    name: str
    value: SecretStr


class QueryCred(_CredBase):
    """
    Represents a credential that is sent as a query parameter in the URL.
    """
    kind: typing.Literal[CredentialKind.QUERY] = CredentialKind.QUERY
    name: str
    value: SecretStr


class CookieCred(_CredBase):
    """
    Represents a credential that is sent as a cookie in the HTTP request.
    """
    kind: typing.Literal[CredentialKind.COOKIE] = CredentialKind.COOKIE
    name: str
    value: SecretStr


class BasicAuthCred(_CredBase):
    """
    Represents credentials for HTTP Basic Authentication.
    """
    kind: typing.Literal[CredentialKind.BASIC] = CredentialKind.BASIC
    username: SecretStr
    password: SecretStr


class BearerTokenCred(_CredBase):
    """
    Represents a credential for Bearer Token Authentication.
    """
    kind: typing.Literal[CredentialKind.BEARER] = CredentialKind.BEARER
    token: SecretStr
    scheme: str = "Bearer"
    header_name: str = "Authorization"


Credential = typing.Annotated[
    typing.Union[
        HeaderCred,
        QueryCred,
        CookieCred,
        BasicAuthCred,
        BearerTokenCred,
    ],
    Field(discriminator="kind"),
]


class AuthResult(BaseModel):
    """
    Represents the result of an authentication process.
    """
    credentials: list[Credential] = Field(default_factory=list,
                                          description="List of credentials used for authentication.")
    token_expires_at: datetime | None = Field(default=None, description="Expiration time of the token, if applicable.")
    raw: dict[str, typing.Any] = Field(default_factory=dict,
                                       description="Raw response data from the authentication process.")

    model_config = ConfigDict(extra="forbid")

    def is_expired(self) -> bool:
        """
        Checks if the authentication token has expired.
        """
        return bool(self.token_expires_at and datetime.now(timezone.utc) >= self.token_expires_at)

    def as_requests_kwargs(self) -> dict[str, typing.Any]:
        """
        Converts the authentication credentials into a format suitable for use with the `httpx` library.
        """
        kw: dict[str, typing.Any] = {"headers": {}, "params": {}, "cookies": {}}

        for cred in self.credentials:
            match cred:
                case HeaderCred():
                    kw["headers"][cred.name] = cred.value.get_secret_value()
                case QueryCred():
                    kw["params"][cred.name] = cred.value.get_secret_value()
                case CookieCred():
                    kw["cookies"][cred.name] = cred.value.get_secret_value()
                case BearerTokenCred():
                    kw["headers"][cred.header_name] = (f"{cred.scheme} {cred.token.get_secret_value()}")
                case BasicAuthCred():
                    kw["auth"] = (
                        cred.username.get_secret_value(),
                        cred.password.get_secret_value(),
                    )

        return kw

    def attach(self, target_kwargs: dict[str, typing.Any]) -> None:
        """
        Attaches the authentication credentials to the target request kwargs.
        """
        merged = self.as_requests_kwargs()
        for k, v in merged.items():
            if isinstance(v, dict):
                target_kwargs.setdefault(k, {}).update(v)
            else:
                target_kwargs[k] = v
