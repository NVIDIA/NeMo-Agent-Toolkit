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
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    headers: dict[str, str] | httpx.Headers | None = Field(default=None,
                                                           description="HTTP headers used for authentication.")
    query_params: dict[str, str] | httpx.QueryParams | None = Field(
        default=None, description="Query parameters used for authentication.")
    cookies: dict[str, str] | httpx.Cookies | None = Field(default=None, description="Cookies used for authentication.")
    body: dict[str, str] | None = Field(default=None, description="Authenticated Body value, if applicable.")
    metadata: dict[str, typing.Any] | None = Field(default=None, description="Additional metadata for the request.")


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


class AuthenticationEndpoint(str, Enum):
    REDIRECT_URI = "/redirect"
    PROMPT_REDIRECT_URI = "/prompt-uri"


class CredentialKind(str, Enum):
    HEADER = "header"
    QUERY = "query"
    COOKIE = "cookie"
    BASIC = "basic_auth"  # (user, pass) tuple
    BEARER = "bearer_token"  # Authorization header


class _CredBase(BaseModel):
    kind: CredentialKind
    model_config = ConfigDict(extra="forbid")


class HeaderCred(_CredBase):
    kind: typing.Literal[CredentialKind.HEADER] = CredentialKind.HEADER
    name: str
    value: SecretStr


class QueryCred(_CredBase):
    kind: typing.Literal[CredentialKind.QUERY] = CredentialKind.QUERY
    name: str
    value: SecretStr


class CookieCred(_CredBase):
    kind: typing.Literal[CredentialKind.COOKIE] = CredentialKind.COOKIE
    name: str
    value: SecretStr


class BasicAuthCred(_CredBase):
    kind: typing.Literal[CredentialKind.BASIC] = CredentialKind.BASIC
    username: SecretStr
    password: SecretStr


class BearerTokenCred(_CredBase):
    kind: typing.Literal[CredentialKind.BEARER] = CredentialKind.BEARER
    token: SecretStr
    scheme: str = "Bearer"  # override to "Token", etc.
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
    credentials: list[Credential]
    token_expires_at: datetime | None = None
    raw: dict[str, typing.Any] = {}  # idP / debug blob

    model_config = ConfigDict(extra="forbid")

    def is_expired(self) -> bool:
        return bool(self.token_expires_at and datetime.now(timezone.utc) >= self.token_expires_at)

    def as_requests_kwargs(self) -> dict[str, typing.Any]:
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
        merged = self.as_requests_kwargs()
        for k, v in merged.items():
            if isinstance(v, dict):
                target_kwargs.setdefault(k, {}).update(v)
            else:
                target_kwargs[k] = v
