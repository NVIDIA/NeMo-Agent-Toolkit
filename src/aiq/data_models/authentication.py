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

from pydantic import Field

from .common import BaseModelRegistryTag
from .common import TypedBaseModel


class AuthenticationBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    pass


AuthenticationBaseConfigT = typing.TypeVar("AuthenticationBaseConfigT", bound=AuthenticationBaseConfig)


class OAuth2Config(AuthenticationBaseConfig):
    """
    OAuth 2.0 authentication configuration model.
    """
    type: typing.Literal["oauth2"] = Field(alias="_type", default="oauth2", description="OAuth 2.0 Config.")
    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    audience: str = Field(description="The audience for OAuth 2.0 authentication.")
    scope: list[str] = Field(description="The scope for OAuth 2.0 authentication.")
    access_token: str | None = Field(default=None, description="The access token for OAuth 2.0 authentication.")


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
