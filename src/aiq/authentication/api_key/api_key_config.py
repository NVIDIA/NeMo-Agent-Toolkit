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
from pydantic import field_validator

from aiq.authentication.exceptions.api_key_exceptions import APIKeyFieldError
from aiq.authentication.exceptions.api_key_exceptions import HeaderNameFieldError
from aiq.authentication.exceptions.api_key_exceptions import HeaderPrefixFieldError
from aiq.builder.authentication import AuthenticationProviderInfo
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_authentication_provider
from aiq.data_models.authentication import AuthenticationBaseConfig


class APIKeyConfig(AuthenticationBaseConfig, name="api_key"):  # TODO EE: Investigate.
    """
    API Key authentication configuration model.

    Supports all standard OpenAPI 3.0 authentication schemes via builder logic,
    based on the combination of HTTPAuthScheme.

    HTTPAuthScheme:
        - basic
        - bearer
        - digest
        - cookie
        - oauth2
        - openidconnect
        - custom

    CredentialLocation:
        - header
        - query
        - cookie
        - body
        - custom

    Mappings:
        - Basic (header): uses `email` and `password`
        - Bearer (header): uses `raw_key` (token) with optional prefix
        - API Key (header): uses `raw_key`
        - API Key (query): uses `raw_key`
        - API Key (cookie): uses `raw_key`
        - Cookie (cookie): uses `raw_key` or `custom`
        - OAuth2 (header): uses `raw_key` (access token)
        - OpenID Connect (header): uses `raw_key` (ID token)
        - Custom (any location): uses `custom`
    """

    raw_key: str = Field(
        description=("Raw API token or credential to be injected into the request. Used for 'bearer', "
                     "'apikey', 'oauth2', and other schemes depending on the configured credential location."))

    header_name: str = Field(
        description="The HTTP header corresponding to the API provider. i.e. 'Authorization', X-API-Key.")
    header_prefix: str = Field(
        description="The HTTP header prefix corresponding to the API provider. i.e 'Bearer', 'JWT'.")

    custom: typing.Any | None = Field(
        description=("Custom authentication configuration for non-standard configuration."
                     "Used when the HTTP scheme is 'custom' or the credential source is 'custom'."),
        default=None)

    @field_validator('raw_key')
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        """
        Validate api_key is non-empty, does not contain whitespace, and is sufficiently long.
        """
        if not value:
            raise APIKeyFieldError('value_missing', 'api_key is required for authentication.')

        if len(value.strip()) != len(value):
            raise APIKeyFieldError('whitespace_found', 'api_key must not have leading/trailing whitespace.')

        if ' ' in value:
            raise APIKeyFieldError('whitespace_found', 'api_key must not contain whitespace.')

        if len(value) < 16:
            raise APIKeyFieldError('too_short',
                                   'api_key must be at least 16 characters long for sufficient entropy. Got: {length}',
                                   {'length': len(value)})
        return value

    @field_validator('header_name')
    @classmethod
    def validate_header_name(cls, value: str) -> str:
        """
        Validate header_name is a valid HTTP header name.
        """
        if not value:
            raise HeaderNameFieldError('value_missing', 'header_name is required.')

        if not value.isascii() or not value.replace("-", "").isalnum():
            raise HeaderNameFieldError('invalid_format',
                                       'header_name must be ASCII and consist of alphanumeric characters or hyphens.')
        return value

    @field_validator('header_prefix')
    @classmethod
    def validate_header_prefix(cls, value: str) -> str:
        """
        Validate header_prefix is standard format (e.g., 'Bearer', 'JWT') and does not contain whitespace.
        """
        if not value:
            raise HeaderPrefixFieldError('value_missing', 'header_prefix is required.')

        if ' ' in value:
            raise HeaderPrefixFieldError('contains_whitespace', 'header_prefix must not contain whitespace.')

        if not value.isalnum():
            raise HeaderPrefixFieldError('invalid_format', 'header_prefix must consist of alphanumeric characters.')
        return value


@register_authentication_provider(config_type=APIKeyConfig)
async def api_key(authentication_provider: APIKeyConfig, builder: Builder):

    yield AuthenticationProviderInfo(config=authentication_provider, description="API key authentication provider.")
