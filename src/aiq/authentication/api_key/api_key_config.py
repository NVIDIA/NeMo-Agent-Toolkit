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

import logging

from pydantic import Field
from pydantic import field_validator

from aiq.authentication.exceptions.api_key_exceptions import APIKeyFieldError
from aiq.authentication.exceptions.api_key_exceptions import HeaderNameFieldError
from aiq.authentication.exceptions.api_key_exceptions import HeaderPrefixFieldError
from aiq.builder.authentication import AuthenticationProviderInfo
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_authentication_provider
from aiq.data_models.authentication import AuthenticationBaseConfig

logger = logging.getLogger(__name__)


class APIKeyConfig(AuthenticationBaseConfig, name="api_key"):
    """
    API Key authentication configuration model.
    """

    raw_key: str = Field(description=("Raw API token or credential to be injected into the request parameter. "
                                      "Used for 'bearer','x-api-key','custom', and other schemes. "))

    header_name: str | None = Field(description="The HTTP header name that MUST be used in conjunction "
                                    "with the header_prefix when HeaderAuthScheme is CUSTOM.",
                                    default=None)
    header_prefix: str | None = Field(description="The HTTP header prefix that MUST be used in conjunction "
                                      "with the header_name when HeaderAuthScheme is CUSTOM.",
                                      default=None)
    username: str | None = Field(
        description="The username used for basic authentication according to the OpenAPI 3.0 spec.", default=None)
    password: str | None = Field(
        description="The password used for basic authentication according to the OpenAPI 3.0 spec.", default=None)

    @field_validator('raw_key')
    @classmethod
    def validate_raw_key(cls, value: str) -> str:
        """
        Validate raw_key field for security requirements.

        Args:
            value: The raw API key value to validate

        Returns:
            str: The validated raw key

        Raises:
            APIKeyFieldError: If validation fails
        """
        if not value:
            raise APIKeyFieldError('value_missing', 'raw_key field value is required.')

        # Check for whitespace
        if len(value.strip()) != len(value):
            raise APIKeyFieldError('whitespace_found',
                                   'raw_key field value cannot have leading or trailing whitespace.')

        # Check for minimum length
        if len(value) < 8:
            raise APIKeyFieldError(
                'value_too_short',
                'raw_key field value must be at least 8 characters long for security. '
                'Got: {length} characters', {'length': len(value)})

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
