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
import re
import string

from pydantic import Field
from pydantic import field_validator

from nat.authentication.exceptions.api_key_exceptions import APIKeyFieldError
from nat.authentication.exceptions.api_key_exceptions import HeaderNameFieldError
from nat.authentication.exceptions.api_key_exceptions import HeaderPrefixFieldError
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import HeaderAuthScheme
from nat.data_models.common import SerializableSecretStr

logger = logging.getLogger(__name__)

# Strict RFC 7230 compliant header name regex
HEADER_NAME_REGEX = re.compile(r"^[!#$%&'*+\-.^_`|~0-9a-zA-Z]+$")


class APIKeyAuthProviderConfig(AuthProviderBaseConfig, name="api_key"):
    """
    API Key authentication configuration model.
    """

    raw_key: SerializableSecretStr = Field(
        description=("Raw API token or credential to be injected into the request parameter. "
                     "Used for 'bearer','x-api-key','custom', and other schemes. "))

    auth_scheme: HeaderAuthScheme = Field(default=HeaderAuthScheme.BEARER,
                                          description=("The HTTP authentication scheme to use. "
                                                       "Supported schemes: BEARER, X_API_KEY, BASIC, CUSTOM."))

    custom_header_name: str | None = Field(description="The HTTP header name that MUST be used in conjunction "
                                           "with the custom_header_prefix when HeaderAuthScheme is CUSTOM.",
                                           default=None)
    custom_header_prefix: str | None = Field(description="The HTTP header prefix that MUST be used in conjunction "
                                             "with the custom_header_name when HeaderAuthScheme is CUSTOM.",
                                             default=None)

    @field_validator('raw_key')
    @classmethod
    def validate_raw_key(cls, value: SerializableSecretStr) -> SerializableSecretStr:
        if not value:
            raise APIKeyFieldError('value_missing', 'raw_key field value is required.')

        if len(value) < 8:
            raise APIKeyFieldError(
                'value_too_short',
                'raw_key field value must be at least 8 characters long for security. '
                f'Got: {len(value)} characters.')

        str_value = value.get_secret_value()
        if len(str_value.strip()) != len(value):
            raise APIKeyFieldError('whitespace_found',
                                   'raw_key field value cannot have leading or trailing whitespace.')

        if any(c in string.whitespace for c in str_value):
            raise APIKeyFieldError('contains_whitespace', 'raw_key must not contain any '
                                   'whitespace characters.')

        return value

    @field_validator('custom_header_name')
    @classmethod
    def validate_custom_header_name(cls, value: str) -> str:
        if not value:
            raise HeaderNameFieldError('value_missing', 'custom_header_name is required.')

        if value != value.strip():
            raise HeaderNameFieldError('whitespace_found',
                                       'custom_header_name field value cannot have leading or trailing whitespace.')

        if any(c in string.whitespace for c in value):
            raise HeaderNameFieldError('contains_whitespace',
                                       'custom_header_name must not contain any whitespace characters.')

        if not HEADER_NAME_REGEX.fullmatch(value):
            raise HeaderNameFieldError(
                'invalid_format',
                'custom_header_name must match the HTTP token syntax: ASCII letters, digits, or allowed symbols.')

        return value

    @field_validator('custom_header_prefix')
    @classmethod
    def validate_custom_header_prefix(cls, value: str) -> str:
        if not value:
            raise HeaderPrefixFieldError('value_missing', 'custom_header_prefix is required.')

        if value != value.strip():
            raise HeaderPrefixFieldError(
                'whitespace_found', 'custom_header_prefix field value cannot have '
                'leading or trailing whitespace.')

        if any(c in string.whitespace for c in value):
            raise HeaderPrefixFieldError('contains_whitespace',
                                         'custom_header_prefix must not contain any whitespace characters.')

        if not value.isascii():
            raise HeaderPrefixFieldError('invalid_format', 'custom_header_prefix must be ASCII.')

        return value

    @field_validator('raw_key', mode='after')
    @classmethod
    def validate_raw_key_after(cls, value: str) -> str:
        if not value:
            raise APIKeyFieldError('value_missing', 'raw_key field value is '
                                   'required after construction.')

        return value
