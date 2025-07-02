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
from datetime import datetime

from pydantic import Field
from pydantic import field_validator

from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigAudienceFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientIDFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientSecretFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigScopeFieldError
from aiq.data_models.authentication import AuthenticationBaseConfig

logger = logging.getLogger(__name__)


class OAuthBaseConfig(AuthenticationBaseConfig):
    """
    Base OAuth 2.0 authentication configuration model.
    Contains common OAuth 2.0 fields used across different OAuth flows.
    """
    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    client_secret: str = Field(description="The secret associated with the client_id.")
    audience: str = Field(description="The resource server the token is intended for.")
    scope: list[str] = Field(description="List of scopes for OAuth 2.0 authentication.")
    authorization_url: str = Field(description="The authorization URL for OAuth 2.0 authentication.")
    authorization_token_url: str = Field(description="The token URL for OAuth 2.0 authentication.")

    access_token: str | None = Field(default=None, description="The access token for OAuth 2.0 authentication.")
    access_token_expires_in: datetime | None = Field(default=None,
                                                     description="The expiration time of the access token.")
    refresh_token: str | None = Field(default=None, description="The refresh token for OAuth 2.0 authentication.")

    @field_validator('client_secret')
    @classmethod
    def validate_client_secret(cls, value: str) -> str:
        """
        Validate client_secret field value.
        """
        if not value:
            raise AuthCodeGrantConfigClientSecretFieldError('value_missing',
                                                            'client_secret is required for OAuth 2.0 authentication.')

        # Check for minimum length
        if len(value) < 16:
            raise AuthCodeGrantConfigClientSecretFieldError(
                'value_too_short',
                'client_secret must be at least 16 characters long for security (RFC 6819 Section 4.1.3). '
                'Got: {length} characters', {
                    'length': len(value), 'minimum_length': 16
                })

        return value

    @field_validator('client_id')
    @classmethod
    def validate_client_id(cls, value: str) -> str:
        """
        Validate client_id field value.
        """
        if not value:
            raise AuthCodeGrantConfigClientIDFieldError('value_missing',
                                                        'client_id is required for OAuth 2.0 authentication')

        # Check for whitespace
        if len(value.strip()) != len(value):
            raise AuthCodeGrantConfigClientIDFieldError('whitespace_found',
                                                        'client_id cannot have leading or trailing whitespace')

        return value

    @field_validator('scope')
    @classmethod
    def validate_scope(cls, value: list[str]) -> list[str]:
        """
        Validate scope field value.
        """
        if not value:
            raise AuthCodeGrantConfigScopeFieldError('value_missing', 'At least one scope is required')

        # Check for empty scope
        empty_scopes = [scope for scope in value if not scope]
        if empty_scopes:
            raise AuthCodeGrantConfigScopeFieldError('value_empty',
                                                     'Scopes cannot be empty', {'empty_scopes': empty_scopes})

        # Check for whitespace
        whitespace_scopes = [scope for scope in value if scope and not scope.strip()]
        if whitespace_scopes:
            raise AuthCodeGrantConfigScopeFieldError('whitespace_found',
                                                     'Scopes cannot contain only whitespace',
                                                     {'whitespace_scopes': whitespace_scopes})

        # Check for overly broad or dangerous scopes
        dangerous_scopes = {'*', 'all', 'root', 'admin', 'superuser'}
        found_dangerous = set(value) & dangerous_scopes
        if found_dangerous:
            raise AuthCodeGrantConfigScopeFieldError(
                'value_too_broad',
                'Overly broad scopes detected. Follow principle of least privilege (RFC 6819 Section 4.3.1). '
                'Dangerous scopes: {dangerous_scopes}', {'dangerous_scopes': list(found_dangerous)})

        # Check for duplicate scopes
        if len(value) != len(set(value)):
            duplicates = [scope for scope in set(value) if value.count(scope) > 1]
            raise AuthCodeGrantConfigScopeFieldError('duplicate_found',
                                                     'Duplicate scopes found: {duplicates}', {'duplicates': duplicates})

        return value

    @field_validator('audience')
    @classmethod
    def validate_audience(cls, value: str) -> str:
        """
        Validate audience field value.
        """
        if not value:
            raise AuthCodeGrantConfigAudienceFieldError('value_missing', 'audience is required.')

        if len(value.strip()) != len(value):
            raise AuthCodeGrantConfigAudienceFieldError('whitespace_found', 'audience has leading/trailing whitespace.')

        if "*" in value:
            raise AuthCodeGrantConfigAudienceFieldError('wildcard_not_allowed', 'wildcards are not permitted.')

        return value
