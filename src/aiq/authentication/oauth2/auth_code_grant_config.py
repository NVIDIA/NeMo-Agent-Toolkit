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
import secrets
from datetime import datetime
from enum import Enum
from urllib.parse import urlparse

from pydantic import Field
from pydantic import ValidationInfo
from pydantic import field_validator

from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigAudienceFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigAuthorizationUrlFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientIDFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientSecretFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientServerUrlFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigConsentPromptKeyFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigScopeFieldError
from aiq.builder.authentication import AuthenticationProviderInfo
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_authentication_provider
from aiq.data_models.authentication import AuthenticationBaseConfig

logger = logging.getLogger(__name__)


class ConsentPromptMode(str, Enum):
    BROWSER = "browser"
    FRONTEND = "frontend"


class AuthCodeGrantConfig(AuthenticationBaseConfig, name="oauth2_authorization_code_grant"):
    """
    OAuth 2.0 authorization code grant authentication configuration model.
    Implements RFC 6819 security validation requirements.
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
                                    " triggering the browser to complete the OAuth process from the front end.")
    client_secret: str = Field(description="The client secret for OAuth 2.0 authentication.")
    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    audience: str = Field(description="The audience for OAuth 2.0 authentication.")
    scope: list[str] = Field(description="The scope for OAuth 2.0 authentication.")
    state: str = Field(default=secrets.token_urlsafe(nbytes=16),
                       description="A URL-safe base64 format 16 byte random string")
    access_token: str | None = Field(default=None, description="The access token for OAuth 2.0 authentication.")
    access_token_expires_in: datetime | None = Field(default=None,
                                                     description="Expiry time of the access token in seconds.")
    refresh_token: str | None = Field(
        default=None, description="The refresh token for OAuth 2.0 authentication used to obtain a new access token.")
    consent_prompt_location_url: str | None = Field(
        default=None,
        description="302 redirect Location header to which the client will be redirected to the consent prompt.")

    @field_validator('client_server_url')
    @classmethod
    def validate_client_server_url(cls, value: str) -> str:
        """
        Validate client_server_url field value.
        """
        if not value:
            raise AuthCodeGrantConfigClientServerUrlFieldError('value_missing',
                                                               'client_server_url field value is required.')

        # Check for valid scheme
        parsed = urlparse(value)
        if parsed.scheme not in ['http', 'https']:
            raise AuthCodeGrantConfigClientServerUrlFieldError(
                'invalid_scheme',
                'client_server_url must use HTTP or HTTPS protocol. Got: {scheme}://', {'scheme': parsed.scheme})

        # Check for valid hostname
        if not parsed.netloc:
            raise AuthCodeGrantConfigClientServerUrlFieldError('invalid_url',
                                                               'client_server_url must have a valid hostname')

        if parsed.scheme == 'http' and not parsed.netloc.startswith(('localhost', '127.0.0.1')):
            logger.warning(
                'HTTP is not recommended for production environment. '
                'Use HTTPS instead for production environment. Value: %s://',
                parsed.scheme)

        return value

    @field_validator('authorization_url', 'authorization_token_url')
    @classmethod
    def validate_authorization_url(cls, value: str, info: ValidationInfo) -> str:
        """
        Validate authorization_url and authorization_token_url field values.
        RFC 6819 Section 4.2.1 - OAuth endpoints MUST use HTTPS to prevent token interception.
        """
        if not value:
            raise AuthCodeGrantConfigAuthorizationUrlFieldError('value_missing',
                                                                '{field_name} is required',
                                                                {'field_name': info.field_name})

        # Check for valid scheme
        parsed = urlparse(value)

        if parsed.scheme != 'https':
            raise AuthCodeGrantConfigAuthorizationUrlFieldError(
                'https_required',
                '{field_name} must use HTTPS protocol for security (RFC 6819 Section 4.2.1). Got: {scheme}://', {
                    'field_name': info.field_name, 'scheme': parsed.scheme
                })

        # Check for valid hostname
        if not parsed.netloc:
            raise AuthCodeGrantConfigAuthorizationUrlFieldError('invalid_url',
                                                                '{field_name} must have a valid hostname',
                                                                {'field_name': info.field_name})

        # Ensure the URL includes a specific path and is not just the domain root
        if not parsed.path or parsed.path == '/':
            raise AuthCodeGrantConfigAuthorizationUrlFieldError(
                'path_missing',
                '{field_name} should include a valid endpoint path (e.g., /oauth/authorize)',
                {'field_name': info.field_name})

        return value

    @field_validator('consent_prompt_key')
    @classmethod
    def validate_consent_prompt_key(cls, value: str) -> str:
        """
        Validate consent prompt key for security.
        """
        if not value:
            raise AuthCodeGrantConfigConsentPromptKeyFieldError('value_missing',
                                                                'consent_prompt_key field value is required.')

        # Check for whitespace
        if len(value.strip()) != len(value):
            raise AuthCodeGrantConfigConsentPromptKeyFieldError(
                'whitespace_found', 'consent_prompt_key field value cannot have leading or trailing whitespace.')

        # Check for minimum length
        if len(value) < 8:
            raise AuthCodeGrantConfigConsentPromptKeyFieldError(
                'value_too_short',
                'consent_prompt_key field value must be at least 8 characters long for security. '
                'Got: {length} characters', {'length': len(value)})

        return value

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


@register_authentication_provider(config_type=AuthCodeGrantConfig)
async def oauth2_authorization_code_grant(authentication_provider: AuthCodeGrantConfig, builder: Builder):

    yield AuthenticationProviderInfo(config=authentication_provider,
                                     description="OAuth 2.0 authorization code grant authentication provider.")
