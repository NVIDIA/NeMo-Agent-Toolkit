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
from urllib.parse import urlparse

from pydantic import Field
from pydantic import ValidationInfo
from pydantic import field_validator

from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigAuthorizationUrlFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientServerUrlFieldError
from aiq.authentication.oauth2.oauth_user_consent_base_config import OAuthUserConsentConfigBase
from aiq.builder.authentication import AuthenticationProviderInfo
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_authentication_provider

logger = logging.getLogger(__name__)


class AuthCodeGrantConfig(OAuthUserConsentConfigBase, name="oauth2_authorization_code_grant"):
    """
    OAuth 2.0 authorization code grant authentication configuration model.
    Implements RFC 6819 security validation requirements.
    """
    client_server_url: str = Field(description="The base url of the API server instance. "
                                   "This is needed to properly construct the redirect uri i.e: http://localhost:8000")
    authorization_url: str = Field(description="The base url to the authorization server in which authorization "
                                   "request are made to receive access codes..")
    authorization_token_url: str = Field(  # TODO EE: Use pydantic secret string type after all testing is complete.
        description="The base url to the authorization token server in which access codes "
        "are exchanged for access tokens.")

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


@register_authentication_provider(config_type=AuthCodeGrantConfig)
async def oauth2_authorization_code_grant(authentication_provider: AuthCodeGrantConfig, builder: Builder):

    yield AuthenticationProviderInfo(config=authentication_provider,
                                     description="OAuth 2.0 Authorization Code Grant authentication provider.")
