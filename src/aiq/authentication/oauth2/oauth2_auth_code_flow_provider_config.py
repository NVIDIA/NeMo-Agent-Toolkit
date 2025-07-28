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
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientIDFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientSecretFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigClientServerUrlFieldError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigScopeFieldError
from aiq.data_models.authentication import AuthProviderBaseConfig

logger = logging.getLogger(__name__)


class OAuth2AuthCodeFlowProviderConfig(AuthProviderBaseConfig, name="oauth2_auth_code_flow"):

    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    client_secret: str = Field(description="The secret associated with the client_id.")
    authorization_url: str = Field(description="The authorization URL for OAuth 2.0 authentication.")
    token_url: str = Field(description="The token URL for OAuth 2.0 authentication.")
    token_endpoint_auth_method: str | None = Field(description="The authentication method for the token endpoint.",
                                                   default=None)
    scopes: list[str] = Field(description="The scopes for OAuth 2.0 authentication.", default_factory=list)
    response_type: str | None = Field(default="code", description="The response type for OAuth 2.0 authentication.")
    access_type: str | None = Field(default="offline", description="The access type for OAuth 2.0 authentication.")
    use_pkce: bool = Field(default=False,
                           description="Whether to use PKCE (Proof Key for Code Exchange) in the OAuth 2.0 flow.")

    authorization_kwargs: dict[str, str] | None = Field(description=("Additional keyword arguments for the "
                                                                     "authorization request."),
                                                        default=None)

    # Configuration for the local server that handles the redirect
    client_url: str = Field(description="The base URL for the client application.", default="http://localhost:8000")
    run_local_redirect_server: bool = Field(default=False,
                                            description="Whether to run a local server to handle the redirect URI.")
    local_redirect_server_port: int = Field(default=8000,
                                            description="Port for the local redirect "
                                            "server to listen on.")
    redirect_path: str = Field(default="/auth/redirect",
                               description="Path for the local redirect server to handle the callback.")

    @property
    def redirect_uri(self) -> str:
        return f"{self.client_url}{self.redirect_path}"

    @field_validator('authorization_url', 'token_url')
    @classmethod
    def validate_authorization_url(cls, value: str, info: ValidationInfo) -> str:
        """
        Validate authorization_url and authorization_token_url field values.
        """
        if not value:
            raise AuthCodeGrantConfigAuthorizationUrlFieldError('value_missing',
                                                                '{field_name} is required',
                                                                {'field_name': info.field_name})

        # Check for valid scheme
        parsed = urlparse(value)

        # if parsed.scheme != 'https': # TODO EE: Breaks example, but needed for security testing.
        #     raise AuthCodeGrantConfigAuthorizationUrlFieldError(
        #         'https_required',
        #         '{field_name} must use HTTPS protocol for security. Got: {scheme}://', {
        #             'field_name': info.field_name, 'scheme': parsed.scheme
        #         })

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

    @field_validator('client_url')
    @classmethod
    def validate_client_server_url(cls, value: str) -> str:
        """
        Validate client_url field value.
        """
        if not value:
            raise AuthCodeGrantConfigClientServerUrlFieldError('value_missing', 'client_url field value is required.')

        # Check for valid scheme
        parsed = urlparse(value)
        if parsed.scheme not in ['http', 'https']:
            raise AuthCodeGrantConfigClientServerUrlFieldError(
                'invalid_scheme', f'client_url must use HTTP or HTTPS protocol. Got: {parsed.scheme}://')

        # Check for valid hostname
        if not parsed.netloc:
            raise AuthCodeGrantConfigClientServerUrlFieldError('invalid_url', 'client_url must have a valid hostname')

        if parsed.scheme == 'http' and not parsed.netloc.startswith(('localhost', '127.0.0.1')):
            logger.warning(
                'HTTP is not recommended for production environment. '
                'Use HTTPS instead for production environment. Value: %s://',
                parsed.scheme)

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
                'client_secret must be at least 16 characters long, ensuring a minimum of 128 bits of entropy.'
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

    @field_validator('scopes')
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

        # Check for duplicate scopes
        if len(value) != len(set(value)):
            duplicates = [scope for scope in set(value) if value.count(scope) > 1]
            raise AuthCodeGrantConfigScopeFieldError('duplicate_found',
                                                     'Duplicate scope found: {duplicates}', {'duplicates': duplicates})

        return value
