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
from typing import TYPE_CHECKING

import httpx
from pydantic import ValidationError

from aiq.authentication.oauth2_authenticator import OAuth2Authenticator
from aiq.data_models.authentication import APIKeyConfig
from aiq.data_models.authentication import AuthenticationProvider
from aiq.data_models.authentication import ExecutionMode
from aiq.data_models.authentication import OAuth2Config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiq.authentication.request_manager import RequestManager
    from aiq.authentication.response_manager import ResponseManager


class AuthenticationManager:

    def __init__(self, request_manager: "RequestManager", response_manager: "ResponseManager") -> None:
        self._oauth2_authenticator: OAuth2Authenticator = OAuth2Authenticator(request_manager=request_manager,
                                                                              response_manager=response_manager)

    def _set_execution_mode(self, execution_mode: ExecutionMode) -> None:
        self._oauth2_authenticator.execution_mode = execution_mode

    async def _validate_auth_provider_credentials(self, authentication_provider_name: str) -> bool:
        """
        Validate the authentication provider credentials.

        Args:
            authentication_provider (str): The name of the registered authentication provider.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            provider: AuthenticationProvider | None = _CredentialsManager()._get_authentication_provider(
                authentication_provider_name)

            if provider is None:
                raise ValueError(f"Authentication provider {authentication_provider_name} not found.")

            if isinstance(provider, OAuth2Config):
                # Set the provider of interest.
                self._oauth2_authenticator.authentication_provider = provider

                is_validated: bool = await self._oauth2_authenticator._validate_credentials()

                # Reset the provider of interest.
                self._oauth2_authenticator.authentication_provider = None

                return is_validated

            if isinstance(provider, APIKeyConfig):
                return await self._validate_api_key_credentials(authentication_provider_name, provider)

        except (Exception, ValueError, TypeError) as e:
            logger.error("Failed to validate authentication provider credentials: %s Error: %s",
                         authentication_provider_name,
                         str(e),
                         exc_info=True)
            return False

    async def _validate_api_key_credentials(self, api_key_name: str, api_key_provider: APIKeyConfig) -> bool:
        """
        Ensure that the API key credentials are valid for the given API key configuration.

        Args:
            api_key_config (APIKeyConfig): The API key configuration instance to validate.

        Returns:
            bool: True if the API key credentials are valid, False otherwise.
        """
        # Ensure there is an API key set.
        if api_key_provider.api_key is None or api_key_provider.api_key == "":
            logger.error("API key is not set or is empty for provider: %s", api_key_name)
            return False

        # Ensure the header name is set.
        if api_key_provider.header_name is None or api_key_provider.header_name == "":
            logger.error("API key config header name is not set or is empty for provider: %s", api_key_name)
            return False

        return True

    async def _set_auth_provider_credentials(self, authentication_provider: str) -> bool:
        """
        Gets and persists the authentication provider credentials for the registered provider.

        Args:
            authentication_provider (str): The name of the registered authentication provider.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            provider: AuthenticationProvider | None = _CredentialsManager()._get_authentication_provider(
                authentication_provider)

            if provider is None:
                raise ValueError(f"Authentication provider {authentication_provider} not found.")

            if isinstance(provider, OAuth2Config):
                # Set the provider of interest.
                self._oauth2_authenticator.authentication_provider = provider

                credentials_set: bool = await self._oauth2_authenticator._get_credentials()

                # Reset the provider of interest.
                self._oauth2_authenticator.authentication_provider = None

                return credentials_set

        except (Exception, ValueError, TypeError) as e:
            logger.error("Failed get and persist authentication provider credentials: %s Error: %s",
                         authentication_provider,
                         str(e),
                         exc_info=True)
            return False

        return False

    async def _construct_authentication_header(self, authentication_provider: str | None) -> httpx.Headers | None:
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            if authentication_provider is None:
                return None

            is_validated: bool = await self._validate_auth_provider_credentials(authentication_provider)

            if (is_validated):

                auth_provider: AuthenticationProvider | None = _CredentialsManager()._get_authentication_provider(
                    authentication_provider)

                # Construct Oauth2.0 authentication header.
                if isinstance(auth_provider, OAuth2Config):
                    auth_header: httpx.Headers = httpx.Headers(
                        {"Authorization": f"Bearer {auth_provider.access_token}"})
                    return auth_header

                # Construct API key authentication header.
                if isinstance(auth_provider, APIKeyConfig):
                    auth_header: httpx.Headers = httpx.Headers(
                        {f"{auth_provider.header_name}": f"{auth_provider.header_prefix} {auth_provider.api_key}"})
                    return auth_header

            else:
                raise ValidationError(
                    f"Authentication provider: {authentication_provider} credentials can not be validated.")

        except (ValidationError) as e:
            logger.error("Authentication provider: %s credentials can not be validated: %s",
                         authentication_provider,
                         str(e),
                         exc_info=True)
            return None
