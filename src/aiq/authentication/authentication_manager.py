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
from aiq.data_models.authentication import OAuth2Config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiq.authentication.request_manager import RequestManager
    from aiq.authentication.response_manager import ResponseManager


class AuthenticationManager:  # TODO EE: Add Tests

    def __init__(self, request_manager: "RequestManager", response_manager: "ResponseManager") -> None:
        self._oauth2_authenticator: OAuth2Authenticator = OAuth2Authenticator(request_manager=request_manager,
                                                                              response_manager=response_manager)

    async def _validate_auth_provider_credentials(self, authentication_provider: str) -> bool:
        """
        Validate the authentication provider credentials.

        Args:
            authentication_provider (str): The name of the registered authentication provider.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            provider: AuthenticationProvider | None = _CredentialsManager()._get_authentication_provider(
                authentication_provider)

            if provider is None:
                raise ValueError(f"Authentication provider {authentication_provider} not found.")

            if not isinstance(provider, (OAuth2Config | APIKeyConfig)):
                raise TypeError(f"Authentication type for {authentication_provider} not supported. Supported types can "
                                f"be found in \"aiq.data_models.authentication\" ")

            if isinstance(provider, OAuth2Config):
                # Set the provider of interest.
                self._oauth2_authenticator.authentication_provider = provider

                is_validated: bool = await self._oauth2_authenticator._validate_credentials()

                # Reset the provider of interest.
                self._oauth2_authenticator.authentication_provider = None

                return is_validated

            if isinstance(provider, APIKeyConfig):
                pass  # TODO EE: Update API key authentication

        except (Exception, ValueError, TypeError) as e:
            logger.error("Failed to validate authentication provider credentials: %s Error: %s",
                         authentication_provider,
                         str(e),
                         exc_info=True)
            return False

        return False

    async def _set_auth_provider_credentials(self, authentication_provider: str) -> bool:
        """
        Gets the authentication provider credentials for the registered provider.

        Args:
            authentication_provider (str): The name of the registered authentication provider.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            provider: AuthenticationProvider | None = _CredentialsManager()._get_authentication_provider(
                authentication_provider)

            if provider is None:
                raise ValueError(f"Authentication provider {authentication_provider} not found.")

            if not isinstance(provider, (OAuth2Config | APIKeyConfig)):
                raise TypeError(f"Authentication type for {authentication_provider} not supported. Supported types can "
                                f"be found in \"aiq.data_models.authentication\" ")

            if isinstance(provider, OAuth2Config):
                # Set the provider of interest.
                self._oauth2_authenticator.authentication_provider = provider

                credentials_set: bool = await self._oauth2_authenticator._get_credentials()

                # Reset the provider of interest.
                self._oauth2_authenticator.authentication_provider = None

                return credentials_set

            if isinstance(provider, APIKeyConfig):
                pass  # TODO EE: Update API key authentication

        except (Exception, ValueError, TypeError) as e:
            logger.error("Failed to validate authentication provider credentials: %s Error: %s",
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

                if isinstance(auth_provider, OAuth2Config):
                    # TODO EE: Authentication schemes to add Basic, API key, JWT
                    auth_header: httpx.Headers = httpx.Headers(
                        {"Authorization": f"Bearer {auth_provider.access_token}"})

                    return auth_header

                if isinstance(auth_provider, APIKeyConfig):
                    pass

            else:
                raise ValidationError(f"Authentication provider: {authentication_provider} is not authenticated.")

        except (ValidationError) as e:
            logger.error("Authentication provider: %s is not authenticated. %s",
                         authentication_provider,
                         str(e),
                         exc_info=True)
            return None
