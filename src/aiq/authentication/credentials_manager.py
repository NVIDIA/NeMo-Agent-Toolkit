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

import asyncio
import logging
import typing
from copy import deepcopy

from aiq.builder.context import Singleton
from aiq.data_models.authentication import AuthenticationProvider
from aiq.data_models.authentication import OAuth2Config
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from aiq.front_ends.fastapi.fastapi_front_end_config import FrontEndBaseConfig

if (typing.TYPE_CHECKING):
    from aiq.data_models.config import AIQConfig

logger = logging.getLogger(__name__)


class _CredentialsManager(metaclass=Singleton):

    def __init__(self):
        """
        Credentials Manager to store AIQ Authorization configurations.
        """
        super().__init__()
        self._authentication_providers: dict[str, AuthenticationProvider] = {}
        self._swap_flag: bool = True
        self._full_config: "AIQConfig" = None
        self._oauth_credentials_flag: asyncio.Event = asyncio.Event()
        self._consent_prompt_flag: asyncio.Event = asyncio.Event()

    def _swap_authorization_providers(self, authentication_providers: dict[str, AuthenticationProvider]) -> None:
        """
        Transfer ownership of the sensitive AIQ Authorization configuration attributes to the
        CredentialsManager.

        Args:
            http_method (str): The HTTP method to validate (e.g., 'GET', 'POST').
            authentication_providers (dict[str, AuthenticationProvider]): Dictionary of registered authentication
            providers.
        """
        if self._swap_flag:
            self._authentication_providers = deepcopy(authentication_providers)
            authentication_providers.clear()
            self._swap_flag = False

    def _get_authentication_provider(self, authentication_provider: str) -> AuthenticationProvider | None:
        """Retrieve the stored authentication provider by registered name."""

        if authentication_provider not in self._authentication_providers:
            logger.error("Authentication provider not found: %s", authentication_provider)
            return None

        return self._authentication_providers.get(authentication_provider)

    def _get_authentication_provider_by_state(self, state: str) -> OAuth2Config | None:
        """Retrieve the stored authentication provider by state."""

        for _, authentication_provider in self._authentication_providers.items():
            if isinstance(authentication_provider, OAuth2Config):
                if authentication_provider.state == state:
                    return authentication_provider

        logger.error("Authentication provider not found by the provided state.")
        return None

    def _get_registered_authentication_provider_name(self,
                                                     authentication_provider: AuthenticationProvider) -> str | None:
        """Retrieve the stored authentication provider name."""

        for registered_provider_name, registered_provider in self._authentication_providers.items():
            if (authentication_provider == registered_provider):
                return registered_provider_name

        logger.error("Authentication provider name not found by the provided authentication provider.")
        return None

    def _get_authentication_provider_by_consent_prompt_key(self, consent_prompt_key: str) -> OAuth2Config | None:
        """Retrieve the stored authentication provider by consent prompt key."""
        for _, authentication_provider in self._authentication_providers.items():
            if isinstance(authentication_provider, OAuth2Config):
                if authentication_provider.consent_prompt_key == consent_prompt_key:
                    return authentication_provider
        return None

    def _validate_and_set_cors_config(self, front_end_config: FrontEndBaseConfig) -> None:
        """
        Validate and set the CORS configuration for the authentication providers.
        """

        default_allow_origins: list[str] = ["http://localhost:3000"]
        default_allow_headers: list[str] = ["Content-Type", "Authorization"]
        default_allow_methods: list[str] = ["POST", "OPTIONS"]

        try:
            if not isinstance(front_end_config, FastApiFrontEndConfig):
                raise ValueError("Configuration is not of type FastApiFrontEndConfig.")

            # Allow the AIQ frontend browser to access the OAuth server in headless execution modes.
            if front_end_config.cors.allow_origins is None:
                front_end_config.cors.allow_origins = default_allow_origins
            else:
                for item in default_allow_origins:
                    if item not in front_end_config.cors.allow_origins:
                        front_end_config.cors.allow_origins.append(item)

            # Allow minimum headers to access the OAuth server in headless execution modes.
            if front_end_config.cors.allow_headers is None:
                front_end_config.cors.allow_headers = default_allow_headers
            else:
                for item in default_allow_headers:
                    if item not in front_end_config.cors.allow_headers:
                        front_end_config.cors.allow_headers.append(item)

            # Allow minimum methods to access the OAuth server in headless execution modes.
            if front_end_config.cors.allow_methods is None:
                front_end_config.cors.allow_methods = default_allow_methods
            else:
                for item in default_allow_methods:
                    if item not in front_end_config.cors.allow_methods:
                        front_end_config.cors.allow_methods.append(item)

            _CredentialsManager().full_config.general.front_end = front_end_config

        except ValueError:
            _CredentialsManager().full_config.general.front_end = FastApiFrontEndConfig(
                cors=FastApiFrontEndConfig.CrossOriginResourceSharing(
                    allow_origins=default_allow_origins,
                    allow_headers=default_allow_headers,
                    allow_methods=default_allow_methods,
                ))

    async def _wait_for_oauth_credentials(self) -> None:
        """
        Block until the oauth credentials are set in the redirect uri.
        """
        await self._oauth_credentials_flag.wait()

    async def _set_oauth_credentials(self):
        """
        Unblock until the oauth credentials are set in the redirect uri.
        """
        self._oauth_credentials_flag.set()

    async def _wait_for_consent_prompt_url(self):
        """
        Block until the consent prompt location header has been retrieved.
        """
        await self._consent_prompt_flag.wait()

    async def _set_consent_prompt_url(self):
        """
        Unblock until the consent prompt location header has been retrieved.
        """
        self._consent_prompt_flag.set()

    @property
    def full_config(self) -> "AIQConfig":
        """Get the loaded AIQConfig."""
        return self._full_config

    @full_config.setter
    def full_config(self, full_config: "AIQConfig") -> None:
        """Set the loaded AIQConfig."""
        self._full_config = full_config
