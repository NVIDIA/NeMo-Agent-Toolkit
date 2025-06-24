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

from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig
from aiq.builder.context import Singleton
from aiq.data_models.authentication import AuthenticationBaseConfig
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
        self._authentication_configs: dict[str, AuthenticationBaseConfig] = {}
        self._swap_flag: bool = True
        self._full_config: "AIQConfig" = None
        self._oauth_credentials_flag: asyncio.Event = asyncio.Event()
        self._consent_prompt_flag: asyncio.Event = asyncio.Event()

    def _swap_authentication_configs(self, authentication_configs: dict[str, AuthenticationBaseConfig]) -> None:
        """
        Transfer ownership of the sensitive AIQ Authorization configuration attributes to the
        CredentialsManager.

        Args:
            http_method (str): The HTTP method to validate (e.g., 'GET', 'POST').
            authentication_configs (dict[str, AuthenticationBaseConfig]): Dictionary of registered authentication
            configs.
        """
        if self._swap_flag:
            self._authentication_configs = deepcopy(authentication_configs)
            authentication_configs.clear()
            self._swap_flag = False

    def _get_authentication_config(self, authentication_config_name: str | None) -> AuthenticationBaseConfig | None:
        """Retrieve the stored authentication config by registered name."""

        if authentication_config_name not in self._authentication_configs:
            logger.error("Authentication config not found: %s", authentication_config_name)
            return None

        return self._authentication_configs.get(authentication_config_name)

    def _get_authentication_config_by_state(self, state: str) -> AuthCodeGrantConfig | None:
        """Retrieve the stored authentication config by state."""

        for _, authentication_config in self._authentication_configs.items():
            if isinstance(authentication_config, AuthCodeGrantConfig):
                if authentication_config.state == state:
                    return authentication_config

        logger.error("Authentication config not found by the provided state.")
        return None

    def _get_registered_authentication_config_name(self, authentication_config: AuthenticationBaseConfig) -> str | None:
        """Retrieve the stored authentication config name."""

        for registered_config_name, registered_config in self._authentication_configs.items():
            if (authentication_config == registered_config):
                return registered_config_name

        logger.error("Authentication config name not found by the provided authentication config model.")
        return None

    def _get_authentication_config_by_consent_prompt_key(self, consent_prompt_key: str) -> AuthCodeGrantConfig | None:
        """Retrieve the stored authentication config by consent prompt key."""
        for _, authentication_config in self._authentication_configs.items():
            if isinstance(authentication_config, AuthCodeGrantConfig):
                if authentication_config.consent_prompt_key == consent_prompt_key:
                    return authentication_config
        return None

    def _validate_and_set_cors_config(self, front_end_config: FrontEndBaseConfig) -> None:
        """
        Validate and set the CORS authentication configuration for the frontend.
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
