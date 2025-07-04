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

from aiq.authentication.interfaces import AuthenticationManagerBase
from aiq.authentication.interfaces import OAuthClientManagerBase
from aiq.authentication.oauth2.oauth_user_consent_base_config import OAuthUserConsentConfigBase
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
        self._authentication_managers: dict[str, AuthenticationManagerBase] = {}
        self._full_config: "AIQConfig | None" = None
        self._oauth_credentials_flag: asyncio.Event = asyncio.Event()
        self._consent_prompt_flag: asyncio.Event = asyncio.Event()

    def validate_unique_consent_prompt_keys(self, authentication_configs: dict[str, AuthenticationBaseConfig]) -> None:
        """
        Validate that all OAuthUserConsentConfigBase instances have unique consent_prompt_key values.

        Args:
            authentication_configs: Authentication configuration objects from config file.

        Raises:
            RuntimeError: If duplicate consent prompt keys are found.
        """
        consent_prompt_keys: list[str] = []

        # Collect all consent prompt keys and their associated config names
        for _, auth_config in authentication_configs.items():
            if isinstance(auth_config, OAuthUserConsentConfigBase):

                if auth_config.consent_prompt_key in consent_prompt_keys:
                    error_message = (f"Duplicate consent_prompt_key found: {auth_config.consent_prompt_key}. "
                                     "Please ensure consent_prompt_key is unique across Authentication configs.")
                    logger.critical(error_message)
                    raise RuntimeError('duplicate_consent_prompt_key', error_message)
                else:
                    consent_prompt_keys.append(auth_config.consent_prompt_key)

    def store_authentication_manager(self, name: str, manager: AuthenticationManagerBase) -> None:
        """
        Store or update an authentication manager instance.

        Args:
            name: The name/key for the authentication manager
            manager: The authentication manager instance to store
        """
        self._authentication_managers[name] = manager

    def get_authentication_manager_by_state(self, state: str) -> OAuthClientManagerBase | None:
        """
        Get authentication manager by the state value.

        Args:
            state: The state value to match.

        Returns:
            The OAuth authentication manager if found, None otherwise.
        """
        for auth_manager in self._authentication_managers.values():
            if (isinstance(auth_manager, OAuthClientManagerBase) and auth_manager.config is not None
                    and hasattr(auth_manager.config, 'state')):
                if auth_manager.config.state == state:
                    return auth_manager
        return None

    def get_authentication_manager_by_consent_prompt_key(self,
                                                         consent_prompt_key: str) -> OAuthClientManagerBase | None:
        """
        Get authentication manager by the consent_prompt_key value.

        Args:
            consent_prompt_key: The consent prompt key to match.

        Returns:
            The OAuth authentication manager if found, None otherwise.
        """
        for auth_manager in self._authentication_managers.values():
            if (isinstance(auth_manager, OAuthClientManagerBase) and auth_manager.config is not None
                    and hasattr(auth_manager.config, 'consent_prompt_key')):
                if auth_manager.config.consent_prompt_key == consent_prompt_key:
                    return auth_manager
        return None

    def validate_and_set_cors_config(self, front_end_config: FrontEndBaseConfig) -> None:
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

            # Check if full_config and general are not None before accessing
            credentials_manager = _CredentialsManager()
            if credentials_manager.full_config is not None and hasattr(credentials_manager.full_config, 'general'):
                credentials_manager.full_config.general.front_end = front_end_config
            else:
                # Handle case where full_config or general is None
                logger.warning("full_config or general is None, cannot set front_end configuration")

        except ValueError:
            credentials_manager = _CredentialsManager()
            if credentials_manager.full_config is not None and hasattr(credentials_manager.full_config, 'general'):
                credentials_manager.full_config.general.front_end = FastApiFrontEndConfig(
                    cors=FastApiFrontEndConfig.CrossOriginResourceSharing(
                        allow_origins=default_allow_origins,
                        allow_headers=default_allow_headers,
                        allow_methods=default_allow_methods,
                    ))

    async def wait_for_oauth_credentials(self) -> None:
        """
        Block until the oauth credentials are set in the redirect uri.
        """
        await self._oauth_credentials_flag.wait()

    async def set_oauth_credentials(self) -> None:
        """
        Unblock when the oauth credentials are set in the redirect uri.
        """
        self._oauth_credentials_flag.set()

    async def wait_for_consent_prompt_url(self) -> None:
        """
        Block until the consent prompt location header has been retrieved.
        """
        await self._consent_prompt_flag.wait()

    async def set_consent_prompt_url(self) -> None:
        """
        Unblock when the consent prompt location header has been retrieved.
        """
        self._consent_prompt_flag.set()

    @property
    def full_config(self) -> "AIQConfig | None":
        """Get the loaded AIQConfig."""
        return self._full_config

    @full_config.setter
    def full_config(self, full_config: "AIQConfig") -> None:
        """Set the loaded AIQConfig."""
        self._full_config = full_config
