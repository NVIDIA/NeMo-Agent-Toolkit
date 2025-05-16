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

    def _get_authentication_provider_by_consent_prompt_key(self, consent_prompt_key: str) -> OAuth2Config | None:
        """Retrieve the stored authentication provider by consent prompt key."""
        for _, authentication_provider in self._authentication_providers.items():
            if isinstance(authentication_provider, OAuth2Config):
                if authentication_provider.consent_prompt_key == consent_prompt_key:
                    return authentication_provider
        return None

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
