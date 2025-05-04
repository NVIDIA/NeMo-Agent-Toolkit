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

from aiq.builder.context import Singleton
from aiq.data_models.authentication import AuthenticationProvider

if (typing.TYPE_CHECKING):
    from aiq.data_models.config import AIQConfig

logger = logging.getLogger(__name__)


class _CredentialsManager(metaclass=Singleton):

    def __init__(self):
        """
        Initializes the Credentials Manager object with the specified AIQ Authorization configuration.
        """
        super().__init__()
        self.__authentication_providers: dict[str, AuthenticationProvider] = {}
        self.__swap_flag: bool = True
        self.__full_config: "AIQConfig" = None
        self.__command_name: str = None  # TODO EE: Need to get a list of command names.
        self.__credentials_flag: asyncio.Event = asyncio.Event()

    def _swap_authorization_providers(self, authentication_providers: dict[str, AuthenticationProvider]) -> None:
        """Atomically transfer ownership of the sensitive AIQ Authorization configuration attributes to the
        CredentialsManager using the copy and swap idiom."""

        if self.__swap_flag:
            self.__authentication_providers = authentication_providers.copy()
            authentication_providers.clear()
            self.__swap_flag = False

    def _get_authentication_provider(self, authentication_provider: str) -> AuthenticationProvider | None:
        """Retrieve the stored authentication providers."""
        if authentication_provider not in self.__authentication_providers:
            logger.warning("Authorization provider not found: %s", authentication_provider, exc_info=True)
            return None

        return self.__authentication_providers.get(authentication_provider)

    async def _get_credentials(self):
        await self.__credentials_flag.wait()  # TODO EE: Maybe add a future here

    async def _set_credentials(self):
        self.__credentials_flag.set()

    @property
    def full_config(self) -> "AIQConfig":
        """Get the full configuration."""
        return self.__full_config

    @full_config.setter
    def full_config(self, full_config: "AIQConfig") -> None:
        """Set the full configuration."""
        self.__full_config = full_config

    @property
    def command_name(self) -> str:
        """Get the front end command name."""
        return self.__command_name

    @command_name.setter
    def command_name(self, command_name: str) -> None:
        """Set the front end command name."""
        self.__command_name = command_name
