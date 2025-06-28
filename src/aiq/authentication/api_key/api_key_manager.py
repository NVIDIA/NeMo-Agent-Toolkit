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

import httpx

from aiq.authentication.api_key.api_key_config import APIKeyConfig
from aiq.authentication.interfaces import AuthenticationManagerBase

logger = logging.getLogger(__name__)


class APIKeyManager(AuthenticationManagerBase):

    def __init__(self, config_name: str, encrypted_config: APIKeyConfig) -> None:
        self._config_name: str = config_name
        self._encrypted_config: APIKeyConfig = encrypted_config
        super().__init__()

    async def validate_authentication_credentials(self) -> bool:
        """
        Ensure that the API key credentials are valid for the given API key configuration.

        Returns:
            bool: True if the API key credentials are valid, False otherwise.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        # Ensure there is an API key set.
        if _CredentialsManager().decrypt_value(
                self._encrypted_config.api_key) is None or _CredentialsManager().decrypt_value(
                    self._encrypted_config.api_key) == "":
            logger.error("API key is not set or is empty for config: %s", self._config_name)
            return False

        # Ensure the header name is set.
        if _CredentialsManager().decrypt_value(
                self._encrypted_config.header_name) is None or _CredentialsManager().decrypt_value(
                    self._encrypted_config.header_name) == "":
            logger.error("API key config header name is not set or is empty for config: %s", self._config_name)
            return False

        return True

    async def get_authentication_header(self) -> httpx.Headers | None:
        """
        Gets the authenticated header for the registered authentication config.

        Returns:
            httpx.Headers | None: Returns the authentication header if the config is valid and credentials are
            functional, otherwise returns None.
        """
        credentials_validated: bool = await self.validate_authentication_credentials()

        if credentials_validated:
            return await self.construct_authentication_header()
        else:
            logger.error("API key credentials are not valid for config: %s authentication header can not be retreived.",
                         self._config_name)
            return None

    async def construct_authentication_header(self) -> httpx.Headers | None:
        """
        Constructs the authenticated API key HTTP header.

        Returns:
            httpx.Headers | None: Returns the constructed HTTP header if the API key is valid, otherwise returns None.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        return httpx.Headers({
            f"{_CredentialsManager().decrypt_value(self._encrypted_config.header_name)}":
                f"{_CredentialsManager().decrypt_value(self._encrypted_config.header_prefix)} {_CredentialsManager().decrypt_value(self._encrypted_config.api_key)}"  # noqa: E501
        })
