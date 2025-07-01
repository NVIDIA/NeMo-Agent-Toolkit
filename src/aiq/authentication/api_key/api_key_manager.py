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
import typing

import httpx

from aiq.authentication.interfaces import AuthenticationManagerBase
from aiq.data_models.authentication import HTTPAuthScheme

logger = logging.getLogger(__name__)

if (typing.TYPE_CHECKING):
    from aiq.authentication.api_key.api_key_config import APIKeyConfig


class APIKeyManager(AuthenticationManagerBase):

    def __init__(self, config: "APIKeyConfig", config_name: str | None = None) -> None:
        self._config_name: str | None = config_name
        self._config: "APIKeyConfig" = config
        super().__init__()

    @property
    def config_name(self) -> str | None:
        """
        Get the name of the authentication configuration.

        Returns:
            str | None: The name of the authentication configuration, or None if not set.
        """
        return self._config_name

    @config_name.setter
    def config_name(self, config_name: str | None) -> None:
        """
        Set the name of the authentication configuration.

        Args:
            config_name (str | None): The name of the authentication configuration.
        """
        self._config_name = config_name

    async def validate_credentials(self) -> bool:
        """
        Ensure that the API key credentials are valid for the given API key configuration.

        Returns:
            bool: True if the API key credentials are valid, False otherwise.
        """
        # Validate the API key credentials are set.
        if not self._config.raw_key or self._config.raw_key == "":  # TODO EE: Update
            return False

        return True

    async def construct_authentication_header(self, http_auth_scheme: HTTPAuthScheme) -> httpx.Headers | None:
        if http_auth_scheme == HTTPAuthScheme.BEARER:
            return httpx.Headers({"Authorization": f"Bearer {self._config.raw_key}"})  # TODO EE: Update

    async def construct_authentication_query(self, http_auth_scheme: HTTPAuthScheme) -> httpx.QueryParams | None:
        return None  # TODO EE: Update

    async def construct_authentication_cookie(self, http_auth_scheme: HTTPAuthScheme) -> httpx.Cookies | None:
        return None  # TODO EE: Update

    async def construct_authentication_body(self, http_auth_scheme: HTTPAuthScheme) -> dict[str, typing.Any] | None:
        return None  # TODO EE: Update

    async def construct_authentication_custom(self, http_auth_scheme: HTTPAuthScheme) -> typing.Any | None:
        return None  # TODO EE: Update
