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
from aiq.authentication.request_manager import RequestManager
from aiq.data_models.authentication import HeaderAuthScheme
from aiq.data_models.authentication import HTTPMethod

logger = logging.getLogger(__name__)

if (typing.TYPE_CHECKING):
    from aiq.authentication.api_key.api_key_config import APIKeyConfig


class APIKeyManager(AuthenticationManagerBase):

    def __init__(self, config: "APIKeyConfig", config_name: str | None = None) -> None:
        self._config_name: str | None = config_name
        self._config: "APIKeyConfig" = config
        self._request_manager: RequestManager = RequestManager()
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
        # Validate the API key credentials are set and non-empty.
        if not self._config.raw_key or self._config.raw_key == "":
            return False

        return True

    async def construct_authentication_header(self,
                                              header_auth_scheme: HeaderAuthScheme = HeaderAuthScheme.BEARER
                                              ) -> httpx.Headers | None:
        """
        Constructs the authenticated HTTP header based on the authentication scheme.

        Args:
            header_auth_scheme (HeaderAuthScheme): The HTTP authentication scheme to use.
                                             Supported schemes: BEARER, X_API_KEY, BASIC, CUSTOM.

        Returns:
            httpx.Headers | None: The constructed HTTP header if successful, otherwise returns None.

        """
        import base64

        from aiq.authentication.interfaces import AUTHORIZATION_HEADER

        if header_auth_scheme == HeaderAuthScheme.BEARER:
            return httpx.Headers({f"{AUTHORIZATION_HEADER}": f"{HeaderAuthScheme.BEARER.value} {self._config.raw_key}"})

        if header_auth_scheme == HeaderAuthScheme.X_API_KEY:
            return httpx.Headers({f"{HeaderAuthScheme.X_API_KEY.value}": f"{self._config.raw_key}"})

        if header_auth_scheme == HeaderAuthScheme.BASIC:
            if not self._config.username or not self._config.password:
                logger.error('Username or password is missing. Please authenticate the provider: %s', self._config_name)
                return None
            token_key: str = f"{self._config.username}:{self._config.password}"
            encoded_key: str = base64.b64encode(token_key.encode("utf-8")).decode("utf-8")
            return httpx.Headers({f"{AUTHORIZATION_HEADER}": f"{HeaderAuthScheme.BASIC.value} {encoded_key}"})

        if header_auth_scheme == HeaderAuthScheme.CUSTOM:
            if not self._config.header_name:
                logger.error('header_name required when using header_auth_scheme CUSTOM: %s', self._config_name)
                return None

            if not self._config.header_prefix:
                logger.error('header_prefix required when using header_auth_scheme CUSTOM: %s', self._config_name)
                return None

            return httpx.Headers(
                {f"{self._config.header_name}": f"{self._config.header_prefix} {self._config.raw_key}"})

        return None

    async def send_request(self,
                           url: str,
                           http_method: str | HTTPMethod,
                           headers: dict | None = None,
                           query_params: dict | None = None,
                           body_data: dict | None = None) -> httpx.Response | None:
        """
        Sends an HTTP request to the specified URL with authentication handling.

        Args:
            url: The target URL for the HTTP request
            http_method: The HTTP method to use (GET, POST, etc.)
            headers: Optional dictionary of HTTP headers to include
            query_params: Optional dictionary of query parameters to include
            body_data: Optional dictionary of request body data

        Returns:
            httpx.Response | None: The HTTP response from the request, or None if the request failed
        """
        return await self._request_manager.send_request(url,
                                                        http_method,
                                                        headers=headers,
                                                        query_params=query_params,
                                                        body_data=body_data)
