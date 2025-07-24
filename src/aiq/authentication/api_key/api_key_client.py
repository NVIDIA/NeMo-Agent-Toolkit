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
from aiq.authentication.interfaces import AuthenticationClientBase
from aiq.data_models.authentication import HeaderAuthScheme, AuthenticatedContext

logger = logging.getLogger(__name__)


class APIKeyClient(AuthenticationClientBase):

    def __init__(self, config: APIKeyConfig, config_name: str | None = None) -> None:
        assert isinstance(config, APIKeyConfig), ("Config is not APIKeyConfig")
        super().__init__(config)

    async def construct_authentication_header(self,
                                              header_auth_scheme: HeaderAuthScheme = HeaderAuthScheme.BEARER
                                              ) -> httpx.Headers | None:
        """
        Constructs the authenticated HTTP header based on the authentication scheme.
        Basic Authentication follows the OpenAPI 3.0 Basic Authentication standard as well as RFC 7617.

        Args:
            header_auth_scheme (HeaderAuthScheme): The HTTP authentication scheme to use.
                                             Supported schemes: BEARER, X_API_KEY, BASIC, CUSTOM.

        Returns:
            httpx.Headers | None: The constructed HTTP header if successful, otherwise returns None.

        """

        from aiq.authentication.interfaces import AUTHORIZATION_HEADER

        if header_auth_scheme == HeaderAuthScheme.BEARER:
            return httpx.Headers({f"{AUTHORIZATION_HEADER}": f"{HeaderAuthScheme.BEARER.value} {self.config.raw_key}"})

        if header_auth_scheme == HeaderAuthScheme.X_API_KEY:
            return httpx.Headers({f"{HeaderAuthScheme.X_API_KEY.value}": f"{self.config.raw_key}"})

        if header_auth_scheme == HeaderAuthScheme.CUSTOM:
            if not self.config.header_name:
                logger.error('header_name required when using header_auth_scheme CUSTOM')
                return None

            if not self.config.header_prefix:
                logger.error('header_prefix required when using header_auth_scheme CUSTOM')
                return None

            return httpx.Headers(
                {f"{self.config.header_name}": f"{self.config.header_prefix} {self.config.raw_key}"})

        return None

    async def authenticate(self, user_id: str) -> AuthenticatedContext:
        """
        Authenticate the user using the API key credentials.

        Args:
            user_id (str): The user ID to authenticate.

        Returns:
            AuthenticatedContext: The authenticated context containing headers, query params, cookies, etc.
        """
        headers = await self.construct_authentication_header(self.config.auth_scheme)
        return AuthenticatedContext(headers=headers)
