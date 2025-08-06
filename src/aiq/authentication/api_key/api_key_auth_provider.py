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

from pydantic import SecretStr

from aiq.authentication.api_key.api_key_auth_provider_config import APIKeyAuthProviderConfig
from aiq.authentication.interfaces import AuthProviderBase
from aiq.data_models.authentication import AuthResult
from aiq.data_models.authentication import BearerTokenCred
from aiq.data_models.authentication import HeaderAuthScheme

logger = logging.getLogger(__name__)


class APIKeyAuthProvider(AuthProviderBase[APIKeyAuthProviderConfig]):

    # fmt: off
    def __init__(self,
                 config: APIKeyAuthProviderConfig,
                 config_name: str | None = None) -> None:  # pylint: disable=unused-argument
        assert isinstance(config, APIKeyAuthProviderConfig), ("Config is not APIKeyAuthProviderConfig")
        super().__init__(config)
    # fmt: on

    async def _construct_authentication_header(self) -> BearerTokenCred:
        """
        Constructs the authenticated HTTP header based on the authentication scheme.
        Basic Authentication follows the OpenAPI 3.0 Basic Authentication standard as well as RFC 7617.

        Args:
            header_auth_scheme (HeaderAuthScheme): The HTTP authentication scheme to use.
                                             Supported schemes: BEARER, X_API_KEY, BASIC, CUSTOM.

        Returns:
            BearerTokenCred: The HTTP headers containing the authentication credentials.
                             Returns None if the scheme is not supported or configuration is invalid.

        """

        from aiq.authentication.interfaces import AUTHORIZATION_HEADER

        config: APIKeyAuthProviderConfig = self.config

        header_auth_scheme = config.auth_scheme

        if header_auth_scheme == HeaderAuthScheme.BEARER:
            return BearerTokenCred(token=SecretStr(f"{config.raw_key}"),
                                   scheme=HeaderAuthScheme.BEARER.value,
                                   header_name=AUTHORIZATION_HEADER)

        if header_auth_scheme == HeaderAuthScheme.X_API_KEY:
            return BearerTokenCred(token=SecretStr(f"{config.raw_key}"),
                                   scheme=HeaderAuthScheme.X_API_KEY.value,
                                   header_name='')

        if header_auth_scheme == HeaderAuthScheme.CUSTOM:
            if not config.custom_header_name:
                raise ValueError('custom_header_name required when using header_auth_scheme=CUSTOM')

            if not config.custom_header_prefix:
                raise ValueError('custom_header_prefix required when using header_auth_scheme=CUSTOM')

            return BearerTokenCred(token=SecretStr(f"{config.raw_key}"),
                                   scheme=config.custom_header_prefix,
                                   header_name=config.custom_header_name)

        raise ValueError(f"Unsupported header auth scheme: {header_auth_scheme}")

    async def authenticate(self, user_id: str | None = None) -> AuthResult | None:
        """
        Authenticate the user using the API key credentials.

        Args:
            user_id (str): The user ID to authenticate.

        Returns:
            AuthenticatedContext: The authenticated context containing headers, query params, cookies, etc.
        """

        headers = await self._construct_authentication_header()

        return AuthResult(credentials=[headers])
