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
from typing import TYPE_CHECKING

from aiq.authentication.interfaces import AuthenticationManagerBase
from aiq.authentication.oauth2_authenticator import OAuth2Authenticator
from aiq.data_models.authentication import OAuth2Config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiq.authentication.request_manager import RequestManager


class AuthenticationManager(AuthenticationManagerBase):

    def __init__(self, request_manager: "RequestManager") -> None:
        self._oauth2_authenticator: OAuth2Authenticator = OAuth2Authenticator(request_manager)

    async def validate_authentication_provider_credentials(self, authentication_provider: str) -> None:
        """
        Validate the authentication provider credentials.

        Args:
            authentication_provider (str): The name of the registered authentication provider.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            if _CredentialsManager()._get_authentication_provider(authentication_provider) is None:
                raise ValueError(f"Authentication provider {authentication_provider} not found.")

            if isinstance(_CredentialsManager()._get_authentication_provider(authentication_provider), OAuth2Config):
                await self._oauth2_authenticator.validate_credentials(authentication_provider)

            raise TypeError(f"Authentication type for {authentication_provider} not supported. Supported types can be "
                            f"found in \"aiq.data_models.authentication\" ")

        except (Exception, ValueError, TypeError) as e:
            logger.error("Falied to validate authentication provider credentials: %s", str(e), exc_info=True)
            return None
