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

from aiq.authentication.interfaces import AuthenticationManagerBase
from aiq.authentication.oauth2_authenticator import OAuth2Authenticator
from aiq.data_models.authentication import AuthenticationProvider
from aiq.data_models.authentication import OAuth2Config


class AuthenticationManager(AuthenticationManagerBase):

    def __init__(self) -> None:
        self._oauth2_authenticator: OAuth2Authenticator = OAuth2Authenticator()

    async def validate_authentication_provider_credentials(self, authentication_provider: str) -> None:
        """
        Validate the authentication credentials.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        authentication_credentials: AuthenticationProvider | None = _CredentialsManager()._get_authentication_provider(
            authentication_provider)

        if authentication_credentials is None:  # TODO EE: Update return type
            raise ValueError(f"Authentication provider {authentication_provider} not found.")

        if isinstance(authentication_credentials, OAuth2Config):
            await self._oauth2_authenticator.validate_credentials(authentication_credentials)
