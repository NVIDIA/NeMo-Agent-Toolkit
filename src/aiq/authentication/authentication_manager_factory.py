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

from aiq.authentication.api_key.api_key_config import APIKeyConfig
from aiq.authentication.api_key.api_key_manager import APIKeyManager
from aiq.authentication.interfaces import AuthenticationManagerBase
from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig
from aiq.authentication.oauth2.auth_code_grant_manager import AuthCodeGrantClientManager
from aiq.data_models.api_server import AuthenticatedRequest
from aiq.data_models.authentication import ExecutionMode

logger = logging.getLogger(__name__)


class AuthenticationManagerFactory:

    def __init__(self, execution_mode: ExecutionMode) -> None:
        self._execution_mode: ExecutionMode = execution_mode

    async def create(self, user_request: AuthenticatedRequest) -> AuthenticationManagerBase | None:
        """
        Create an instance of the appropriate authentication manager based on the configuration provided.

        Args:
            user_request (AuthenticatedRequest | None): Authentication manager configuration model.

        Returns:
            AuthenticationManagerBase | None: Authentication manager instance or None if the config is not recognized.
        """
        if user_request.authentication_config_name is None or user_request.authentication_config is None:
            return None

        if isinstance(user_request.authentication_config, AuthCodeGrantConfig):
            return AuthCodeGrantClientManager(config_name=user_request.authentication_config_name,
                                              encrypted_config=user_request.authentication_config,
                                              execution_mode=self._execution_mode)

        if isinstance(user_request.authentication_config, APIKeyConfig):
            return APIKeyManager(config_name=user_request.authentication_config_name,
                                 encrypted_config=user_request.authentication_config)

        return None
