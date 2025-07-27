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

from abc import ABC
from abc import abstractmethod

from aiq.data_models.authentication import AuthenticatedContext
from aiq.data_models.authentication import AuthenticationBaseConfig
from aiq.data_models.authentication import AuthFlowType
from aiq.data_models.authentication import AuthResult

AUTHORIZATION_HEADER = "Authorization"


class AuthenticationClientBase(ABC):
    """
    Base class for authenticating to API services.
    This class provides an interface for authenticating to API services.
    """

    def __init__(self, config: AuthenticationBaseConfig):
        """
        Initialize the AuthenticationClientBase with the given configuration.

        Args:
            config (AuthenticationBaseConfig): Configuration items for authentication.
        """
        self.config = config

    @abstractmethod
    async def authenticate(self, user_id: str | None = None) -> AuthResult:
        """
        Perform the authentication process for the client.

        This method handles the necessary steps to authenticate the client with the
        target API service, which may include obtaining tokens, refreshing credentials,
        or completing multi-step authentication flows.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        # This method will call the frontend FlowHandlerBase `authenticate` method
        pass


class FlowHandlerBase(ABC):
    """
    Handles front-end specifc flows for authentication clients.

    Each front end will define a FlowHandler that will implement the authenticate method.

    The `authenticate` method will be stored as the callback in the AIQContextState.user_auth_callback
    """

    @abstractmethod
    async def authenticate(self, config: AuthenticationBaseConfig, method: AuthFlowType) -> AuthenticatedContext:
        """
        Perform the authentication process for the client.

        This method handles the necessary steps to authenticate the client with the
        target API service, which may include obtaining tokens, refreshing credentials,
        or completing multistep authentication flows.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass
