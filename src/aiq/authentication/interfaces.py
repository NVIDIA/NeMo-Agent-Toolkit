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

import httpx


class RequestManagerBase:
    """
    Base class for handling API requests.
    This class provides an interface for making API requests.
    """
    pass


class ResponseManagerBase:
    """
    Base class for handling API responses.
    This class provides an interface for handling API responses.
    """
    pass


class AuthenticationManagerBase(ABC):
    """
    Base class for authenticating to API services.
    This class provides an interface for authenticating to API services.
    """

    @abstractmethod
    async def validate_authentication_credentials(self) -> bool:
        pass

    @abstractmethod
    async def get_authentication_header(self) -> httpx.Headers | None:
        pass

    @abstractmethod
    async def construct_authentication_header(self) -> httpx.Headers | None:
        pass


class OAuthClientBase(ABC):
    """
    Base class for managing OAuth clients.
    This class provides an interface for managing OAuth clients.
    """

    @abstractmethod
    async def _send_authorization_request(self) -> httpx.URL:
        pass

    @abstractmethod
    async def _initiate_authorization_code_flow_console(self) -> None:
        pass

    @abstractmethod
    async def _initiate_authorization_code_flow_server(self) -> None:
        pass
