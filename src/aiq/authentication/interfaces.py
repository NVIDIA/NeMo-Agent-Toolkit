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


class AuthenticationBase(ABC):
    """
    Base class for authenticating to API services.
    This class provides an interface for authenticating to API services.
    """

    @abstractmethod
    async def _validate_credentials(self) -> bool:
        pass

    @abstractmethod
    async def _get_credentials(self) -> bool:
        pass
