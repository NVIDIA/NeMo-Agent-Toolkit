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

from aiq.authentication.authentication_manager import AuthenticationManager
from aiq.authentication.interfaces import RequestManagerBase


class RequestManager(RequestManagerBase):

    def __init__(self, url: str, method: str, headers: dict, params: dict, data: dict) -> None:
        self._url: str = url
        self._method: str = method
        self._headers: dict = headers
        self._params: dict = params
        self._data: dict = data
        self._authentication_manager: AuthenticationManager = AuthenticationManager()

    @property
    def authentication_manager(self) -> AuthenticationManager:
        return self._authentication_manager
