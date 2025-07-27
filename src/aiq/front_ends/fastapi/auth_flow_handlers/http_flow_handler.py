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

from aiq.authentication.interfaces import FlowHandlerBase
from aiq.data_models.authentication import AuthenticatedContext
from aiq.data_models.authentication import AuthFlowType
from aiq.data_models.authentication import AuthProviderBaseConfig


class HTTPAuthenticationFlowHandler(FlowHandlerBase):

    async def authenticate(self, config: AuthProviderBaseConfig, method: AuthFlowType) -> AuthenticatedContext:

        raise NotImplementedError(f"Authentication method '{method}' is not supported by the HTTP frontend."
                                  f" Do you have Websockets enabled?")
