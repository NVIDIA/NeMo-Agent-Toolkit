# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiq.authentication.http_basic_auth.http_basic_auth_exchanger import HTTPBasicAuthExchanger
from aiq.builder.authentication import AuthenticationProviderInfo
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_authentication_client
from aiq.cli.register_workflow import register_authentication_provider
from aiq.data_models.authentication import AuthenticationBaseConfig


class HTTPBasicAuthConfig(AuthenticationBaseConfig, name="http_basic_auth"):
    pass


@register_authentication_provider(config_type=HTTPBasicAuthConfig)
async def api_key(authentication_provider: HTTPBasicAuthConfig, builder: Builder):

    yield AuthenticationProviderInfo(config=authentication_provider,
                                     description="HTTP Basic "
                                     "authentication provider.")


@register_authentication_client(config_type=HTTPBasicAuthConfig)
async def api_key_client(authentication_provider: HTTPBasicAuthConfig, builder: Builder):

    yield HTTPBasicAuthExchanger(authentication_provider)
