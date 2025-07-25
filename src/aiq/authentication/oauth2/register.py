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

from aiq.builder.authentication import AuthenticationProviderInfo
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_authentication_client
from aiq.cli.register_workflow import register_authentication_provider

from .authorization_code_flow_config import OAuth2AuthorizationCodeFlowConfig
from .client import OAuth2Client


@register_authentication_provider(config_type=OAuth2AuthorizationCodeFlowConfig)
async def oauth2(authentication_provider: OAuth2AuthorizationCodeFlowConfig, builder: Builder):
    yield AuthenticationProviderInfo(config=authentication_provider, description="OAuth 2.0 authentication provider.")


@register_authentication_client(config_type=OAuth2AuthorizationCodeFlowConfig)
async def oauth2_client(authentication_provider: OAuth2AuthorizationCodeFlowConfig, builder: Builder):
    yield OAuth2Client(authentication_provider)
