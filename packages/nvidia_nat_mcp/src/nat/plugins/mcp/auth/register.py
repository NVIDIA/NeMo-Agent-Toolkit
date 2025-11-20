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

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_auth_provider
from nat.plugins.mcp.auth.auth_provider import MCPOAuth2Provider
from nat.plugins.mcp.auth.auth_provider_config import MCPOAuth2ProviderConfig
from nat.plugins.mcp.auth.service_account.provider import MCPServiceAccountProvider
from nat.plugins.mcp.auth.service_account.provider_config import MCPServiceAccountProviderConfig


@register_auth_provider(config_type=MCPOAuth2ProviderConfig)
async def mcp_oauth2_provider(authentication_provider: MCPOAuth2ProviderConfig, builder: Builder):
    """Register MCP OAuth2 authentication provider with NAT system."""
    yield MCPOAuth2Provider(authentication_provider, builder=builder)


@register_auth_provider(config_type=MCPServiceAccountProviderConfig)
async def mcp_service_account_provider(authentication_provider: MCPServiceAccountProviderConfig, builder: Builder):
    """Register MCP Service Account authentication provider with NAT system."""
    yield MCPServiceAccountProvider(authentication_provider, builder=builder)
